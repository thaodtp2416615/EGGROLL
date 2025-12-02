"""
EGGROLL Initialization for Translation Model Finetuning (PyTorch Implementation)

Based on the official implementation from:
https://github. com/ESHyperscale/HyperscaleES/blob/main/llm_experiments/general_do_evolution_multi_gpu.py

This module implements Step 1: Initialization of the EGGROLL algorithm
for finetuning a Transformer-based translation model. 
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Literal
from pathlib import Path
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datetime import datetime


# ============================================================================
# Configuration Dataclass (adapted from Args in original code)
# ============================================================================

@dataclass
class EggrollConfig:
    """
    Configuration for EGGROLL finetuning. 
    
    Hyperparameters following the paper "Evolution Strategies at Hyperscale"
    """
    # Random seed for reproducibility
    seed: int = 0
    
    # Model configuration
    model_name: str = "Helsinki-NLP/opus-mt-en-vi"  # Translation model
    dtype: Optional[str] = "float32"  # "float32", "float16", "bfloat16"
    
    # EGGROLL core hyperparameters
    sigma: float = 1e-3              # Noise standard deviation (σ)
    lr_scale: float = 1.0            # Learning rate scale (α)
    rank: int = 16                   # Low-rank dimension (r << d)
    
    # Population and batch settings
    population_size: int = 64        # N: Number of perturbed models per iteration
    generations_per_prompt: int = 8  # Number of generations per unique prompt
    
    # Training settings
    num_epochs: int = 100
    validate_every: int = 10
    save_every: int = 100
    log_output_every: int = 10
    
    # Generation settings
    generation_length: int = 100
    thinking_length: int = 100
    answer_length: int = 100
    temperature: float = 0.0
    val_temperature: float = 0.0
    
    # Noiser settings
    noise_reuse: int = 1             # Number of times to reuse noise
    freeze_nonlora: bool = True      # Whether to freeze non-LoRA parameters
    
    # Paths
    output_directory: Optional[str] = "."
    save_path: Optional[str] = "."
    load_path: Optional[str] = None
    save_model: bool = True
    load_model: bool = False
    
    # Device settings
    device: str = "cuda" if torch.cuda. is_available() else "cpu"
    num_gpus: int = 1
    parallel_generations_per_gpu: int = 32


# ============================================================================
# Optimizer State (adapted from optax pattern)
# ============================================================================

@dataclass
class OptimizerState:
    """State for the optimizer (similar to optax optimizer state)"""
    step: int = 0
    momentum: Optional[Dict[str, torch.Tensor]] = None
    velocity: Optional[Dict[str, torch. Tensor]] = None


# ============================================================================
# EGGROLL Noiser Parameters
# ============================================================================

@dataclass
class FrozenNoiserParams:
    """
    Frozen (constant) parameters for EGGROLL noiser.
    These don't change during training. 
    """
    group_size: int                  # Number of samples per group for normalization
    freeze_nonlora: bool             # Whether to freeze non-LoRA parameters
    noise_reuse: int                 # Number of times to reuse noise
    rank: int                        # Low-rank dimension r
    solver_type: str = "sgd"         # Optimizer type: "sgd" or "adam"
    solver_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class NoiserParams:
    """
    Mutable parameters for EGGROLL noiser. 
    These are updated during training. 
    """
    sigma: float                     # Current noise standard deviation
    opt_state: OptimizerState        # Optimizer state


# ============================================================================
# Parameter Classification Map (for ES updates)
# ============================================================================

class ESMapType:
    """Classification of parameters for ES updates"""
    FULL = 0      # Full parameter update (non-LoRA)
    LORA = 1      # LoRA-style low-rank update
    FROZEN = 2    # Frozen, no update
    NOOP = 3      # No operation


# ============================================================================
# EGGROLL Noiser Class (PyTorch Implementation)
# ============================================================================

class EggrollNoiser:
    """
    EGGROLL Noiser for Evolution Strategies.
    
    Implements low-rank perturbations for memory-efficient ES on large models.
    
    Key idea: Instead of storing full perturbation matrix ε of size (d x d),
    we store two smaller matrices A (d x r) and B (d x r) where r << d,
    such that ε ≈ A @ B. T
    """
    
    @classmethod
    def init_noiser(
        cls,
        params: Dict[str, torch.Tensor],
        sigma: float,
        lr: float,
        solver: str = "sgd",
        solver_kwargs: Optional[Dict[str, Any]] = None,
        group_size: int = 0,
        freeze_nonlora: bool = False,
        noise_reuse: int = 0,
        rank: int = 1,
        device: str = "cuda"
    ) -> Tuple[FrozenNoiserParams, NoiserParams]:
        """
        Initialize the EGGROLL noiser. 
        
        Adapted from EggRoll. init_noiser in the original JAX implementation.
        
        Args:
            params: Dictionary of model parameters {name: tensor}
            sigma: Noise standard deviation (σ)
            lr: Learning rate (α)
            solver: Optimizer type ("sgd" or "adam")
            solver_kwargs: Additional optimizer arguments
            group_size: Number of samples per group for fitness normalization
            freeze_nonlora: Whether to freeze non-LoRA parameters
            noise_reuse: Number of times to reuse noise patterns
            rank: Low-rank dimension r for perturbations
            device: Device to use for tensors
            
        Returns:
            frozen_noiser_params: Constant noiser configuration
            noiser_params: Mutable noiser state
        """
        if solver_kwargs is None:
            solver_kwargs = {}
        
        # Initialize optimizer state (momentum/velocity buffers)
        opt_state = cls._init_optimizer_state(params, solver, device)
        
        # Create frozen parameters (constants)
        frozen_noiser_params = FrozenNoiserParams(
            group_size=group_size,
            freeze_nonlora=freeze_nonlora,
            noise_reuse=noise_reuse,
            rank=rank,
            solver_type=solver,
            solver_kwargs={"lr": lr, **solver_kwargs}
        )
        
        # Create mutable noiser parameters
        noiser_params = NoiserParams(
            sigma=sigma,
            opt_state=opt_state
        )
        
        return frozen_noiser_params, noiser_params
    
    @classmethod
    def _init_optimizer_state(
        cls,
        params: Dict[str, torch.Tensor],
        solver: str,
        device: str
    ) -> OptimizerState:
        """Initialize optimizer state (momentum/velocity buffers)"""
        opt_state = OptimizerState(step=0)
        
        if solver == "adam":
            # Adam needs both momentum and velocity
            opt_state.momentum = {
                name: torch.zeros_like(p, device=device)
                for name, p in params.items()
            }
            opt_state.velocity = {
                name: torch.zeros_like(p, device=device)
                for name, p in params.items()
            }
        elif solver == "sgd_momentum":
            # SGD with momentum only needs momentum buffer
            opt_state.momentum = {
                name: torch. zeros_like(p, device=device)
                for name, p in params.items()
            }
            
        return opt_state


# ============================================================================
# Parameter ES Map Builder
# ============================================================================

def build_es_map(
    model: nn.Module,
    freeze_embeddings: bool = True,
    freeze_layer_norm: bool = True,
    lora_target_modules: Optional[list] = None
) -> Dict[str, int]:
    """
    Build ES classification map for model parameters.
    
    Determines which parameters get:
    - FULL updates (standard ES)
    - LORA updates (low-rank ES via EGGROLL)
    - FROZEN (no updates)
    
    Args:
        model: The neural network model
        freeze_embeddings: Whether to freeze embedding layers
        freeze_layer_norm: Whether to freeze layer normalization
        lora_target_modules: List of module names to apply LoRA updates
        
    Returns:
        es_map: Dictionary mapping parameter names to ESMapType
    """
    if lora_target_modules is None:
        # Default: apply LoRA to attention and FFN weight matrices
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "out_proj",  # Attention
            "fc1", "fc2",  # FFN
            "self_attn", "encoder_attn",  # For MarianMT / OPUS models
        ]
    
    es_map = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            es_map[name] = ESMapType. FROZEN
            continue
            
        # Check if embedding layer
        if "embed" in name. lower():
            es_map[name] = ESMapType. FROZEN if freeze_embeddings else ESMapType. FULL
            continue
            
        # Check if layer norm
        if "layer_norm" in name.lower() or "layernorm" in name.lower():
            es_map[name] = ESMapType.FROZEN if freeze_layer_norm else ESMapType. FULL
            continue
            
        # Check if bias (biases typically get full updates or frozen)
        if "bias" in name. lower():
            es_map[name] = ESMapType. FULL
            continue
            
        # Check if target for LoRA updates (weight matrices)
        is_lora_target = any(target in name.lower() for target in lora_target_modules)
        if is_lora_target and "weight" in name.lower():
            es_map[name] = ESMapType.LORA
        else:
            es_map[name] = ESMapType. FULL
            
    return es_map


# ============================================================================
# Random Key Generation (similar to JAX's key splitting)
# ============================================================================

class RandomKeyGenerator:
    """
    Random key generator for reproducible noise generation.
    Similar to JAX's random key system. 
    """
    
    def __init__(self, seed: int):
        self.base_seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
    def fold_in(self, key_id: int) -> 'RandomKeyGenerator':
        """Create a new key by folding in an additional integer"""
        new_gen = RandomKeyGenerator(self. base_seed)
        new_gen.generator.manual_seed(self.base_seed + key_id * 31337)
        return new_gen
    
    def split(self, num_keys: int) -> list:
        """Split into multiple independent keys"""
        return [self.fold_in(i) for i in range(num_keys)]
    
    @property
    def seed(self) -> int:
        """Get current seed for this key"""
        return self. base_seed


def simple_es_tree_key(
    params: Dict[str, torch.Tensor],
    base_key: RandomKeyGenerator,
    scan_map: Optional[Dict[str, int]] = None
) -> Dict[str, RandomKeyGenerator]:
    """
    Generate a tree of random keys matching the parameter structure.
    
    Each parameter gets its own independent random key for noise generation.
    
    Args:
        params: Dictionary of model parameters
        base_key: Base random key generator
        scan_map: Optional scan map (for scanned/stacked layers)
        
    Returns:
        Dictionary mapping parameter names to random keys
    """
    keys = {}
    for i, name in enumerate(params. keys()):
        keys[name] = base_key.fold_in(i)
    return keys


# ============================================================================
# Main Initialization Function
# ============================================================================

def initialize_eggroll(
    config: EggrollConfig
) -> Tuple[
    nn.Module,                      # model
    Dict[str, torch.Tensor],        # params
    FrozenNoiserParams,             # frozen_noiser_params  
    NoiserParams,                   # noiser_params
    Dict[str, int],                 # es_map
    Dict[str, RandomKeyGenerator],  # base_evo_keys
    Any,                            # tokenizer
]:
    """
    Initialize EGGROLL for finetuning a translation model.
    
    This implements Step 1 of the EGGROLL algorithm as described in
    "Evolution Strategies at Hyperscale" (arXiv:2511.16652). 
    
    Args:
        config: EGGROLL configuration
        
    Returns:
        model: Pre-trained translation model
        params: Dictionary of model parameters
        frozen_noiser_params: Frozen noiser configuration
        noiser_params: Mutable noiser state
        es_map: Parameter classification for ES updates
        base_evo_keys: Random keys for each parameter
        tokenizer: Model tokenizer
    """
    print("=" * 60)
    print("EGGROLL Initialization")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. Set random seeds for reproducibility
    # -------------------------------------------------------------------------
    print(f"\n[1] Setting random seed: {config.seed}")
    torch.manual_seed(config. seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    master_key = RandomKeyGenerator(config.seed)
    base_model_key = master_key.fold_in(0)
    base_gen_key = master_key.fold_in(1)
    base_valid_key = master_key.fold_in(2)
    
    # -------------------------------------------------------------------------
    # 2. Load pre-trained translation model
    # -------------------------------------------------------------------------
    print(f"\n[2] Loading pre-trained model: {config.model_name}")
    
    # Determine dtype
    dtype_map = {
        "float32": torch. float32,
        "float16": torch.float16,
        "bfloat16": torch. bfloat16,
    }
    torch_dtype = dtype_map.get(config.dtype, torch. float32)
    
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
    ). to(config.device)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    print(f"   Model loaded: {model.__class__.__name__}")
    print(f"   Device: {config.device}")
    print(f"   Dtype: {torch_dtype}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model. parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # -------------------------------------------------------------------------
    # 3. Extract parameters as dictionary
    # -------------------------------------------------------------------------
    print(f"\n[3] Extracting model parameters")
    
    params = {name: param. data.clone() for name, param in model.named_parameters()}
    print(f"   Number of parameter tensors: {len(params)}")
    
    # -------------------------------------------------------------------------
    # 4. Build ES classification map
    # -------------------------------------------------------------------------
    print(f"\n[4] Building ES parameter classification map")
    
    es_map = build_es_map(
        model,
        freeze_embeddings=True,
        freeze_layer_norm=True,
    )
    
    # Count by type
    type_counts = {
        "FULL": sum(1 for v in es_map. values() if v == ESMapType. FULL),
        "LORA": sum(1 for v in es_map. values() if v == ESMapType. LORA),
        "FROZEN": sum(1 for v in es_map. values() if v == ESMapType. FROZEN),
    }
    print(f"   FULL updates: {type_counts['FULL']} parameters")
    print(f"   LORA updates: {type_counts['LORA']} parameters")
    print(f"   FROZEN: {type_counts['FROZEN']} parameters")
    
    # -------------------------------------------------------------------------
    # 5.  Initialize EGGROLL Noiser
    # -------------------------------------------------------------------------
    print(f"\n[5] Initializing EGGROLL Noiser")
    print(f"   Sigma (σ): {config. sigma}")
    print(f"   Learning rate scale (α): {config.lr_scale}")
    print(f"   Low-rank dimension (r): {config. rank}")
    print(f"   Population size (N): {config.population_size}")
    print(f"   Freeze non-LoRA: {config.freeze_nonlora}")
    
    frozen_noiser_params, noiser_params = EggrollNoiser. init_noiser(
        params=params,
        sigma=config.sigma,
        lr=config.lr_scale,
        solver="sgd",
        group_size=config. generations_per_prompt,
        freeze_nonlora=config. freeze_nonlora,
        noise_reuse=config. noise_reuse,
        rank=config.rank,
        device=config.device,
    )
    
    # -------------------------------------------------------------------------
    # 6. Generate random keys for each parameter
    # -------------------------------------------------------------------------
    print(f"\n[6] Generating random keys for parameters")
    
    base_evo_keys = simple_es_tree_key(params, base_model_key)
    print(f"   Generated {len(base_evo_keys)} random keys")
    
    # -------------------------------------------------------------------------
    # 7.  Compute derived settings
    # -------------------------------------------------------------------------
    print(f"\n[7] Computing derived settings")
    
    config.generation_length = config. thinking_length + config.answer_length
    
    total_devices = config.num_gpus
    config.total_parallel_generations = total_devices * config. parallel_generations_per_gpu
    config.prompts_per_epoch = config.total_parallel_generations // config.generations_per_prompt
    
    print(f"   Total parallel generations: {config. total_parallel_generations}")
    print(f"   Prompts per epoch: {config.prompts_per_epoch}")
    print(f"   Generation length: {config. generation_length}")
    
    # -------------------------------------------------------------------------
    # 8. Optional: Load from checkpoint
    # -------------------------------------------------------------------------
    if config.load_model and config.load_path:
        print(f"\n[8] Loading checkpoint from: {config.load_path}")
        checkpoint = torch.load(config.load_path, map_location=config.device)
        
        if "params" in checkpoint:
            params = checkpoint["params"]
            model.load_state_dict({k: v for k, v in params. items()})
        if "noiser_params" in checkpoint:
            noiser_params = checkpoint["noiser_params"]
        if "es_map" in checkpoint:
            es_map = checkpoint["es_map"]
            
        print("   Checkpoint loaded successfully")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EGGROLL Initialization Complete!")
    print("=" * 60)
    print(f"""
Summary:
--------
• Model: {config. model_name}
• Parameters: {total_params:,} total, {trainable_params:,} trainable
• EGGROLL Config:
  - σ (sigma): {config.sigma}
  - α (learning rate): {config.lr_scale}
  - r (rank): {config.rank}
  - N (population): {config.population_size}
• ES Map: {type_counts['LORA']} LoRA, {type_counts['FULL']} Full, {type_counts['FROZEN']} Frozen
• Device: {config.device}
""")
    
    return (
        model,
        params,
        frozen_noiser_params,
        noiser_params,
        es_map,
        base_evo_keys,
        tokenizer,
    )


# ============================================================================
# Utility: Save/Load Checkpoints
# ============================================================================

def save_checkpoint(
    path: str,
    params: Dict[str, torch.Tensor],
    frozen_noiser_params: FrozenNoiserParams,
    noiser_params: NoiserParams,
    es_map: Dict[str, int],
    epoch: int = 0,
):
    """Save EGGROLL checkpoint"""
    Path(path).parent. mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "params": params,
        "frozen_noiser_params": frozen_noiser_params,
        "noiser_params": noiser_params,
        "es_map": es_map,
        "epoch": epoch,
        "timestamp": datetime.now(). isoformat(),
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(path: str, device: str = "cuda"):
    """Load EGGROLL checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create configuration
    config = EggrollConfig(
        # Model
        model_name="Helsinki-NLP/opus-mt-en-vi",  # English to Vietnamese
        
        # EGGROLL hyperparameters (from paper)
        sigma=1e-3,           # Noise standard deviation
        lr_scale=1.0,         # Learning rate
        rank=16,              # Low-rank dimension (r << d)
        
        # Population settings
        population_size=64,   # N: number of perturbed models
        generations_per_prompt=8,
        
        # Training
        num_epochs=100,
        freeze_nonlora=True,  # Only update LoRA-style parameters
        
        # Device
        device="cuda" if torch.cuda. is_available() else "cpu",
    )
    
    # Initialize EGGROLL
    (
        model,
        params,
        frozen_noiser_params,
        noiser_params,
        es_map,
        base_evo_keys,
        tokenizer,
    ) = initialize_eggroll(config)
    
    # Now ready for Step 2: Low-Rank Perturbation Generation
    print("\nReady for Step 2: Generate Low-Rank Perturbations!")
    print(f"Next: For each of the {config.population_size} population members:")
    print(f"  - Sample A_i of size (d × {config.rank})")
    print(f"  - Sample B_i of size (d × {config. rank})")
    print(f"  - Apply perturbation: θ_i = θ + σ(A_i × B_i^T)")
