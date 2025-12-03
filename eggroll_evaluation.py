"""
EGGROLL Step 4: Evaluation / Compute Rewards

This module computes rewards (fitness scores) for translations generated
by each population member. 

Key advantage of EGGROLL: Can directly optimize non-differentiable metrics
like BLEU, METEOR, COMET, or even human preference scores! 

Based on: https://github.com/ESHyperscale/HyperscaleES
Paper: "Evolution Strategies at Hyperscale" (arXiv:2511.16652)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

# Suppress tokenizer warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from previous steps
from eggroll_forward_pass import GenerationResult


# ============================================================================
# Reward Data Structures
# ============================================================================

@dataclass
class RewardResult:
    """Reward computation result for a single sample"""
    member_index: int
    sample_index: int
    input_text: str
    hypothesis: str  # Generated translation
    reference: str   # Ground truth translation
    reward: float    # Computed reward score
    metric_name: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PopulationRewards:
    """Aggregated rewards for entire population"""
    epoch: int
    rewards: Dict[int, List[float]]  # member_idx -> list of rewards
    mean_rewards: Dict[int, float]   # member_idx -> mean reward
    aggregated_rewards: np.ndarray   # Shape: (population_size,) - for ES update
    
    # Statistics
    best_member_idx: int
    best_reward: float
    worst_member_idx: int
    worst_reward: float
    mean_reward: float
    std_reward: float
    
    # Detailed results
    detailed_results: Optional[Dict[int, List[RewardResult]]] = None


# ============================================================================
# Abstract Reward Function Interface
# ============================================================================

class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    
    EGGROLL can use ANY reward function, including:
    - BLEU score
    - METEOR
    - COMET
    - BERTScore
    - Human preference scores
    - Custom task-specific metrics
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the reward metric"""
        pass
    
    @property
    def higher_is_better(self) -> bool:
        """Whether higher scores are better (default: True)"""
        return True
    
    @abstractmethod
    def compute(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> float:
        """
        Compute reward for a single hypothesis-reference pair.
        
        Args:
            hypothesis: Generated translation
            reference: Ground truth translation
            source: Optional source text (needed for some metrics like COMET)
            
        Returns:
            Reward score (higher is better if higher_is_better=True)
        """
        pass
    
    def compute_batch(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Compute rewards for a batch of samples.
        
        Default implementation calls compute() for each sample.
        Override for batch-optimized implementations.
        """
        if sources is None:
            sources = [None] * len(hypotheses)
            
        return [
            self.compute(hyp, ref, src)
            for hyp, ref, src in zip(hypotheses, references, sources)
        ]


# ============================================================================
# BLEU Score Implementation
# ============================================================================

class BLEUReward(RewardFunction):
    """
    BLEU (Bilingual Evaluation Understudy) score reward. 
    
    Most common metric for machine translation evaluation.
    Measures n-gram precision with brevity penalty.
    """
    
    def __init__(
        self,
        smoothing: bool = True,
        max_ngram: int = 4,
        weights: Optional[Tuple[float, ... ]] = None,
    ):
        """
        Initialize BLEU reward. 
        
        Args:
            smoothing: Whether to use smoothing for short sentences
            max_ngram: Maximum n-gram order (typically 4)
            weights: Weights for each n-gram order (default: uniform)
        """
        self.smoothing = smoothing
        self.max_ngram = max_ngram
        self. weights = weights or tuple([1.0 / max_ngram] * max_ngram)
        
        # Try to import sacrebleu for better BLEU computation
        try:
            import sacrebleu
            self._use_sacrebleu = True
            self._sacrebleu = sacrebleu
        except ImportError:
            self._use_sacrebleu = False
            warnings.warn(
                "sacrebleu not installed. Using nltk BLEU.  "
                "Install with: pip install sacrebleu"
            )
            
    @property
    def name(self) -> str:
        return "BLEU"
    
    def compute(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> float:
        """Compute sentence-level BLEU score."""
        if not hypothesis. strip() or not reference.strip():
            return 0.0
            
        if self._use_sacrebleu:
            return self._compute_sacrebleu(hypothesis, reference)
        else:
            return self._compute_nltk(hypothesis, reference)
    
    def _compute_sacrebleu(self, hypothesis: str, reference: str) -> float:
        """Compute BLEU using sacrebleu library."""
        try:
            # sentence_bleu returns a BLEU object with score attribute
            bleu = self._sacrebleu. sentence_bleu(
                hypothesis,
                [reference],
                smooth_method='exp' if self.smoothing else 'none',
            )
            return bleu.score / 100.0  # Normalize to [0, 1]
        except Exception as e:
            warnings.warn(f"sacrebleu error: {e}")
            return 0. 0
    
    def _compute_nltk(self, hypothesis: str, reference: str) -> float:
        """Compute BLEU using nltk library."""
        try:
            from nltk.translate. bleu_score import sentence_bleu, SmoothingFunction
            
            # Tokenize
            hyp_tokens = hypothesis.lower().split()
            ref_tokens = reference.lower().split()
            
            if len(hyp_tokens) == 0:
                return 0.0
                
            # Compute BLEU
            if self.smoothing:
                smoothing = SmoothingFunction(). method1
            else:
                smoothing = SmoothingFunction(). method0
                
            score = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=self.weights,
                smoothing_function=smoothing,
            )
            return score
        except ImportError:
            warnings.warn("nltk not installed.  Returning 0.")
            return 0.0
        except Exception as e:
            warnings.warn(f"BLEU computation error: {e}")
            return 0.0
    
    def compute_batch(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[float]:
        """Compute BLEU for batch of samples."""
        return [
            self. compute(hyp, ref)
            for hyp, ref in zip(hypotheses, references)
        ]


# ============================================================================
# METEOR Score Implementation
# ============================================================================

class METEORReward(RewardFunction):
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) score.
    
    Considers synonyms, stemming, and word order in addition to exact matches.
    Often correlates better with human judgment than BLEU. 
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize METEOR reward. 
        
        Args:
            language: Target language for stemming/synonyms
        """
        self.language = language
        
        try:
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            from nltk.translate.meteor_score import meteor_score
            self._meteor_score = meteor_score
            self._available = True
        except ImportError:
            self._available = False
            warnings.warn(
                "nltk not installed.  METEOR will return 0.  "
                "Install with: pip install nltk"
            )
            
    @property
    def name(self) -> str:
        return "METEOR"
    
    def compute(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> float:
        """Compute METEOR score."""
        if not self._available:
            return 0.0
            
        if not hypothesis.strip() or not reference.strip():
            return 0.0
            
        try:
            # Tokenize
            hyp_tokens = hypothesis.lower().split()
            ref_tokens = reference.lower(). split()
            
            score = self._meteor_score([ref_tokens], hyp_tokens)
            return score
        except Exception as e:
            warnings. warn(f"METEOR error: {e}")
            return 0.0


# ============================================================================
# chrF Score Implementation
# ============================================================================

class ChrFReward(RewardFunction):
    """
    chrF (character F-score) metric. 
    
    Character-level metric that works well for morphologically rich languages. 
    Less sensitive to tokenization than BLEU. 
    """
    
    def __init__(self, char_order: int = 6, word_order: int = 2, beta: float = 2.0):
        """
        Initialize chrF reward.
        
        Args:
            char_order: Maximum character n-gram order
            word_order: Maximum word n-gram order (0 for chrF, 2 for chrF++)
            beta: Importance of recall vs precision
        """
        self.char_order = char_order
        self.word_order = word_order
        self.beta = beta
        
        try:
            import sacrebleu
            self._sacrebleu = sacrebleu
            self._available = True
        except ImportError:
            self._available = False
            warnings.warn("sacrebleu not installed. chrF will return 0.")
            
    @property
    def name(self) -> str:
        return f"chrF{'++' if self.word_order > 0 else ''}"
    
    def compute(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> float:
        """Compute chrF score."""
        if not self._available:
            return 0.0
            
        if not hypothesis.strip() or not reference.strip():
            return 0.0
            
        try:
            chrf = self._sacrebleu.sentence_chrf(
                hypothesis,
                [reference],
                char_order=self. char_order,
                word_order=self.word_order,
                beta=self.beta,
            )
            return chrf.score / 100.0  # Normalize to [0, 1]
        except Exception as e:
            warnings.warn(f"chrF error: {e}")
            return 0.0


# ============================================================================
# COMET Score Implementation (Neural Metric)
# ============================================================================

class COMETReward(RewardFunction):
    """
    COMET (Crosslingual Optimized Metric for Evaluation of Translation). 
    
    Neural metric that often correlates best with human judgment. 
    Requires source text for computation.
    
    Note: Requires GPU for efficient computation.
    """
    
    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 8,
        device: str = "cuda",
    ):
        """
        Initialize COMET reward.
        
        Args:
            model_name: COMET model to use
            batch_size: Batch size for inference
            device: Device for model
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        
        try:
            from comet import download_model, load_from_checkpoint
            self._download_model = download_model
            self._load_from_checkpoint = load_from_checkpoint
            self._available = True
        except ImportError:
            self._available = False
            warnings.warn(
                "COMET not installed. Install with: pip install unbabel-comet"
            )
            
    @property
    def name(self) -> str:
        return "COMET"
    
    def _load_model(self):
        """Lazy load COMET model."""
        if self._model is None and self._available:
            model_path = self._download_model(self.model_name)
            self._model = self._load_from_checkpoint(model_path)
            self._model.to(self.device)
            self._model.eval()
    
    def compute(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> float:
        """Compute COMET score (requires source)."""
        if not self._available:
            return 0. 0
            
        if source is None:
            warnings.warn("COMET requires source text.  Returning 0.")
            return 0.0
            
        scores = self. compute_batch([hypothesis], [reference], [source])
        return scores[0]
    
    def compute_batch(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[float]:
        """Compute COMET for batch (more efficient)."""
        if not self._available or sources is None:
            return [0. 0] * len(hypotheses)
            
        self._load_model()
        
        try:
            data = [
                {"src": src, "mt": hyp, "ref": ref}
                for src, hyp, ref in zip(sources, hypotheses, references)
            ]
            
            with torch.no_grad():
                output = self._model. predict(
                    data,
                    batch_size=self.batch_size,
                    gpus=1 if self.device == "cuda" else 0,
                )
                
            return output.scores
        except Exception as e:
            warnings.warn(f"COMET error: {e}")
            return [0.0] * len(hypotheses)


# ============================================================================
# Length Ratio Reward (Simple Baseline)
# ============================================================================

class LengthRatioReward(RewardFunction):
    """
    Simple reward based on length ratio between hypothesis and reference. 
    
    Useful for debugging or as a baseline.
    Reward = 1 - |len(hyp)/len(ref) - 1| (capped at 0)
    """
    
    @property
    def name(self) -> str:
        return "LengthRatio"
    
    def compute(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> float:
        """Compute length ratio reward."""
        if not reference.strip():
            return 0.0
            
        hyp_len = len(hypothesis.split())
        ref_len = len(reference.split())
        
        if ref_len == 0:
            return 0.0
            
        ratio = hyp_len / ref_len
        reward = max(0.0, 1.0 - abs(ratio - 1.0))
        
        return reward


# ============================================================================
# Composite Reward (Combine Multiple Metrics)
# ============================================================================

class CompositeReward(RewardFunction):
    """
    Combine multiple reward functions with weights.
    
    Useful for multi-objective optimization:
    reward = w1*BLEU + w2*METEOR + w3*Length
    """
    
    def __init__(
        self,
        rewards: List[Tuple[RewardFunction, float]],
        normalize: bool = True,
    ):
        """
        Initialize composite reward.
        
        Args:
            rewards: List of (reward_function, weight) tuples
            normalize: Whether to normalize weights to sum to 1
        """
        self.rewards = rewards
        
        if normalize:
            total_weight = sum(w for _, w in rewards)
            self.rewards = [(r, w / total_weight) for r, w in rewards]
            
    @property
    def name(self) -> str:
        names = [f"{w:. 1f}*{r. name}" for r, w in self.rewards]
        return f"Composite({'+'.join(names)})"
    
    def compute(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> float:
        """Compute weighted sum of rewards."""
        total = 0.0
        for reward_fn, weight in self. rewards:
            score = reward_fn. compute(hypothesis, reference, source)
            total += weight * score
        return total


# ============================================================================
# Reward Aggregator for Population
# ============================================================================

class PopulationRewardAggregator:
    """
    Aggregates rewards across population for ES updates.
    
    Handles:
    1. Computing rewards for all population members
    2.  Aggregating per-sample rewards into per-member rewards
    3.  Fitness shaping (normalization) for ES
    """
    
    def __init__(
        self,
        reward_function: RewardFunction,
        aggregation: str = "mean",  # "mean", "sum", "min", "max"
        fitness_shaping: str = "centered_rank",  # "none", "standardize", "centered_rank"
    ):
        """
        Initialize aggregator.
        
        Args:
            reward_function: Reward function to use
            aggregation: How to aggregate per-sample rewards into per-member reward
            fitness_shaping: How to shape fitness for ES update
        """
        self.reward_function = reward_function
        self.aggregation = aggregation
        self.fitness_shaping = fitness_shaping
        
    def compute_rewards(
        self,
        generation_results: Dict[int, List[GenerationResult]],
        references: List[str],
        sources: Optional[List[str]] = None,
        epoch: int = 0,
    ) -> PopulationRewards:
        """
        Compute rewards for entire population.
        
        Args:
            generation_results: {member_idx: [GenerationResult, ...]}
            references: Ground truth translations
            sources: Optional source texts
            epoch: Current epoch
            
        Returns:
            PopulationRewards with all reward information
        """
        rewards_by_member: Dict[int, List[float]] = {}
        detailed_results: Dict[int, List[RewardResult]] = {}
        
        for member_idx, results in generation_results. items():
            member_rewards = []
            member_details = []
            
            for sample_idx, result in enumerate(results):
                hypothesis = result.output_text
                reference = references[sample_idx]
                source = sources[sample_idx] if sources else result.input_text
                
                # Compute reward
                reward = self.reward_function.compute(
                    hypothesis=hypothesis,
                    reference=reference,
                    source=source,
                )
                
                member_rewards.append(reward)
                
                # Store detailed result
                member_details.append(RewardResult(
                    member_index=member_idx,
                    sample_index=sample_idx,
                    input_text=result.input_text,
                    hypothesis=hypothesis,
                    reference=reference,
                    reward=reward,
                    metric_name=self.reward_function. name,
                ))
                
            rewards_by_member[member_idx] = member_rewards
            detailed_results[member_idx] = member_details
        
        # Aggregate per-member rewards
        mean_rewards = {}
        for member_idx, rewards in rewards_by_member.items():
            if self.aggregation == "mean":
                mean_rewards[member_idx] = np. mean(rewards)
            elif self.aggregation == "sum":
                mean_rewards[member_idx] = np.sum(rewards)
            elif self.aggregation == "min":
                mean_rewards[member_idx] = np.min(rewards)
            elif self. aggregation == "max":
                mean_rewards[member_idx] = np.max(rewards)
            else:
                mean_rewards[member_idx] = np. mean(rewards)
        
        # Create aggregated rewards array (sorted by member index)
        population_size = len(generation_results)
        aggregated = np.array([mean_rewards[i] for i in range(population_size)])
        
        # Apply fitness shaping
        shaped_rewards = self._apply_fitness_shaping(aggregated)
        
        # Compute statistics
        best_idx = int(np.argmax(aggregated))
        worst_idx = int(np.argmin(aggregated))
        
        return PopulationRewards(
            epoch=epoch,
            rewards=rewards_by_member,
            mean_rewards=mean_rewards,
            aggregated_rewards=shaped_rewards,
            best_member_idx=best_idx,
            best_reward=aggregated[best_idx],
            worst_member_idx=worst_idx,
            worst_reward=aggregated[worst_idx],
            mean_reward=float(np.mean(aggregated)),
            std_reward=float(np.std(aggregated)),
            detailed_results=detailed_results,
        )
    
    def _apply_fitness_shaping(self, rewards: np.ndarray) -> np.ndarray:
        """
        Apply fitness shaping for ES stability.
        
        Fitness shaping helps ES by:
        1.  Reducing variance in gradients
        2.  Making updates invariant to reward scale
        3. Emphasizing relative ordering over absolute values
        """
        if self. fitness_shaping == "none":
            return rewards
            
        elif self.fitness_shaping == "standardize":
            # Zero mean, unit variance
            mean = np. mean(rewards)
            std = np. std(rewards) + 1e-8
            return (rewards - mean) / std
            
        elif self.fitness_shaping == "centered_rank":
            # Rank-based shaping (most robust)
            # Maps rewards to [-0.5, 0.5] based on rank
            n = len(rewards)
            ranks = np. argsort(np.argsort(rewards))  # Ranks from 0 to n-1
            shaped = (ranks. astype(np.float32) + 0.5) / n - 0.5
            return shaped
            
        else:
            return rewards


# ============================================================================
# Convenience Functions
# ============================================================================

def create_reward_function(
    metric: str = "bleu",
    **kwargs,
) -> RewardFunction:
    """
    Factory function to create reward functions.
    
    Args:
        metric: One of "bleu", "meteor", "chrf", "comet", "length", "composite"
        **kwargs: Additional arguments for the specific metric
        
    Returns:
        RewardFunction instance
    """
    metric = metric.lower()
    
    if metric == "bleu":
        return BLEUReward(**kwargs)
    elif metric == "meteor":
        return METEORReward(**kwargs)
    elif metric in ["chrf", "chrf++"]:
        return ChrFReward(**kwargs)
    elif metric == "comet":
        return COMETReward(**kwargs)
    elif metric == "length":
        return LengthRatioReward()
    elif metric == "composite":
        # Default composite: 0.5*BLEU + 0.3*METEOR + 0. 2*Length
        return CompositeReward([
            (BLEUReward(), 0. 5),
            (METEORReward(), 0. 3),
            (LengthRatioReward(), 0.2),
        ])
    else:
        raise ValueError(f"Unknown metric: {metric}")


def print_reward_summary(population_rewards: PopulationRewards):
    """Print summary of population rewards."""
    print("\n" + "=" * 60)
    print(f"Reward Summary (Epoch {population_rewards.epoch})")
    print("=" * 60)
    
    print(f"\nPopulation Statistics:")
    print(f"  Mean reward:  {population_rewards. mean_reward:.4f}")
    print(f"  Std reward:   {population_rewards.std_reward:.4f}")
    print(f"  Best reward:  {population_rewards.best_reward:.4f} (member {population_rewards.best_member_idx})")
    print(f"  Worst reward: {population_rewards. worst_reward:.4f} (member {population_rewards.worst_member_idx})")
    
    print(f"\nPer-Member Mean Rewards:")
    for member_idx, mean_reward in sorted(population_rewards. mean_rewards.items()):
        bar = "█" * int(mean_reward * 40)
        marker = " ← BEST" if member_idx == population_rewards.best_member_idx else ""
        print(f"  Member {member_idx}: {mean_reward:.4f} {bar}{marker}")
    
    print(f"\nShaped Rewards (for ES update):")
    print(f"  {population_rewards.aggregated_rewards}")


def print_sample_results(
    population_rewards: PopulationRewards,
    num_samples: int = 2,
):
    """Print sample results for inspection."""
    print("\n" + "-" * 60)
    print("Sample Results")
    print("-" * 60)
    
    if population_rewards.detailed_results is None:
        print("No detailed results available.")
        return
    
    # Show samples from best and worst members
    for member_idx in [population_rewards.best_member_idx, population_rewards.worst_member_idx]:
        results = population_rewards. detailed_results[member_idx][:num_samples]
        
        label = "BEST" if member_idx == population_rewards.best_member_idx else "WORST"
        print(f"\n[Member {member_idx} - {label}]")
        
        for result in results:
            print(f"  Source: {result.input_text[:60]}...")
            print(f"  Hyp:    {result.hypothesis[:60]}...")
            print(f"  Ref:    {result. reference[:60]}...")
            print(f"  {result.metric_name}: {result.reward:.4f}")
            print()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EGGROLL Step 4: Evaluation / Compute Rewards")
    print("=" * 70)
    
    # Simulated generation results (from Step 3)
    # In practice, these come from PerturbedModelGenerator
    
    test_sources = [
        "Hello, how are you today?",
        "The weather is very nice.",
        "I love learning new languages.",
        "Machine translation is fascinating.",
    ]
    
    test_references = [
        "Xin chào, hôm nay bạn khỏe không?",
        "Thời tiết rất đẹp.",
        "Tôi thích học ngôn ngữ mới.",
        "Dịch máy thật thú vị.",
    ]
    
    # Simulated hypotheses from 4 population members
    # Member 0: Good translations
    # Member 1: Okay translations
    # Member 2: Poor translations
    # Member 3: Mediocre translations
    
    simulated_hypotheses = {
        0: [  # Good
            "Xin chào, hôm nay bạn có khỏe không? ",
            "Thời tiết rất đẹp.",
            "Tôi yêu việc học ngôn ngữ mới.",
            "Dịch máy thật là thú vị.",
        ],
        1: [  # Okay
            "Chào bạn, bạn khỏe không? ",
            "Thời tiết đẹp.",
            "Tôi thích học ngôn ngữ.",
            "Dịch máy rất hay.",
        ],
        2: [  # Poor
            "Xin chào.",
            "Trời đẹp.",
            "Học ngôn ngữ.",
            "Máy dịch.",
        ],
        3: [  # Mediocre
            "Chào, hôm nay thế nào?",
            "Thời tiết tốt.",
            "Tôi muốn học ngôn ngữ mới.",
            "Dịch thuật máy móc rất hay.",
        ],
    }
    
    # Create mock GenerationResult objects
    from dataclasses import dataclass as dc
    
    @dc
    class MockGenerationResult:
        member_index: int
        input_text: str
        output_text: str
        input_ids: Any = None
        output_ids: Any = None
        generation_time: float = 0.0
    
    generation_results = {}
    for member_idx, hypotheses in simulated_hypotheses.items():
        generation_results[member_idx] = [
            MockGenerationResult(
                member_index=member_idx,
                input_text=src,
                output_text=hyp,
            )
            for src, hyp in zip(test_sources, hypotheses)
        ]
    
    # Test different reward functions
    print("\n" + "-" * 70)
    print("Testing Reward Functions")
    print("-" * 70)
    
    # 1. BLEU Score
    print("\n1. BLEU Score")
    bleu_reward = BLEUReward(smoothing=True)
    
    aggregator = PopulationRewardAggregator(
        reward_function=bleu_reward,
        aggregation="mean",
        fitness_shaping="centered_rank",
    )
    
    population_rewards = aggregator.compute_rewards(
        generation_results=generation_results,
        references=test_references,
        sources=test_sources,
        epoch=0,
    )
    
    print_reward_summary(population_rewards)
    print_sample_results(population_rewards, num_samples=1)
    
    # 2. Length Ratio (simple baseline)
    print("\n" + "-" * 70)
    print("2. Length Ratio (baseline)")
    
    length_reward = LengthRatioReward()
    aggregator_length = PopulationRewardAggregator(
        reward_function=length_reward,
        aggregation="mean",
        fitness_shaping="standardize",
    )
    
    length_rewards = aggregator_length.compute_rewards(
        generation_results=generation_results,
        references=test_references,
        epoch=0,
    )
    
    print(f"Mean reward: {length_rewards.mean_reward:. 4f}")
    print(f"Best member: {length_rewards.best_member_idx} ({length_rewards.best_reward:.4f})")
    
    # 3.  Composite reward
    print("\n" + "-" * 70)
    print("3. Composite Reward (BLEU + Length)")
    
    composite_reward = CompositeReward([
        (BLEUReward(), 0.7),
        (LengthRatioReward(), 0.3),
    ])
    
    aggregator_composite = PopulationRewardAggregator(
        reward_function=composite_reward,
        aggregation="mean",
        fitness_shaping="centered_rank",
    )
    
    composite_rewards = aggregator_composite.compute_rewards(
        generation_results=generation_results,
        references=test_references,
        epoch=0,
    )
    
    print_reward_summary(composite_rewards)
    
    # Show fitness shaping comparison
    print("\n" + "-" * 70)
    print("Fitness Shaping Comparison")
    print("-" * 70)
    
    raw_rewards = np.array([population_rewards.mean_rewards[i] for i in range(4)])
    
    print(f"\nRaw rewards:        {raw_rewards}")
    print(f"Standardized:       {(raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-8)}")
    print(f"Centered rank:      {population_rewards.aggregated_rewards}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Step 4 Complete!")
    print("=" * 70)
    print(f"""
Summary:
--------
• Computed rewards for {len(generation_results)} population members
• Each member evaluated on {len(test_references)} samples
• Metric: {bleu_reward.name}

Key Outputs for Step 5:
-----------------------
population_rewards.aggregated_rewards = {population_rewards. aggregated_rewards}

This shaped fitness array will be used in Step 5 to estimate gradients:
  Δθ ≈ (1/Nσ) Σ R_i * (A_i × B_i^T)

Where R_i = aggregated_rewards[i]

Next: Step 5 - Gradient Estimation
""")
