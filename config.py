"""Central configuration (dataclass + YAML loader). Keeps every module on one contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class AnalyzerConfig:
    model_name: str = "gpt2"  # lightweight local scorer; swap for e.g. Qwen/Qwen2.5-0.5B
    device: Optional[str] = None  # None => auto (cuda/mps/cpu)
    max_length: int = 1024
    stride: int = 512


@dataclass
class ClassifierConfig:
    tfidf_max_features: int = 5000
    tfidf_ngram_max: int = 2
    use_stats_features: bool = True
    model_type: str = "auto"  # auto | xgboost | histgb | logreg | mlp
    xgb_estimators: int = 300  # boosting rounds (longer training = higher)
    gb_max_iter: int = 300  # histgb iterations (longer training = higher)
    mlp_hidden_layers: list[int] = field(default_factory=lambda: [128, 64])
    mlp_max_iter: int = 500
    test_size: float = 0.2
    random_state: int = 42
    save_path: str = "artifacts/classifier.joblib"


@dataclass
class GeneratorConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    fallback_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"  # tiny + instruction-tuned
    load_in_4bit: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    max_input_chars: int = 6000


@dataclass
class RLConfig:
    algo: str = "ppo"  # ppo | dpo
    output_dir: str = "artifacts/rl_adapter"
    num_epochs: int = 1
    batch_size: int = 4
    mini_batch_size: int = 1
    learning_rate: float = 1.4e-5
    kl_coef: float = 0.2
    target_kl: float = 6.0
    seed: int = 42


@dataclass
class PipelineConfig:
    target_score: float = 0.85
    max_iters: int = 4
    num_candidates: int = 2
    max_chars: int = 20_000
    min_similarity: float = 0.35  # reject rewrites that drift off-topic
    polish_passes: int = 2  # refinement rounds on the winner (0 = off)


@dataclass
class AppConfig:
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    seed: int = 42
    artifacts_dir: str = "artifacts"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        import yaml  # local dep, listed in requirements.txt

        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        cfg = cls()
        for section in ("analyzer", "classifier", "generator", "rl", "pipeline"):
            if section in raw and isinstance(raw[section], dict):
                target = getattr(cfg, section)
                for k, v in raw[section].items():
                    if hasattr(target, k):
                        setattr(target, k, v)
        for k in ("seed", "artifacts_dir"):
            if k in raw:
                setattr(cfg, k, raw[k])
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzer": asdict(self.analyzer),
            "classifier": asdict(self.classifier),
            "generator": asdict(self.generator),
            "rl": asdict(self.rl),
            "pipeline": asdict(self.pipeline),
            "seed": self.seed,
            "artifacts_dir": self.artifacts_dir,
        }


def apply_light_mode(cfg: "AppConfig") -> "AppConfig":
    """Trade speed for memory: smaller windows/features/batches/models.

    Effects: analyzer window 512, TF-IDF capped at 2000 features, fallback
    generator SmolLM2-360M-Instruct (~700MB), 1 candidate/iter, RL batch 1.
    Training/inference get slower but peak RAM, CPU and disk use drop.
    """
    cfg.analyzer.max_length = min(cfg.analyzer.max_length, 512)
    cfg.analyzer.stride = min(cfg.analyzer.stride, 256)
    cfg.classifier.tfidf_max_features = min(cfg.classifier.tfidf_max_features, 2000)
    cfg.generator.fallback_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    cfg.pipeline.num_candidates = 1
    cfg.pipeline.polish_passes = 1
    cfg.rl.batch_size = 1
    cfg.rl.mini_batch_size = 1
    return cfg
