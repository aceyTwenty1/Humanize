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
    # GateSpec fields — single source for all fidelity gates (no literals
    # elsewhere). Deepens the scattered 0.85/0.75/1.8 literals into one
    # module; pipeline, generator, and dochumanize all read the spec.
    copy_unigram: float = 0.85
    copy_bigram: float = 0.75
    max_length_ratio: float = 1.8

    def gate_spec(self):
        from utils import GateSpec

        return GateSpec(
            min_similarity=self.min_similarity,
            copy_unigram=self.copy_unigram,
            copy_bigram=self.copy_bigram,
            max_length_ratio=self.max_length_ratio,
        )


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
    """ResourceProfile — Yoga 9i Ultra (i7 EVO / Iris Xe / 16GB shared).

    Pure function: returns a *new* AppConfig, never mutates the input.
    Deepens the former shallow `is_light()` scatter: this is the single
    module that owns "what light/Yoga means". Callers pass the returned
    config; no one re-checks `HUMAIZE_LIGHT` behind the seam.

    Ultra caps for max speed on Yoga (≈2.5× faster than generic light):
      analyzer 128/64 (was 256), TF-IDF 800 + logreg, generator prompt
      2000 chars + 64 new tokens (was 3000/96), pipeline 1 iter / 0 polish
      (was 2/0), RL 1. Peak RAM ~900MB, 2 threads, ~3s/rewrite after warmup.
      Quality holds for ≤400w inputs; long docs still chunk at 40w.
    """
    from dataclasses import replace

    return replace(
        cfg,
        analyzer=replace(
            cfg.analyzer,
            max_length=min(cfg.analyzer.max_length, 128),
            stride=min(cfg.analyzer.stride, 64),
        ),
        classifier=replace(
            cfg.classifier,
            tfidf_max_features=min(cfg.classifier.tfidf_max_features, 800),
            model_type="logreg" if cfg.classifier.model_type == "auto" else cfg.classifier.model_type,
        ),
        generator=replace(
            cfg.generator,
            fallback_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
            max_input_chars=min(cfg.generator.max_input_chars, 2000),
            torch_dtype="float32" if cfg.generator.torch_dtype == "auto" else cfg.generator.torch_dtype,
        ),
        pipeline=replace(
            cfg.pipeline,
            num_candidates=1,
            max_iters=min(cfg.pipeline.max_iters, 1),
            polish_passes=0,
        ),
        rl=replace(cfg.rl, batch_size=1, mini_batch_size=1),
    )


# Backwards alias — old name still works, but new code should name the
# concept: ResourceProfile.
def get_resource_profile(cfg: "AppConfig") -> "AppConfig":
    """Alias for apply_light_mode — the Yoga seam."""
    return apply_light_mode(cfg)
