"""MODULE 2 — Local ML Classifier & Reward Model.

Trains a *learned* discriminator on paired human (1) vs AI (0) texts:

    features = TF-IDF (learned vocabulary weights)  +  statistical features
               from PatternAnalyzer (burstiness, entropy, ...)

No hardcoded rules: every weight/threshold is fit from data. The model
outputs a continuous Human-Likeness Score in [0, 1] and doubles as the
automated Reward Model for the RL loop (Actor generates -> Critic scores).

Backends (``model_type="auto"`` picks the first available):
    xgboost  -> XGBClassifier (preferred, gradient-boosted trees)
    histgb   -> sklearn HistGradientBoostingClassifier (no extra dep)
    logreg   -> sklearn LogisticRegression (tiny-data fallback)
    mlp      -> sklearn MLPClassifier (feed-forward neural net; opt in with
                model_type="mlp" or --model-type mlp — shines with 100+
                samples, overkill for the 20-sample demo)

A fine-tuned local DeBERTa variant can be slotted in later behind the same
``human_likeness_score`` interface; the TF-IDF+GBM path is the default
because it trains on CPU in seconds and is fully local.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from config import ClassifierConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from analyzer import PatternAnalyzer
except ImportError:  # pragma: no cover - package-layout fallback
    from src.humaize.analyzer import PatternAnalyzer  # type: ignore


@dataclass
class ClassifierReport:
    accuracy: float
    f1: float
    roc_auc: float
    n_train: int
    n_test: int
    backend: str


@dataclass
class FlaggedFeature:
    """One stat feature pulling the text toward the AI class centroid."""

    name: str
    value: float
    human_mean: float
    ai_mean: float
    gap: float  # how much closer to AI than to human (z units)

    @property
    def direction(self) -> str:
        """Which way to move this feature to look more human (learned)."""
        return "raise" if self.human_mean > self.value else "lower"

    def describe(self) -> str:
        return (
            f"{self.name}: yours {self.value:.2f} → move {self.direction} "
            f"toward human {self.human_mean:.2f} (AI sits at {self.ai_mean:.2f})"
        )


@dataclass
class PatternExplanation:
    """Which *learned* patterns make a text recognizable as AI.

    All reference numbers are class centroids measured from the training
    data at fit time — no hardcoded thresholds.
    """

    score: float
    flagged: list[FlaggedFeature]
    token_hits: list[tuple[str, float]]
    directional_tokens: bool  # True: weights point at AI; False: influence only

    def guidance(self, max_features: int = 2) -> str:
        """Compact editing direction for the generator, phrased from measurements.

        Deliberately terse (top features only): small local models echo long
        analyses back into their output. Detection stays fully learned; this
        is just how the findings are communicated. Each feature names the
        *direction* (raise/lower toward the human centroid) so the rewrite
        targets the pattern instead of paraphrasing blindly.
        """
        bits = []
        for f in self.flagged[:max(max_features, 1)]:
            bits.append(
                f"{f.direction} {f.name} (yours {f.value:.2f}, "
                f"human-like {f.human_mean:.2f}, AI-like {f.ai_mean:.2f})"
            )
        if self.token_hits:
            toks = ", ".join(f"'{t}'" for t, _ in self.token_hits[:4])
            bits.append(f"rephrase {toks}")
        if not bits:
            return ""
        return "Edit targets (keep meaning, move toward human values): " + "; ".join(bits) + "."

    def summary(self) -> str:
        bits = [f"{f.name}→AI ({f.value:.2f} vs human {f.human_mean:.2f})" for f in self.flagged]
        if self.token_hits:
            bits.append("tokens: " + ", ".join(t for t, _ in self.token_hits))
        return "; ".join(bits) if bits else "no dominant AI pattern"

    def details(self) -> str:
        """Multi-line breakdown: per-feature direction + token evidence."""
        lines = [f"human-likeness {self.score:.3f}"]
        for f in self.flagged:
            lines.append(f"  • {f.describe()}  [gap {f.gap:.2f}σ]")
        if self.token_hits:
            toks = ", ".join(f"'{t}' ({w:.2f})" for t, w in self.token_hits)
            scope = "AI-pointing weights" if self.directional_tokens else "high-influence terms"
            lines.append(f"  • loaded {scope}: {toks}")
        if len(lines) == 1:
            lines.append("  • no dominant AI pattern")
        return "\n".join(lines)


class LocalPatternClassifier:
    """Learned human-vs-AI discriminator + RL reward model."""

    def __init__(
        self,
        config: Optional[ClassifierConfig] = None,
        analyzer: Optional[PatternAnalyzer] = None,
    ) -> None:
        self.config = config or ClassifierConfig()
        self.analyzer = analyzer  # may be None -> LM-free cheap stats only
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=(1, self.config.tfidf_ngram_max),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.scaler = StandardScaler()
        self.model = None
        self.backend = "unfitted"
        self._stat_dim = 0
        self._stat_names: list[str] = []
        self._human_mean = None  # per-class stat centroids (fit from data)
        self._ai_mean = None
        self._stat_std = None

    # ------------------------------------------------------------ backend pick
    def _build_backend(self, n_samples: int):
        kind = self.config.model_type.lower()
        if kind == "auto":
            try:
                import xgboost  # noqa: F401

                kind = "xgboost"
            except ImportError:
                kind = "histgb" if n_samples >= 20 else "logreg"
        if kind == "xgboost":
            from xgboost import XGBClassifier

            self.backend = "xgboost"
            from utils import is_light

            # --light caps rounds for RAM; otherwise honor the training budget.
            n_est = min(self.config.xgb_estimators, 100) if is_light() else self.config.xgb_estimators
            return XGBClassifier(
                n_estimators=n_est,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=self.config.random_state,
                n_jobs=2 if is_light() else -1,
            )
        if kind == "histgb":
            from sklearn.ensemble import HistGradientBoostingClassifier

            from utils import is_light

            self.backend = "histgb"
            max_iter = min(self.config.gb_max_iter, 100) if is_light() else self.config.gb_max_iter
            return HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=0.06,
                max_depth=None,
                random_state=self.config.random_state,
            )
        if kind == "mlp":
            from sklearn.neural_network import MLPClassifier

            from utils import is_light

            self.backend = "mlp"
            hidden = tuple(self.config.mlp_hidden_layers or [128, 64])
            max_iter = self.config.mlp_max_iter
            if is_light():
                hidden = tuple(h // 2 for h in hidden)
                max_iter = min(max_iter, 300)
            return MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=max_iter,
                random_state=self.config.random_state,
            )
        from sklearn.linear_model import LogisticRegression

        self.backend = "logreg"
        return LogisticRegression(max_iter=2000, C=2.0)

    # --------------------------------------------------------------- features
    def _current_stat_names(self) -> list[str]:
        if self.analyzer is not None and self.analyzer.has_lm:
            try:
                from analyzer import STAT_FEATURE_NAMES
            except ImportError:  # pragma: no cover
                from src.humaize.analyzer import STAT_FEATURE_NAMES  # type: ignore
            return list(STAT_FEATURE_NAMES)
        return list(PatternAnalyzer.CHEAP_FEATURE_NAMES)

    def _stat_features(self, texts: list[str]) -> np.ndarray:
        if self.analyzer is not None and self.analyzer.has_lm:
            return np.array(
                [self.analyzer.analyze(t).to_feature_vector() for t in texts], dtype=float
            )
        # LM-free path: cheap stats only (also used in CI / fast training)
        return np.array([PatternAnalyzer.cheap_stats_vector(t) for t in texts], dtype=float)

    def _featurize_fit(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
        """Fit featurizers on TRAIN texts only; also return raw stat matrix."""
        from scipy.sparse import csr_matrix, hstack

        tfidf = self.vectorizer.fit_transform(texts)
        if not self.config.use_stats_features:
            self._stat_dim = 0
            self._stat_names: list[str] = []
            return tfidf, None
        stats = self._stat_features(texts)
        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(stats)
        self._stat_dim = stats.shape[1]
        self._stat_names = self._current_stat_names()
        return hstack([tfidf, csr_matrix(self.scaler.transform(stats))]), stats

    def _featurize(self, texts: list[str]) -> np.ndarray:
        from scipy.sparse import csr_matrix, hstack

        tfidf = self.vectorizer.transform(texts)
        if not self.config.use_stats_features or self._stat_dim == 0:
            return tfidf
        stats = self._stat_features(texts)
        if stats.shape[1] != self._stat_dim:
            raise RuntimeError(
                f"Stat feature mismatch: critic trained with {self._stat_dim} stats "
                f"features ({getattr(self, '_stat_names', '?')}) but current analyzer "
                f"produces {stats.shape[1]}. Retrain with the same --fast setting you "
                f"infer with (LM stats need the scorer model loaded in both)."
            )
        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        return hstack([tfidf, csr_matrix(self.scaler.transform(stats))])

    # ---------------------------------------------------------------- training
    def fit(self, texts: list[str], labels: list[int]) -> ClassifierReport:
        """Fit the discriminator. Labels: 1 = human, 0 = AI."""
        if len(texts) != len(labels) or len(texts) < 4:
            raise ValueError("Need >= 4 paired texts with matching labels")
        y = np.array(labels, dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("Need both classes (0=AI and 1=human) in training data")

        n = len(texts)
        # Split FIRST, then fit featurizers on train only — fitting TF-IDF /
        # scaler on all texts leaks test vocabulary/scales into training.
        can_stratify = min(np.bincount(y)) >= 2 and n >= 6
        idx_tr, idx_te = train_test_split(
            np.arange(n),
            test_size=min(self.config.test_size, 0.4),
            random_state=self.config.random_state,
            stratify=y if can_stratify else None,
        )
        Xtr, raw_tr = self._featurize_fit([texts[i] for i in idx_tr])
        Xte = self._featurize([texts[i] for i in idx_te])
        ytr, yte = y[idx_tr], y[idx_te]
        # HistGB/XGB dislike sparse input -> densify the (small) matrices
        self.model = self._build_backend(len(idx_tr))
        import scipy.sparse as sp

        backend_needs_dense = self.backend in ("histgb", "xgboost", "mlp")
        Xt = Xtr.toarray() if (backend_needs_dense and sp.issparse(Xtr)) else Xtr
        Xe = Xte.toarray() if (backend_needs_dense and sp.issparse(Xte)) else Xte
        logger.info("LocalPatternClassifier: training backend=%s on %d samples", self.backend, len(idx_tr))
        self.model.fit(Xt, ytr)
        self._fit_centroids(raw_tr, ytr)
        proba = self.model.predict_proba(Xe)[:, 1]
        pred = (proba >= 0.5).astype(int)
        report = ClassifierReport(
            accuracy=float(accuracy_score(yte, pred)),
            f1=float(f1_score(yte, pred, zero_division=0)),
            roc_auc=float(roc_auc_score(yte, proba)) if len(set(yte.tolist())) > 1 else 0.5,
            n_train=len(idx_tr),
            n_test=len(idx_te),
            backend=self.backend,
        )
        logger.info(
            "LocalPatternClassifier: acc=%.3f f1=%.3f auc=%.3f (%s)",
            report.accuracy,
            report.f1,
            report.roc_auc,
            self.backend,
        )
        return report

    # ------------------------------------------------------- scoring / reward
    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Classifier is not trained — call fit() or load() first.")

    def _fit_centroids(self, raw_stats: np.ndarray | None, ytr: np.ndarray) -> None:
        """Mean stat vector per class (+pooled std) from TRAIN split stats."""
        if raw_stats is None or raw_stats.shape[0] != len(ytr):
            self._human_mean = self._ai_mean = self._stat_std = None
            return
        h, a = raw_stats[ytr == 1], raw_stats[ytr == 0]
        if len(h) == 0 or len(a) == 0:
            self._human_mean = self._ai_mean = self._stat_std = None
            return
        self._human_mean = h.mean(axis=0).astype(float)
        self._ai_mean = a.mean(axis=0).astype(float)
        pooled = raw_stats.std(axis=0, ddof=1)
        self._stat_std = np.where(~np.isfinite(pooled) | (pooled < 1e-6), 1e-6, pooled).astype(float)

    def explain(self, text: str, top_features: int = 4, top_tokens: int = 6) -> PatternExplanation:
        """Attribute an AI verdict to learned training patterns.

        Compares the text's stat features against the human/AI class
        centroids measured at fit time, and ranks the input's tokens by
        learned model weights. Raises RuntimeError on pre-centroid
        artifacts (retrain to enable explanations).
        """
        self._check_fitted()
        if self._human_mean is None or self._ai_mean is None or self._stat_std is None:
            raise RuntimeError(
                "No pattern centroids stored — retrain the critic (fit() now "
                "records per-class training profiles) to enable explain()."
            )
        vec = np.nan_to_num(self._stat_features([text])[0], nan=0.0, posinf=0.0, neginf=0.0)
        names = self._stat_names or [f"stat_{i}" for i in range(len(vec))]
        flagged: list[FlaggedFeature] = []
        for i, (v, hu, ai, sd) in enumerate(zip(vec, self._human_mean, self._ai_mean, self._stat_std)):
            z_ai, z_hu = abs(v - ai) / sd, abs(v - hu) / sd
            if z_ai < z_hu:
                flagged.append(FlaggedFeature(names[i], float(v), float(hu), float(ai), float(z_hu - z_ai)))
        flagged.sort(key=lambda f: f.gap, reverse=True)
        token_hits, directional = self._attribute_tokens(text, top_tokens)
        return PatternExplanation(
            score=float(self.predict_proba_human([text])[0]),
            flagged=flagged[:max(top_features, 1)],
            token_hits=token_hits,
            directional_tokens=directional,
        )

    def _attribute_tokens(self, text: str, top_k: int) -> tuple[list[tuple[str, float]], bool]:
        """Rank the input's tokens by learned weight (label 1 = human, so
        negative linear weights pull toward AI; tree importances are
        unsigned influence). Returns (hits, directional)."""
        import scipy.sparse as sp

        try:
            tv = self.vectorizer.transform([text]).tocsr()
            if tv.nnz == 0:
                return [], False
            vocab_size = len(self.vectorizer.vocabulary_)
            if self.backend == "logreg" and hasattr(self.model, "coef_"):
                w = np.asarray(self.model.coef_[0][:vocab_size], dtype=float)
                directional = True
            elif hasattr(self.model, "feature_importances_"):
                w = np.asarray(self.model.feature_importances_[:vocab_size], dtype=float)
                directional = False
            else:
                return [], False
            inv = {j: t for t, j in self.vectorizer.vocabulary_.items()}
            vals = tv.toarray()[0]
            scored = []
            for j in tv.indices:
                contrib = vals[j] * (-w[j] if directional else w[j])
                if contrib > 0:
                    scored.append((inv.get(j, "?"), float(contrib)))
            scored.sort(key=lambda kv: kv[1], reverse=True)
            return scored[:max(top_k, 1)], directional
        except Exception:
            return [], False

    def predict_proba_human(self, texts: list[str]) -> np.ndarray:
        """P(human) for each text in [0, 1]."""
        self._check_fitted()
        import scipy.sparse as sp

        X = self._featurize(texts)
        if self.backend in ("histgb", "xgboost", "mlp") and sp.issparse(X):
            X = X.toarray()
        return np.asarray(self.model.predict_proba(X)[:, 1], dtype=float).clip(0.0, 1.0)

    def human_likeness_score(self, text: str) -> float:
        """Continuous Human-Likeness Score for a single text."""
        return float(self.predict_proba_human([text])[0])

    def global_importances(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Top learned signals across the whole training set.

        Covers TF-IDF vocabulary weights plus stat-feature weights for
        linear backends; unsigned tree importances otherwise. Names are
        prefixed ``tok:`` / ``stat:`` so callers can tell them apart.
        """
        self._check_fitted()
        import numpy as np

        top_k = max(top_k, 1)
        if self.backend == "logreg" and hasattr(self.model, "coef_"):
            w = np.asarray(self.model.coef_[0], dtype=float)
            names = (
                [f"tok:{t}" for t, _ in sorted(
                    self.vectorizer.vocabulary_.items(), key=lambda kv: kv[1])]
                + [f"stat:{n}" for n in self._stat_names]
            )
            ranked = sorted(zip(names, -w), key=lambda kv: kv[1], reverse=True)
            return [(n, float(v)) for n, v in ranked[:top_k] if v > 0]
        if hasattr(self.model, "feature_importances_"):
            w = np.asarray(self.model.feature_importances_, dtype=float)
            vocab_size = len(self.vectorizer.vocabulary_)
            inv = {j: t for t, j in self.vectorizer.vocabulary_.items()}
            names = ([f"tok:{inv.get(j, '?')}" for j in range(vocab_size)]
                     + [f"stat:{n}" for n in self._stat_names])
            ranked = sorted(zip(names, w), key=lambda kv: kv[1], reverse=True)
            return [(n, float(v)) for n, v in ranked[:top_k] if v > 0]
        return []

    def reward(self, texts: list[str] | str) -> list[float] | float:
        """Reward-model interface for the RL loop (PPO/TRL expects scalars)."""
        single = isinstance(texts, str)
        scores = self.predict_proba_human([texts] if single else list(texts))
        out = [float(s) for s in scores]
        return out[0] if single else out

    # -------------------------------------------------------------- persistence
    def save(self, path: Optional[str | Path] = None) -> Path:
        self._check_fitted()
        dest = Path(path or self.config.save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "scaler": self.scaler,
                "model": self.model,
                "backend": self.backend,
                "stat_dim": self._stat_dim,
                "stat_names": getattr(self, "_stat_names", []),
                "human_mean": self._human_mean,
                "ai_mean": self._ai_mean,
                "stat_std": self._stat_std,
                "config": self.config,
            },
            dest,
        )
        logger.info("LocalPatternClassifier: saved to %s", dest)
        return dest

    @classmethod
    def load(
        cls, path: str | Path, analyzer: Optional[PatternAnalyzer] = None
    ) -> "LocalPatternClassifier":
        blob = joblib.load(path)
        obj = cls(config=blob.get("config", ClassifierConfig()), analyzer=analyzer)
        obj.vectorizer = blob["vectorizer"]
        obj.scaler = blob["scaler"]
        obj.model = blob["model"]
        obj.backend = blob.get("backend", "loaded")
        obj._stat_dim = blob.get("stat_dim", 0)
        obj._stat_names = blob.get("stat_names", [])
        obj._human_mean = blob.get("human_mean", None)
        obj._ai_mean = blob.get("ai_mean", None)
        obj._stat_std = blob.get("stat_std", None)
        logger.info("LocalPatternClassifier: loaded from %s (%s)", path, obj.backend)
        return obj


@dataclass
class CriticBundle:
    """CriticBundle — the paired (critic, analyzer) artifact.

    Single interface `load(path)` owns the 34-vs-30 decision. Callers
    never re-derive `need_lm`; the bundle matches `stat_dim` to the
    analyzer's expected dims. This is the deep module for
    Human-Likeness scoring — one seam, high leverage.
    """

    critic: LocalPatternClassifier
    analyzer: "PatternAnalyzer"
    stat_dim: int

    @classmethod
    def load(
        cls,
        path: str | Path,
        analyzer_config: Optional["AnalyzerConfig"] = None,
        analyzer: Optional["PatternAnalyzer"] = None,
    ) -> "CriticBundle":
        """Load a saved critic and its matched analyzer.

        If `analyzer` is given, it is used directly. Otherwise the
        analyzer's `load_model` is chosen from the artifact's stat_dim:
        cheap (30) → LM-free, full (34) → with LM. No caller guesses
        `--fast`.
        """
        # Lazy import to avoid circular
        try:
            from analyzer import PatternAnalyzer as PA
        except ImportError:  # pragma: no cover
            from src.humaize.analyzer import PatternAnalyzer as PA  # type: ignore

        from config import AnalyzerConfig as AC  # local

        blob = joblib.load(path)
        # Reuse LocalPatternClassifier.load logic but keep blob for stat_dim
        obj = LocalPatternClassifier.load(path, analyzer=analyzer)
        if analyzer is not None:
            ana = analyzer
        else:
            cheap_len = len(PA.CHEAP_FEATURE_NAMES)
            stat_dim = blob.get("stat_dim", 0)
            need_lm = stat_dim != cheap_len and stat_dim != 0
            cfg = analyzer_config or AC()
            ana = PA(config=cfg, load_model=need_lm)
            obj.analyzer = ana
        return cls(critic=obj, analyzer=obj.analyzer, stat_dim=obj._stat_dim)

    def human_likeness_score(self, text: str) -> float:
        return self.critic.human_likeness_score(text)

    def explain(self, text: str, **kw):
        return self.critic.explain(text, **kw)


if __name__ == "__main__":  # smoke test: python classifier.py
    from data_loader import load_paired_dataset, texts_and_labels

    samples = load_paired_dataset(None)
    texts, labels = texts_and_labels(samples)
    clf = LocalPatternClassifier(analyzer=PatternAnalyzer(load_model=False))
    print(clf.fit(texts, labels))
    print("human-like:", round(clf.human_likeness_score(samples[0].text), 3))
    print("ai-like:   ", round(clf.human_likeness_score(samples[-1].text), 3))
