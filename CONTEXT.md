# Humaize — Domain Glossary

Good seams are named after domain concepts. Use these terms when naming deep modules; don't invent synonyms.

- **Human-Likeness Score** — `classifier.LocalPatternClassifier` output in [0,1]; learned from paired data, never a rule.
- **PatternExplanation** — per-text attribution: which stat features + tokens pull toward AI centroid.
- **GateSpec / Fidelity** — the triple gate: `min_similarity` (recall), `is_near_copy` (uni+bi), `MAX_LENGTH_RATIO` (bloat). One spec, all gates read it.
- **CriticBundle** — the paired artifact: `(critic, analyzer)` with matched `stat_dim`. Single interface `load(path)` owns the 34-vs-30 decision.
- **ResourceProfile** — the resource seam: caps for analyzer window, TF-IDF, generator model, batch sizes, threads. One module owns "what light/Yoga means". Deepens `config.apply_light_mode`.
- **RewriteSanitizer** — the 5-step prompt/guidance/meta sanitization chain inside `generator`. One interface `sanitize(text, payload, guidance)`.
- **EffortPreset** — named search budgets (`quick/standard/deep/max`) mapping to `PipelineConfig` values. Single source in `config`.
- **Payload Isolation** — `<input_text>` contract; only inner text reaches models.
- **Document Unit** — a humanizable span: paragraph or table cell, with short-label/numeric skip policy.

ADRs: see `docs/adr/` (none yet). If a future review re-proposes a rejected seam, record an ADR.
