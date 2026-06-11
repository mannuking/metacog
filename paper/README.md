# Paper directory

All artifacts needed to write the final research paper live here.

## Layout

```
paper/
├── figures/      — reliability diagrams, training curves, calibration plots (PNG + SVG + CSV)
├── tables/       — main result tables (CSV + LaTeX + Markdown)
├── configs/      — every hyperparameter config used in every run (YAML/TOML)
├── references/   — PDFs, arxiv links, BibTeX entries for all cited works
├── OUTLINE.md    — paper section outline (filled in as we go)
└── README.md     — this file
```

## Rules

1. **Every run writes to `results/` AND copies its summary to `paper/tables/` + its figures to `paper/figures/`.**
2. **Every config is committed to `paper/configs/` BEFORE the run starts.** No "I'll add it later" — the run is invalid if its config isn't on disk.
3. **No edits to past run outputs.** If we re-run with new metrics, that's a new run with a new prefix, not a rewrite.
4. **CSV is the canonical format** for tables (long-term preservation). Markdown and LaTeX are derived.
5. **`paper/OUTLINE.md` is updated at the start of each phase** with what was done and what will be in the paper.

## Paper metadata (target venue: TBD)

- Working title: "Calibrated Self-Knowledge in 35B-Parameter Open-Weight Models via Reinforcement Learning on Verifiable Metacognitive Rewards"
- Authors: Jai kumar Meena
- Repo: github.com/mannuking/metacog
- Target venues (in order of preference): NeurIPS 2026 Workshop on Self-Improving Agents · ICML 2026 Workshop on Metacognition in ML · COLM 2026 · arXiv preprint
- Page target: 8 pages main + unlimited appendix
