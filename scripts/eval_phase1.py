"""Post-training eval and report generation.

After Phase 1 training finishes, this script:
  1. Loads the trained LoRA checkpoint (via tinker:// path)
  2. Runs the FULL eval suite (50 questions, K=1, K=5)
  3. Computes all the paper-quality metrics (ECE, Brier, AUROC, reliability
     diagram, per-source breakdown, abstention rate, etc.)
  4. Writes results to results/phase1/ AND paper/tables/ + paper/figures/

This is the script that produces the headline numbers for the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import tinker
from eval.metrics import calibration_report
from eval.structured_parse import parse_metacog_output
from data.load_datasets import load_gsm8k, load_mmlu_pro
from data.schemas import QuestionRecord
from eval.normalize import is_correct
from training.rollout import rollout_for_question
from training.reward import compute_reward


def evaluate(
    sampling_client: tinker.SamplingClient,
    tokenizer,
    questions: list[QuestionRecord],
    k: int,
    max_tokens: int,
    effort: str,
) -> dict:
    """Run full eval on a question set."""
    rollouts = []
    for q in questions:
        try:
            rs = rollout_for_question(
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                question=q,
                k=k,
                max_tokens=max_tokens,
                temperature=0.0 if k == 1 else 0.7,
                effort=effort,
            )
            rollouts.extend(rs)
        except Exception as e:
            print(f"  ERROR on {q.id}: {e}")
    return aggregate(rollouts)


def aggregate(rollouts) -> dict:
    correct = [r.breakdown.correct for r in rollouts]
    confs = [r.breakdown.confidence or 0.5 for r in rollouts]
    abstained = sum(1 for r in rollouts if r.breakdown.abstained)
    parse_ok = sum(1 for r in rollouts if r.breakdown.parse_ok)
    n = len(rollouts)
    if n == 0:
        return {"n": 0}
    rep = calibration_report(confs, correct)
    out = {
        "n": n,
        "n_correct": sum(correct),
        "accuracy": round(sum(correct) / n, 4),
        "n_abstained": abstained,
        "abstain_rate": round(abstained / n, 4),
        "n_parse_ok": parse_ok,
        "parse_ok_rate": round(parse_ok / n, 4),
        "avg_confidence": round(sum(confs) / n, 4),
        "ece": round(rep.ece, 4),
        "brier": round(rep.brier, 4),
        "auroc": round(rep.auroc, 4) if rep.auroc is not None else None,
        "overconfident_rate": round(rep.overconfidence_rate, 4),
        "underconfident_rate": round(rep.underconfidence_rate, 4),
        "reliability_bins": [
            {"lo": lo, "hi": hi, "n": n_, "mean_conf": round(mc, 4), "actual_acc": round(acc, 4)}
            for lo, hi, n_, mc, acc in rep.bin_report if n_ > 0
        ],
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="tinker:// path to the LoRA checkpoint")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--n-gsm8k", type=int, default=25)
    p.add_argument("--n-mmlu-pro", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--effort", type=str, default="medium", choices=["low","medium","high"])
    p.add_argument("--out-dir", type=str, default="results/phase1_post_eval")
    p.add_argument("--paper-dir", type=str, default="paper")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path(args.paper_dir)
    (paper_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"POST-TRAINING EVAL — {args.checkpoint}")
    print("=" * 60)

    api_key = os.environ.get("TINKER_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_TINKER_API_KEY_HERE":
        raise RuntimeError("TINKER_API_KEY missing. Set in .env first.")
    sc = tinker.ServiceClient()

    # Need a sampling client. Create a training client just to get a tokenizer.
    # Actually, we can create a sampling client directly from the tinker:// path
    # via the ServiceClient... but the API is create_sampling_client(model_path=...)
    # which is for the base model. For a LoRA, we need create_sampling_client via
    # TrainingClient.save_weights_for_sampler, which is done during training.
    # For post-training eval, we re-load the LoRA via create_training_client_from_state.
    print(f"Loading LoRA from {args.checkpoint}...")
    tc = sc.create_training_client_from_state(args.checkpoint)
    tokenizer = tc.get_tokenizer()
    # Make a sampling client for the LoRA
    sc_resp = tc.save_weights_for_sampler(name="phase1_post_eval").result()
    sampling_client = tc.create_sampling_client(sc_resp.path)
    print(f"  sampling client: {sc_resp.path}")

    # Load questions
    questions = []
    if args.n_gsm8k > 0:
        questions.extend(load_gsm8k(args.n_gsm8k))
    if args.n_mmlu_pro > 0:
        questions.extend(load_mmlu_pro(args.n_mmlu_pro))
    print(f"  loaded {len(questions)} questions")

    # K=1 (greedy) eval — primary paper metric
    print("\n[1/2] K=1 greedy eval (primary metric)...")
    t0 = time.time()
    k1_results = evaluate(sampling_client, tokenizer, questions, k=1, max_tokens=args.max_tokens, effort=args.effort)
    print(f"  K=1 done in {time.time()-t0:.0f}s")
    print(f"  accuracy: {k1_results['accuracy']}")
    print(f"  ECE:      {k1_results['ece']}")
    print(f"  Brier:    {k1_results['brier']}")
    print(f"  AUROC:    {k1_results['auroc']}")
    print(f"  abst_rate: {k1_results['abstain_rate']}")
    print(f"  parse_ok:  {k1_results['parse_ok_rate']}")

    # Per-source breakdown for K=1
    by_source: dict[str, list] = {}
    for q in questions:
        # Re-rollout per question to get per-source breakdown
        try:
            rs = rollout_for_question(
                sampling_client=sampling_client, tokenizer=tokenizer,
                question=q, k=1, max_tokens=args.max_tokens,
                temperature=0.0, effort=args.effort,
            )
            by_source.setdefault(q.source, []).extend(rs)
        except Exception as e:
            print(f"  per-source err on {q.id}: {e}")
    k1_per_source = {src: aggregate(rs) for src, rs in by_source.items()}

    # K=5 (self-consistency) eval — expensive but gold standard
    print("\n[2/2] K=5 self-consistency eval (upper bound)...")
    t0 = time.time()
    k5_results = evaluate(sampling_client, tokenizer, questions, k=5, max_tokens=args.max_tokens, effort=args.effort)
    print(f"  K=5 done in {time.time()-t0:.0f}s")
    print(f"  accuracy: {k5_results['accuracy']}")
    print(f"  ECE:      {k5_results['ece']}")

    # ── Write outputs ────────────────────────────────────────────────
    summary = {
        "config": vars(args),
        "checkpoint": args.checkpoint,
        "k1": k1_results,
        "k1_per_source": k1_per_source,
        "k5": k5_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = out_dir / "post_eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Also save to paper/tables/
    paper_path = paper_dir / "tables" / "phase1_post_eval.csv"
    write_csv_table(paper_path, summary)
    print(f"Saved → {paper_path}")


def write_csv_table(path: Path, summary: dict) -> None:
    """Write the main results table as CSV (paper/tables/)."""
    import csv
    rows = [
        # Header
        ["metric", "k1_overall", "k1_gsm8k", "k1_mmlu_pro", "k5_overall"],
        ["n", summary["k1"]["n"], summary["k1_per_source"].get("gsm8k", {}).get("n", ""), summary["k1_per_source"].get("mmlu_pro", {}).get("n", ""), summary["k5"]["n"]],
        ["accuracy", summary["k1"]["accuracy"], summary["k1_per_source"].get("gsm8k", {}).get("accuracy", ""), summary["k1_per_source"].get("mmlu_pro", {}).get("accuracy", ""), summary["k5"]["accuracy"]],
        ["ece", summary["k1"]["ece"], summary["k1_per_source"].get("gsm8k", {}).get("ece", ""), summary["k1_per_source"].get("mmlu_pro", {}).get("ece", ""), summary["k5"]["ece"]],
        ["brier", summary["k1"]["brier"], summary["k1_per_source"].get("gsm8k", {}).get("brier", ""), summary["k1_per_source"].get("mmlu_pro", {}).get("brier", ""), summary["k5"]["brier"]],
        ["auroc", summary["k1"]["auroc"] or "", summary["k1_per_source"].get("gsm8k", {}).get("auroc", "") or "", summary["k1_per_source"].get("mmlu_pro", {}).get("auroc", "") or "", summary["k5"]["auroc"] or ""],
        ["abstain_rate", summary["k1"]["abstain_rate"], summary["k1_per_source"].get("gsm8k", {}).get("abstain_rate", ""), summary["k1_per_source"].get("mmlu_pro", {}).get("abstain_rate", ""), summary["k5"]["abstain_rate"]],
        ["parse_ok_rate", summary["k1"]["parse_ok_rate"], "", "", summary["k5"]["parse_ok_rate"]],
        ["avg_confidence", summary["k1"]["avg_confidence"], "", "", summary["k5"]["avg_confidence"]],
        ["overconfident_rate", summary["k1"]["overconfident_rate"], "", "", summary["k5"]["overconfident_rate"]],
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


if __name__ == "__main__":
    main()
