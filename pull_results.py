#!/usr/bin/env python3
"""
Pull MTEB/RTEB benchmark results for sub-1B open embedding models.

Reads from:
  - github.com/embeddings-benchmark/results (cloned locally)
  - Optional RTEB Code CSV export from HuggingFace leaderboard

Outputs: models.json (array of model objects with per-task scores)

Usage:
  # First time: clone the results repo
  git clone --depth 1 https://github.com/embeddings-benchmark/results /tmp/mteb-results

  # Run with just the repo
  python pull_results.py

  # Run with RTEB CSV to fill gaps
  python pull_results.py --csv ~/Downloads/rteb-code-leaderboard.csv

  # Custom repo path
  python pull_results.py --repo /path/to/mteb-results/results
"""

import argparse
import csv
import json
import math
import os
import sys

# === Benchmark task lists ===
# Each benchmark is a list of task names matching JSON filenames in the results repo.

BENCHMARKS = {
    "Prose": [
        "ArguAna", "ClimateFEVERHardNegatives", "FEVERHardNegatives", "FiQA2018",
        "HotpotQAHardNegatives", "TRECCOVID", "Touche2020Retrieval.v3", "SCIDOCS",
        "CQADupstackGamingRetrieval", "CQADupstackUnixRetrieval",
    ],
    "RTEB_Code": [
        "AppsRetrieval", "Code1Retrieval", "DS1000Retrieval", "FreshStackRetrieval",
        "HumanEvalRetrieval", "MBPPRetrieval", "WikiSQLRetrieval",
    ],
    "CoIR": [
        "COIRCodeSearchNetRetrieval", "CodeFeedbackMT", "CodeFeedbackST",
        "CodeSearchNetCCRetrieval", "CodeTransOceanContest", "CodeTransOceanDL",
        "CosQA", "StackOverflowQA", "SyntheticText2SQL",
    ],
    "Instruct": [
        "Core17InstructionRetrieval", "News21InstructionRetrieval",
        "Robust04InstructionRetrieval",
    ],
    "LongCtx": [
        "LEMBNarrativeQARetrieval", "LEMBNeedleRetrieval", "LEMBPasskeyRetrieval",
        "LEMBQMSumRetrieval", "LEMBSummScreenFDRetrieval", "LEMBWikimQARetrieval",
        "MultiLongDocRetrieval",
    ],
    "Reason": [
        "TempReasonL1", "TempReasonL2Context", "TempReasonL2Fact", "TempReasonL2Pure",
        "TempReasonL3Context", "TempReasonL3Fact", "TempReasonL3Pure",
    ],
}

RTEB_CSV_COLS = [
    "AppsRetrieval", "Code1Retrieval", "DS1000Retrieval", "FreshStackRetrieval",
    "HumanEvalRetrieval", "MBPPRetrieval", "WikiSQLRetrieval",
]


def safe_number(val):
    """Return None for inf/nan, otherwise return the value."""
    if val is None:
        return None
    try:
        if math.isinf(val) or math.isnan(val):
            return None
    except TypeError:
        pass
    return val


def get_score(filepath):
    """Extract ndcg_at_10 from a task result JSON file."""
    try:
        with open(filepath) as f:
            d = json.load(f)
        test = d.get("scores", {}).get("test", [])
        if isinstance(test, list) and len(test) > 0:
            return test[0].get("ndcg_at_10", test[0].get("main_score"))
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None


def get_model_meta(model_dir):
    """Read model_meta.json from the first revision directory that has one."""
    for rev in os.listdir(model_dir):
        meta_path = os.path.join(model_dir, rev, "model_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
    return {}


def load_csv_rteb(csv_path):
    """Load RTEB Code scores from HuggingFace leaderboard CSV export.

    CSV values are percentages (e.g. 88.915). Returns dict of
    model_name_lower -> {task: score_as_fraction}.
    """
    result = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            model = row.get("Model", "").strip()
            if not model:
                continue
            scores = {}
            for col in RTEB_CSV_COLS:
                val = row.get(col, "").strip()
                if val:
                    try:
                        scores[col] = float(val) / 100.0
                    except ValueError:
                        pass
            if scores:
                result[model.lower()] = scores
    return result


def collect_results(repo_dir, csv_rteb, max_params=1_000_000_000):
    """Walk the results repo and collect benchmark scores for qualifying models."""
    results = []
    seen = set()

    for dir_name in sorted(os.listdir(repo_dir)):
        model_dir = os.path.join(repo_dir, dir_name)
        if not os.path.isdir(model_dir):
            continue

        meta = get_model_meta(model_dir)
        params = meta.get("n_parameters")
        if params is None or params > max_params:
            continue
        if meta.get("open_weights") is False:
            continue

        full_name = meta.get("name", dir_name.replace("__", "/"))
        short_name = full_name.split("/")[-1] if "/" in full_name else full_name

        if short_name.lower() in seen:
            continue
        seen.add(short_name.lower())

        # Collect scores from all revisions
        bench_data = {}
        for bench_name, task_list in BENCHMARKS.items():
            task_scores = {}
            for rev in os.listdir(model_dir):
                rev_dir = os.path.join(model_dir, rev)
                if not os.path.isdir(rev_dir):
                    continue
                for task in task_list:
                    fpath = os.path.join(rev_dir, f"{task}.json")
                    if os.path.exists(fpath):
                        score = get_score(fpath)
                        if score is not None:
                            task_scores[task] = round(score * 100, 2)
            if task_scores:
                bench_data[bench_name] = task_scores

        # Merge CSV RTEB Code data (fills gaps, repo data takes precedence)
        csv_key = short_name.lower()
        if csv_key in csv_rteb:
            if "RTEB_Code" not in bench_data:
                bench_data["RTEB_Code"] = {}
            for task, score in csv_rteb[csv_key].items():
                if task not in bench_data["RTEB_Code"]:
                    bench_data["RTEB_Code"][task] = round(score * 100, 2)

        if not bench_data:
            continue
        if not any(bn in bench_data for bn in ["Prose", "RTEB_Code", "CoIR"]):
            continue

        entry = {
            "name": short_name,
            "full_name": full_name,
            "params": params,
            "max_tokens": safe_number(meta.get("max_tokens")),
            "embed_dim": safe_number(meta.get("embed_dim")),
            "benchmarks": {},
        }

        for bn, task_scores in bench_data.items():
            entry["benchmarks"][bn] = {
                "scores": task_scores,
                "avg": round(sum(task_scores.values()) / len(task_scores), 2),
                "n": len(task_scores),
                "total": len(BENCHMARKS[bn]),
            }

        results.append(entry)

    results.sort(key=lambda x: x["params"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Pull MTEB/RTEB benchmark results")
    parser.add_argument(
        "--repo",
        default="/tmp/mteb-results/results",
        help="Path to the embeddings-benchmark/results repo 'results' directory",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional RTEB Code CSV export from HuggingFace leaderboard",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=1_000_000_000,
        help="Maximum parameter count (default: 1B)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON path (default: models.json next to this script)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.repo):
        print(f"Error: Results repo not found at {args.repo}", file=sys.stderr)
        print("Clone it first:", file=sys.stderr)
        print(
            "  git clone --depth 1 https://github.com/embeddings-benchmark/results /tmp/mteb-results",
            file=sys.stderr,
        )
        sys.exit(1)

    csv_rteb = {}
    if args.csv:
        if not os.path.isfile(args.csv):
            print(f"Error: CSV file not found at {args.csv}", file=sys.stderr)
            sys.exit(1)
        csv_rteb = load_csv_rteb(args.csv)
        print(f"Loaded {len(csv_rteb)} models from CSV", file=sys.stderr)

    print(f"Scanning {args.repo}...", file=sys.stderr)
    results = collect_results(args.repo, csv_rteb, args.max_params)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also write a .js version loadable via <script> (works with file://)
    js_path = output_path.replace(".json", ".js")
    with open(js_path, "w") as f:
        f.write("const MODEL_DATA = ")
        json.dump(results, f)
        f.write(";\n")

    print(f"Wrote {len(results)} models to {output_path} and {js_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
