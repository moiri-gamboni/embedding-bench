# Embedding Model Benchmark Explorer

Interactive browser-based tool for comparing sub-1B parameter open embedding models across multiple retrieval benchmarks.

## Benchmarks

- **Prose** -- general text retrieval (ArguAna, FiQA, TRECCOVID, etc.)
- **RTEB Code** -- code retrieval (HumanEval, MBPP, DS1000, etc.)
- **CoIR** -- code information retrieval (CodeSearchNet, CosQA, StackOverflowQA, etc.)
- **Instruct** -- instruction-following retrieval
- **LongCtx** -- long context retrieval (LEMB suite)
- **Reason** -- temporal reasoning retrieval

All scores are NDCG@10.

## Usage

Open `index.html` in a browser. The data is bundled in `models.js` so it works from `file://` with no server.

## Updating the data

Pull fresh results from the [MTEB results repo](https://github.com/embeddings-benchmark/results):

```bash
git clone --depth 1 https://github.com/embeddings-benchmark/results /tmp/mteb-results
python pull_results.py
```

Optionally merge RTEB Code scores from a HuggingFace leaderboard CSV export:

```bash
python pull_results.py --csv ~/Downloads/rteb-code-leaderboard.csv
```
