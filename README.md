# Movie Recommender (Matrix Factorization + Content-Based + Hybrid)

A movie recommender built to explore **how recommendation systems (YouTube,
Netflix, …) actually work**, on the MovieLens `ml-latest-small` dataset
(~100k ratings, 610 users, ~9.7k movies). It implements the three building
blocks real systems combine:

1. **Collaborative filtering** (`main.py`, `surprise.SVD`) — *"people whose
   ratings resemble yours also liked this."* Matrix factorization: learns
   `R ≈ P · Qᵀ`, a `k`-dim latent vector per user and per movie via regularized
   SGD. The core of the project; trained once on a static snapshot. **Genre-blind.**
2. **Content-based** (`content.py`) — *"this is similar to what you liked."*
   TF-IDF–weighted genre vectors; recommends by cosine similarity to a profile
   built from the user's liked movies. **Genre-coherent, but quality-blind.**
3. **Hybrid** — blends the two (min-max normalize each, then
   `α·rating + (1−α)·genre`). What real systems actually do. **Both genre-aware
   and quality-aware.**

## Setup

`scikit-surprise` is a Cython package with **no wheels for Python 3.14**, so the
project uses **Python 3.12** in a virtual environment. Nothing is installed globally.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # or call .venv/bin/python directly
pip install -r requirements.txt
```

## Usage

```bash
python main.py --rating-id 612          # collaborative (rating-based) top-10
python main.py --genre-id 612           # content-based (genre) top-10
python main.py --hybrid-id 612 --alpha 0.5   # hybrid blend (α: 1=rating, 0=genre)
python main.py --personas               # collaborative vs content-based, every persona
python main.py --experiments            # run experiments, write figures/ + results.txt
python main.py --search "blade runner"  # find movieIds by title substring
```

The three single-user flags let you compare the models on the same user:
`--rating-id` returns genre-blind acclaimed films, `--genre-id` returns on-genre
(but obscure) films, `--hybrid-id` returns on-genre *and* well-regarded films.

## Data

`ml-latest-small/` holds the untouched GroupLens dataset. **`ratings.csv` and
`movies.csv` are never modified** (only `ratings` and the `genres`/`title`
columns of `movies` are used; `tags.csv`/`links.csv` are unused). Custom users
live in `my_ratings.csv` (`userId,movieId,rating`, rows may carry a trailing
`# Title` comment) and are concatenated in code before training — keeping the
dataset reproducible and custom users separate.

`my_ratings.csv` ships with five **demonstration personas** (userIds 611–615),
each shaped around a deliberate taste so the output is predictable:

| user | persona | expectation |
|------|---------|-------------|
| 611 | cerebral / arthouse | dark, intelligent films |
| 612 | action junkie | action / blockbuster |
| 613 | animation / family | Pixar / Disney / Ghibli |
| 614 | horror fan | horror |
| 615 | "all over the place" | mixed genres, no clear signal |

## Evaluation (collaborative model)

A **per-user hold-out split** (20% of each user's ratings, users with ≥5
ratings, fixed seed) feeds two metrics:

- **RMSE (headline):** SVD vs mean-based baselines that can also predict ratings.
- **precision@10 (secondary):** SVD vs a popularity baseline; relevant = rated ≥ 4.0,
  averaged over users with ≥1 relevant held-out item, candidates filtered to
  movies with ≥10 ratings.

| # | Experiment | Result |
|---|------------|--------|
| 1 | **RMSE vs baselines** | SVD **~0.88** beats user-mean ~0.94, item-mean ~0.97, global-mean ~1.05 |
| 1 | **precision@10 vs popularity** | SVD **loses** to popularity (see note) |
| 2 | **k-sweep (RMSE vs k)** | best around **k=20**; rises by k=100 — under/overfit tradeoff |
| 3 | **Learning curve (RMSE vs #users)** | RMSE falls as users grow — more data, better predictions |

Figures (labels in **Slovene**) land in `figures/`: `rmse_comparison.png`,
`k_sweep.png`, `learning_curve.png`, `precision_comparison.png`. The same run
writes the numbers to `results.txt`. (Exact values depend on how many personas
are in `my_ratings.csv`, since they are merged into training.)

### Why the collaborative model "loses" on precision@10

A known result, not a bug. Random per-user hold-out makes *popular* movies
disproportionately likely to land in the test set, so always recommending
popular movies scores well on top-N precision. SVD ranks by predicted *rating*
(quality), a different signal than *which movies a user engages with*
(popularity). A minimum-support filter barely moved it — confirming the cause is
the metric's popularity bias. Hence RMSE (what SVD optimizes) is the honest
headline metric.

### What the three models reveal

- **Collaborative** predicts ratings well (wins RMSE) but is **genre-blind** —
  the action persona does *not* get action back, because CF never reads genres.
- **Content-based** is **genre-coherent** (action→action) but **quality-blind**,
  surfacing obscure single-genre films.
- **Hybrid** combines them: action films that are also well-regarded. Sweeping
  `--alpha` visibly shifts recommendations from genre-forward to acclaimed-forward.

## Limitations

- **Cold start:** a user/movie absent from training has no latent vector and no
  fallback (`--rating-id` for an unknown user prints a cold-start message).
- **Static snapshot:** trains once; production systems retrain for concept drift.
- **Small dataset:** on `ml-latest-small` the latent factors are dominated by
  overall popularity rather than genre — larger data would carry more taste structure.
- **Classical by design:** matrix factorization + TF-IDF, no neural net / GPU.

## Layout

```
movie-recommender/
├── ml-latest-small/   # untouched dataset
├── my_ratings.csv     # custom users / personas (separate from the dataset)
├── main.py            # data, split, SVD, evaluation, hybrid, CLI
├── content.py         # content-based (genre) recommender
├── plots.py           # figure generation (matplotlib, Agg, Slovene labels)
├── figures/           # generated report figures
├── results.txt        # generated metrics
├── REPORT_NOTES.md    # writeup material (numbers + rationale)
├── requirements.txt
└── README.md
```
