"""Movie recommender via matrix factorization (surprise.SVD) on MovieLens.

Entry paths:
  python main.py --rating-id 611       # collaborative (SVD, rating-based) top-10
  python main.py --genre-id 611        # content-based (genre) top-10
  python main.py --hybrid-id 611       # hybrid blend of the two (--alpha to tune)
  python main.py --personas            # collaborative vs content-based, per persona
  python main.py --experiments         # run experiments 1-3, write figures/ + results.txt
  python main.py --search "matrix"     # find movieIds by title substring

Core model is collaborative filtering via matrix factorization: R ~= P x Q^T,
learned by regularized SGD, trained once on a static snapshot (limitation: cold
start). The genre and hybrid models are content-based / blended contrasts.
"""

import argparse
import os

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy

import content
import plots

# --- locked design constants (see memory / CLAUDE.md) ---------------------
SEED = 42                 # one global seed -> reproducible split & figures
RELEVANT_THRESHOLD = 4.0  # a held-out movie is "relevant" if rated >= this
TEST_FRAC = 0.2           # per-user fraction held out for the test set
MIN_RATINGS = 5           # users need >= this many ratings to enter the eval
MIN_SUPPORT = 10          # a movie is recommendable only if >= this many train ratings
K_AT = 10                 # precision@K and display N
K_SWEEP = [2, 5, 10, 20, 50, 100]
DEFAULT_K = 50            # n_factors for the single-user recommend paths
RATING_SCALE = (0.5, 5.0)

DATA_DIR = "ml-latest-small"


# --- data -----------------------------------------------------------------
def load_data():
    """Load MovieLens ratings merged with the personas/personal ratings.

    Never edits ratings.csv; my_ratings.csv is concatenated in code so the dataset
    stays reproducible and custom users are kept separate from the original data.
    """
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))[
        ["userId", "movieId", "rating"]
    ]
    if os.path.exists("my_ratings.csv"):
        # comment="#" lets each row carry a trailing "# Title" note for readability
        mine = pd.read_csv("my_ratings.csv", comment="#")[["userId", "movieId", "rating"]]
        ratings = pd.concat([ratings, mine], ignore_index=True)
    return ratings


def load_titles():
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    return dict(zip(movies.movieId, movies.title))


# --- per-user split (serves both RMSE and precision@k) --------------------
def per_user_split(df, test_frac=TEST_FRAC, min_ratings=MIN_RATINGS, seed=SEED):
    """Hold out `test_frac` of EACH eligible user's ratings into the test set.

    Users with < min_ratings stay entirely in train (too sparse to hold out).
    Returns (train_df, test_df).
    """
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, group in df.groupby("userId"):
        if len(group) < min_ratings:
            continue
        n_test = max(1, int(round(len(group) * test_frac)))
        test_idx.extend(rng.choice(group.index.values, size=n_test, replace=False))
    test_df = df.loc[test_idx]
    train_df = df.drop(index=test_idx)
    return train_df, test_df


def build_model(train_df, n_factors, n_epochs=20, seed=SEED):
    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(train_df[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=seed)
    algo.fit(trainset)
    return algo, trainset


# --- recommendation scoring (fast path via raw SVD factors) ---------------
def supported_items(train_df, min_support=MIN_SUPPORT):
    """Set of movieIds with >= min_support ratings in train (recommendable items).

    Filtering out long-tail items stops predicted-rating ranking from surfacing
    niche, high-variance movies that a few ratings can inflate. Standard practice
    for top-N evaluation; applied identically to SVD and the baseline for fairness.
    """
    counts = train_df.groupby("movieId").size()
    return set(counts[counts >= min_support].index)


def topk_svd(algo, trainset, raw_uids, k=K_AT, allowed=None):
    """Top-k unrated, supported items per user, computed from SVD factors directly.

    Ranks by predicted rating (global_mean + b_u + b_i + q_i . p_u), vectorized
    over all items -- far faster than calling algo.predict() in a loop over ~9.7k
    movies. `allowed` (set of movieIds) restricts candidates; cold users skipped.
    """
    item_base = trainset.global_mean + algo.bi  # (n_items,)
    out = {}
    for ruid in raw_uids:
        try:
            iuid = trainset.to_inner_uid(ruid)
        except ValueError:
            continue  # cold user: no learned vector (named limitation)
        scores = item_base + algo.bu[iuid] + algo.qi @ algo.pu[iuid]
        rated = {iid for (iid, _) in trainset.ur[iuid]}
        recs = []
        for iid in np.argsort(-scores):
            if iid in rated:
                continue
            raw = trainset.to_raw_iid(int(iid))
            if allowed is not None and raw not in allowed:
                continue
            recs.append(raw)
            if len(recs) >= k:
                break
        out[ruid] = recs
    return out


def svd_scores(algo, trainset, user_id, allowed=None):
    """Predicted rating for every unrated, supported movie. Returns {movieId: score}.

    Raises ValueError (via to_inner_uid) if the user is cold (no learned vector).
    """
    iuid = trainset.to_inner_uid(user_id)
    scores = trainset.global_mean + algo.bi + algo.bu[iuid] + algo.qi @ algo.pu[iuid]
    rated = {iid for (iid, _) in trainset.ur[iuid]}
    out = {}
    for iid in trainset.all_items():
        if iid in rated:
            continue
        raw = int(trainset.to_raw_iid(iid))
        if allowed is not None and raw not in allowed:
            continue
        out[raw] = float(scores[iid])
    return out


def topk_popular(train_df, trainset, raw_uids, k=K_AT, allowed=None):
    """Popularity baseline: most-rated movies (by count), minus already-rated."""
    counts = train_df.groupby("movieId").size().sort_values(ascending=False)
    if allowed is not None:
        counts = counts[counts.index.isin(allowed)]
    popular = counts.index.tolist()
    out = {}
    for ruid in raw_uids:
        try:
            iuid = trainset.to_inner_uid(ruid)
            rated = {trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[iuid]}
        except ValueError:
            rated = set()
        out[ruid] = [m for m in popular if m not in rated][:k]
    return out


# --- evaluation -----------------------------------------------------------
def relevant_sets(test_df, threshold=RELEVANT_THRESHOLD):
    """Map userId -> set of movieIds they rated >= threshold in the test set."""
    rel = test_df[test_df.rating >= threshold]
    return {uid: set(g.movieId) for uid, g in rel.groupby("userId")}


def precision_at_k(topk_dict, relevant, k=K_AT):
    """Mean precision@k, averaged ONLY over users with >=1 relevant test item.

    Returns (mean_precision, n_users_averaged).
    """
    scores = []
    for uid, recs in topk_dict.items():
        rel = relevant.get(uid)
        if not rel:
            continue  # exclude zero-relevant users (precision is meaningless)
        hits = sum(1 for m in recs[:k] if m in rel)
        scores.append(hits / k)
    return (float(np.mean(scores)), len(scores)) if scores else (0.0, 0)


def eval_users_in(trainset, relevant):
    """Users that have a learned vector AND >=1 relevant held-out item."""
    users = []
    for uid in relevant:
        try:
            trainset.to_inner_uid(uid)
            users.append(uid)
        except ValueError:
            continue
    return users


def svd_rmse(algo, test_df):
    """RMSE of the fitted SVD over the held-out ratings (unknown ids -> global mean)."""
    testset = list(zip(test_df.userId, test_df.movieId, test_df.rating))
    return accuracy.rmse(algo.test(testset), verbose=False)


def baseline_rmses(train_df, test_df):
    """RMSE of mean-based baselines that CAN predict ratings (SVD's honest rivals).

    global-mean: predict the overall train mean for everything.
    user-mean:   predict each user's own train average (cold -> global).
    item-mean:   predict each movie's train average (cold -> global).
    """
    truth = test_df.rating.to_numpy()
    gmean = train_df.rating.mean()
    umean = train_df.groupby("userId").rating.mean()
    imean = train_df.groupby("movieId").rating.mean()

    def rmse(pred):
        return float(np.sqrt(np.mean((pred - truth) ** 2)))

    return {
        "global-mean": rmse(np.full(len(test_df), gmean)),
        "user-mean": rmse(test_df.userId.map(umean).fillna(gmean).to_numpy()),
        "item-mean": rmse(test_df.movieId.map(imean).fillna(gmean).to_numpy()),
    }


# --- experiments ----------------------------------------------------------
def experiment_baseline(train_df, test_df, log):
    """Experiment 1: SVD vs baselines.

    Primary (RMSE): SVD vs global/user/item-mean -> the metric SVD optimizes; it wins.
    Secondary (precision@10): SVD vs popularity -> honest caveat; popularity wins on
    top-N because random hold-out favors popular movies (known result).
    """
    algo, trainset = build_model(train_df, n_factors=DEFAULT_K)

    # --- primary: RMSE ---
    rmses = baseline_rmses(train_df, test_df)
    rmses[f"SVD (k={DEFAULT_K})"] = svd_rmse(algo, test_df)
    log("[Exp 1a] RMSE  " + "  ".join(f"{k}={v:.4f}" for k, v in rmses.items()))
    plots.plot_rmse_comparison(rmses)

    # --- secondary: precision@10 (the honest caveat) ---
    relevant = relevant_sets(test_df)
    users = eval_users_in(trainset, relevant)
    allowed = supported_items(train_df)
    svd_p, n = precision_at_k(topk_svd(algo, trainset, users, allowed=allowed), relevant)
    pop_p, _ = precision_at_k(topk_popular(train_df, trainset, users, allowed=allowed), relevant)
    log(f"[Exp 1b] precision@{K_AT} over {n} users: "
        f"SVD={svd_p:.4f}  popularity={pop_p:.4f}  (popularity wins on top-N)")
    plots.plot_precision_comparison(pop_p, svd_p, n, DEFAULT_K, K_AT)


def experiment_ksweep(train_df, test_df, log):
    """Experiment 2: RMSE vs k (other hyperparams at surprise defaults). Lower is better."""
    results = []
    for k in K_SWEEP:
        algo, _ = build_model(train_df, n_factors=k)
        r = svd_rmse(algo, test_df)
        results.append((k, r))
        log(f"[Exp 2] k={k:>3}  RMSE={r:.4f}")
    best_k = min(results, key=lambda r: r[1])[0]
    log(f"[Exp 2] best k = {best_k} (lowest RMSE)")

    plots.plot_ksweep(results)
    return best_k


def experiment_learning_curve(train_df, test_df, best_k, log, n_eval_users=100):
    """Experiment 3: RMSE vs training-set size, with a FIXED test set.

    A fixed set of `n_eval_users` is always in training (their train portion);
    we grow the number of additional 'background' users from 0 upward and measure
    RMSE on those eval users' held-out ratings (which never change). More users ->
    each movie has more ratings -> better-constrained item vectors -> lower RMSE.
    RMSE has no popularity/catalog-size confound, so the curve is interpretable.
    """
    rng = np.random.default_rng(SEED)
    train_users = set(train_df.userId.unique())
    eval_pool = [u for u in test_df.userId.unique() if u in train_users]
    rng.shuffle(eval_pool)
    eval_users = eval_pool[:n_eval_users]
    eval_test = test_df[test_df.userId.isin(eval_users)]
    eval_set = set(eval_users)
    background = [u for u in train_users if u not in eval_set]
    rng.shuffle(background)

    eval_train = train_df[train_df.userId.isin(eval_users)]
    results = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        n_bg = int(round(len(background) * frac))
        bg_users = background[:n_bg]
        size_train = pd.concat(
            [eval_train, train_df[train_df.userId.isin(bg_users)]], ignore_index=True
        )
        algo, _ = build_model(size_train, n_factors=best_k)
        r = svd_rmse(algo, eval_test)
        total_users = size_train.userId.nunique()
        results.append((total_users, r))
        log(f"[Exp 3] train users={total_users:>3}  RMSE={r:.4f}")

    plots.plot_learning_curve(results, n_eval_users)


def run_experiments():
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    df = load_data()
    train_df, test_df = per_user_split(df)
    log(f"Split: {len(train_df)} train / {len(test_df)} test ratings "
        f"({df.userId.nunique()} users, seed={SEED})")
    experiment_baseline(train_df, test_df, log)
    best_k = experiment_ksweep(train_df, test_df, log)
    experiment_learning_curve(train_df, test_df, best_k, log)

    with open("results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nFigures written to {plots.FIG_DIR}/ ; numbers written to results.txt")


# --- CLI demo path --------------------------------------------------------
def _print_ranked(header, titles):
    """Print a numbered top-N list under a header (shared by the recommend commands)."""
    print(f"\n{header}\n")
    for rank, title in enumerate(titles, 1):
        print(f"  {rank:>2}. {title}")


def recommend(user_id, n=K_AT):
    """Collaborative (SVD, rating-based) recommendations for one user."""
    df = load_data()
    titles = load_titles()
    algo, trainset = build_model(df, n_factors=DEFAULT_K)
    try:
        trainset.to_inner_uid(user_id)
    except ValueError:
        print(f"User {user_id} has no ratings in the data -> cold start: "
              f"no latent vector exists, so no recommendations can be made.\n"
              f"(This is a named limitation of the model, not a bug.)")
        return
    allowed = supported_items(df)
    recs = topk_svd(algo, trainset, [user_id], k=n, allowed=allowed)[user_id]
    _print_ranked(f"Top {n} recommendations for user {user_id} (SVD, k={DEFAULT_K}):",
                  [titles.get(m, f"movieId {m}") for m in recs])


def recommend_genre(user_id, n=K_AT):
    """Content-based (genre) recommendations for one user -- the genre model alone."""
    df = load_data()
    recs = content.content_topk(content.build_content_model(), df, user_id, k=n)
    if not recs:
        print(f"User {user_id} has no liked movies (rating >= {content.LIKE_THRESHOLD}) "
              f"in the data, so no genre profile can be built.")
        return
    _print_ranked(f"Top {n} content-based (genre) recommendations for user {user_id}:",
                  [title for _, title in recs])


def _minmax(scores, keys):
    """Rescale {key: value} to [0, 1] over `keys` (so two score scales become comparable)."""
    vals = [scores[k] for k in keys]
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    return {k: (scores[k] - lo) / span for k in keys}


def recommend_hybrid(user_id, alpha=0.5, n=K_AT):
    """Hybrid recs: blend the rating model and the genre model (how real systems work).

    Each model scores on a different scale (SVD ~ predicted rating; content ~ cosine),
    so we min-max each to [0,1] over the shared candidates, then blend:
        score = alpha * rating + (1 - alpha) * genre
    alpha=1 -> pure rating model, alpha=0 -> pure genre model.
    """
    df = load_data()
    titles = load_titles()
    algo, trainset = build_model(df, n_factors=DEFAULT_K)
    allowed = supported_items(df)
    try:
        rating = svd_scores(algo, trainset, user_id, allowed=allowed)
    except ValueError:
        print(f"User {user_id} is not in the data -> cold start: no recommendations.")
        return
    genre = content.content_scores(content.build_content_model(), df, user_id)
    if not genre:
        print(f"User {user_id} has no liked movies (rating >= {content.LIKE_THRESHOLD}); "
              f"cannot build the genre half of the hybrid.")
        return

    candidates = rating.keys() & genre.keys()  # well-supported, genre-known, unrated
    r_norm = _minmax(rating, candidates)
    g_norm = _minmax(genre, candidates)
    blended = {m: alpha * r_norm[m] + (1 - alpha) * g_norm[m] for m in candidates}

    top = sorted(blended, key=lambda m: blended[m], reverse=True)[:n]
    _print_ranked(f"Top {n} hybrid recommendations for user {user_id} "
                  f"(alpha={alpha}: {alpha:.0%} rating / {1 - alpha:.0%} genre):",
                  [titles.get(m, f"movieId {m}") for m in top])


def recommend_personas(n=K_AT):
    """Contrast each custom persona's recommendations: collaborative vs content-based.

    Collaborative filtering (SVD) ranks by predicted rating -> genre-blind, leans
    toward broadly-acclaimed films. Content-based ranks by genre similarity ->
    genre-coherent by construction. Side by side, this shows what CF does NOT do.
    """
    if not os.path.exists("my_ratings.csv"):
        print("No my_ratings.csv found.")
        return
    df = load_data()
    titles = load_titles()
    algo, trainset = build_model(df, n_factors=DEFAULT_K)
    allowed = supported_items(df)
    content_model = content.build_content_model()
    custom = sorted(pd.read_csv("my_ratings.csv", comment="#").userId.unique())
    cf = topk_svd(algo, trainset, custom, k=n, allowed=allowed)

    for uid in custom:
        cf_recs = [titles.get(m, f"movieId {m}") for m in cf.get(uid, [])]
        cb_recs = [t for _, t in content.content_topk(content_model, df, uid, k=n)]
        print(f"\n{'='*100}\nUser {uid}")
        print(f"{'COLLABORATIVE (SVD, rating-based)':<50}{'CONTENT-BASED (genre-based)'}")
        for i in range(n):
            left = f"{i+1:>2}. {cf_recs[i]}" if i < len(cf_recs) else ""
            right = f"{i+1:>2}. {cb_recs[i]}" if i < len(cb_recs) else ""
            print(f"{left:<50}{right}")


def search_titles(query):
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    hits = movies[movies.title.str.contains(query, case=False, na=False)]
    if hits.empty:
        print(f"No titles matching {query!r}")
        return
    for _, row in hits.head(30).iterrows():
        print(f"  {row.movieId:>6}  {row.title}")


def main():
    p = argparse.ArgumentParser(description="MovieLens SVD recommender")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--rating-id", type=int,
                   help="collaborative (rating-based) top-10 recs for this user")
    g.add_argument("--genre-id", type=int,
                   help="content-based (genre) top-10 recs for this user")
    g.add_argument("--hybrid-id", type=int,
                   help="hybrid (rating+genre blend) top-10 recs for this user")
    g.add_argument("--personas", action="store_true",
                   help="print top-10 recs for every custom user in my_ratings.csv")
    g.add_argument("--experiments", action="store_true",
                   help="run experiments 1-3, write figures/ and results.txt")
    g.add_argument("--search", type=str, metavar="QUERY",
                   help="find movieIds by title substring (for my_ratings.csv)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="hybrid blend weight: 1=pure rating, 0=pure genre (default 0.5)")
    args = p.parse_args()

    if args.experiments:
        run_experiments()
    elif args.personas:
        recommend_personas()
    elif args.search:
        search_titles(args.search)
    elif args.genre_id is not None:
        recommend_genre(args.genre_id)
    elif args.hybrid_id is not None:
        recommend_hybrid(args.hybrid_id, args.alpha)
    else:
        recommend(args.rating_id)


if __name__ == "__main__":
    main()
