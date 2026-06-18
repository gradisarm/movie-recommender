"""Content-based recommender — the contrast to the collaborative-filtering SVD.

Where SVD learns from rating co-occurrence (and is genre-blind), this recommends
by CONTENT similarity: each movie is a TF-IDF-weighted genre vector, a user's
profile is the (rating-weighted) average of the genres they liked, and movies
are scored by cosine similarity to that profile. Genre-coherent by construction
— it demonstrates exactly what collaborative filtering does NOT do.

TF-IDF over genres reuses the same idea as the earlier spam-classifier project:
down-weight ubiquitous genres (Drama) and reward distinctive ones (Horror, Western).
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = "ml-latest-small"
LIKE_THRESHOLD = 4.0  # only "liked" movies shape the content profile


def build_content_model():
    """Return TF-IDF genre vectors (row-normalized) + lookup tables."""
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    movies = movies[movies.genres != "(no genres listed)"].reset_index(drop=True)

    onehot = movies.genres.str.get_dummies(sep="|")
    # IDF: rarer genres carry more signal than ubiquitous ones.
    doc_freq = onehot.sum(axis=0).to_numpy()
    idf = np.log(len(movies) / doc_freq)
    mat = onehot.to_numpy(dtype=float) * idf

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms  # unit rows -> dot product == cosine

    movie_ids = movies.movieId.to_numpy()
    return {
        "mat": mat,
        "movie_ids": movie_ids,
        "id_to_row": {int(mid): i for i, mid in enumerate(movie_ids)},
        "titles": dict(zip(movies.movieId, movies.title)),
    }


def content_scores(model, ratings_df, user_id):
    """Genre-cosine similarity of every unrated movie to the user's liked-movie profile.

    Returns {movieId: similarity in [0, 1]}; empty if the user has no liked movies.
    """
    id_to_row = model["id_to_row"]
    user = ratings_df[ratings_df.userId == user_id]
    liked = user[(user.rating >= LIKE_THRESHOLD) & (user.movieId.isin(id_to_row))]
    if liked.empty:
        return {}

    rows = [id_to_row[int(m)] for m in liked.movieId]
    profile = np.average(model["mat"][rows], axis=0, weights=liked.rating.to_numpy())
    pn = np.linalg.norm(profile)
    if pn:
        profile = profile / pn

    sims = model["mat"] @ profile
    rated = set(user.movieId)
    ids = model["movie_ids"]
    return {int(ids[i]): float(sims[i]) for i in range(len(ids)) if int(ids[i]) not in rated}


def content_topk(model, ratings_df, user_id, k=10):
    """Top-k movies by genre-cosine similarity to the user's liked-movie profile."""
    scores = content_scores(model, ratings_df, user_id)
    titles = model["titles"]
    top = sorted(scores, key=lambda m: scores[m], reverse=True)[:k]
    return [(m, titles.get(m, f"movieId {m}")) for m in top]
