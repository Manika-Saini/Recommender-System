import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ratings = pd.read_csv(
    os.path.join(BASE_DIR, "data/ml-100k/u.data"),
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)
movies = pd.read_csv(
    os.path.join(BASE_DIR, "data/ml-100k/u.item"),
    sep="|",
    encoding="latin-1",
    names=[
        "movie_id", "title", "release_date", "video_release_date",
        "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
        "Children", "Comedy", "Crime", "Documentary", "Drama",
        "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
)
data = pd.merge(ratings, movies, on="movie_id")
user_movie_matrix = data.pivot_table(
    index="user_id",
    columns="title",
    values="rating"
).fillna(0)
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)
genre_cols = movies.columns[5:]
def create_genre_text(row):
    genres = []
    for genre in genre_cols:
        if row[genre] == 1:
            genres.append(genre)
    return " ".join(genres)
movies["genres"] = movies.apply(create_genre_text, axis=1)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies["genres"])
content_similarity = cosine_similarity(tfidf_matrix)
def hybrid_recommend(user_id, n=5):
    if user_id not in user_movie_matrix.index:
        return []
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
    top_users = similar_users.head(10)
    user_movies = user_movie_matrix.loc[user_id]
    watched = user_movies[user_movies > 0].index
    collab_scores = {}
    for other_user, sim_score in top_users.items():
        other_movies = user_movie_matrix.loc[other_user]
        liked_movies = other_movies[other_movies > 3]
        for movie, rating in liked_movies.items():
            if movie not in watched:
                if movie not in collab_scores:
                    collab_scores[movie] = 0
                collab_scores[movie] += sim_score * rating
    content_scores = {}
    for movie in watched:
        idx_list = movies[movies["title"].str.strip() == movie.strip()].index
        if len(idx_list) == 0:
            continue
        idx = idx_list[0]
        sim_scores = list(enumerate(content_similarity[idx]))
        for i, score in sim_scores:
            title = movies.iloc[i]["title"]
            if title not in watched:
                if title not in content_scores:
                    content_scores[title] = 0
                content_scores[title] += score
    final_scores = {}
    all_movies = set(collab_scores.keys()).union(content_scores.keys())
    for movie in all_movies:
        final_scores[movie] = (
            0.6 * collab_scores.get(movie, 0) +
            0.4 * content_scores.get(movie, 0)
        )
    sorted_movies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, score in sorted_movies[:n]]