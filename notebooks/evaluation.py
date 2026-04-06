import pandas as pd
from hybrid_recommender import hybrid_recommend

ratings = pd.read_csv(
    "../data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "../data/ml-100k/u.item",
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

def precision_at_k(user_id, k=5):
    recommended_titles = hybrid_recommend(user_id, n=k)
    recommended_ids = movies[movies["title"].str.strip().isin([t.strip() for t in recommended_titles])]["movie_id"].tolist()
    actual = ratings[(ratings["user_id"] == user_id) &(ratings["rating"] >= 4)]["movie_id"].tolist()
    if len(recommended_ids) == 0:
        return 0
    common = set(recommended_ids).intersection(set(actual))
    print("\n DEBUG PRECISION")
    print("Recommended IDs:", recommended_ids[:10])
    print("Actual IDs:", actual[:10])
    print("Intersection:", common)
    return len(common) / len(recommended_ids)

def recall_at_k(user_id, k=5):
    recommended_titles = hybrid_recommend(user_id, n=k)
    recommended_ids = movies[movies["title"].str.strip().isin([t.strip() for t in recommended_titles])]["movie_id"].tolist()
    actual = ratings[(ratings["user_id"] == user_id) &(ratings["rating"] >= 4)]["movie_id"].tolist()
    if len(actual) == 0:
        return 0
    common = set(recommended_ids).intersection(set(actual))
    return len(common) / len(actual)

user_id = 1

print(f"\nEvaluation for user {user_id}:\n")
print("Precision:", precision_at_k(user_id))
print("Recall:", recall_at_k(user_id))