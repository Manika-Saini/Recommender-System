import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings=pd.read_csv(
    "../data/ml-100k/u.data",
    sep="\t",
    names=["user_id","movie_id","rating","timestamp"]
)

movies=pd.read_csv(
    "../data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    names=[
        "movie_id","title","release_date","video_release_date",
        "IMDb_URL","unknown","Action","Adventure","Animation",
        "Children","Comedy","Crime","Documentary","Drama",
        "Fantasy","Film-Noir","Horror","Musical","Mystery",
        "Romance","Sci-Fi","Thriller","War","Western"
    ]
)

data=pd.merge(ratings,movies,on="movie_id")

user_movie_matrix=data.pivot_table(
    index="user_id",
    columns="title",
    values="rating"
).fillna(0)


user_similarity=cosine_similarity(user_movie_matrix)
user_similarity_df=pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def recommend_movies(user_id,n_recommendations=5):
    similar_users=user_similarity_df[user_id].sort_values(ascending=False)
    similar_users=similar_users.drop(user_id)
    top_users=similar_users.head(10)
    user_movies=user_movie_matrix.loc[user_id]
    watched_movies=user_movies[user_movies>0].index
    movie_scores={}
    for other_user, similarity_score in top_users.items():
        other_movies=user_movie_matrix.loc[other_user]
        liked_movies=other_movies[other_movies>3]
        for movie, rating in liked_movies.items():
            if movie not in watched_movies:
                if movie not in movie_scores:
                    movie_scores[movie]=0
                movie_scores[movie] +=similarity_score*rating
    sorted_movies = sorted(movie_scores.items(),key=lambda x: x[1], reverse=True)
    recommendations =[movie for movie,score in sorted_movies[:n_recommendations]]
    return recommendations

user_id=1
print(f"\n Recommendation for user {user_id}:\n")
recs=recommend_movies(user_id)
if recs:
    for i,movie in enumerate(recs,start=1):
        print(f"{i}.{movie}")
else:
    print("No recommendations found")