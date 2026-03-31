import pandas as pd

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

print("Number of users:",ratings["user_id"].nunique())
print("Number of movies:",ratings["movie_id"].nunique())

print("\nSample Data:\n")
print(data.head())


avg_rating=data.groupby("title")["rating"].mean()
rating_count=data.groupby("title")["rating"].count()
movie_stats=pd.DataFrame({
    "avg_rating": avg_rating,
    "rating_count": rating_count
})

movie_stats=movie_stats[movie_stats["rating_count"]>50]
top_movies=movie_stats.sort_values(
    by=["avg_rating","rating_count"],
    ascending=False
)

print("\n Top Recommended Movies:\n")
for i,(title,row) in enumerate(top_movies.head(10).iterrows(),start=1):
    print(f"{i}.{title} ⭐ {round(row["avg_rating"],2)} ({row["rating_count"]} ratings)")
