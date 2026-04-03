import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

genre_cols=movies.columns[5:]

def create_genre_text(row):
    genres=[]
    for genre in genre_cols:
        if row[genre]==1:
            genres.append(genre)
    return " ".join(genres)

movies["genres"]=movies.apply(create_genre_text,axis=1)


tfidf= TfidfVectorizer()
tfidf_matrix=tfidf.fit_transform(movies["genres"])

cosine_sim=cosine_similarity(tfidf_matrix)

def recommend_similar_movies(movie_title,n=5):
    idx=movies[movies["title"]==movie_title].index[0]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x: x[1], reverse=True)
    sim_scores=sim_scores[1:n+1]
    movie_indices=[i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices]

movie_name = "Star Wars (1977)"
print(f"\n Movies similar to : {movie_name}\n")
recs=recommend_similar_movies(movie_name)
for i, movie in enumerate(recs,start=1):
    print(f"{i}.{movie}")
    