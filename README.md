# Movie Recommendation System

A scalable, end-to-end movie recommendation engine built with Python — covering popularity-based filtering, collaborative filtering, and a deployable web application. Powered by the MovieLens 100K dataset.

---

## Project Structure

```
Recommender-System/
├── data/ml-100k/        
├── notebooks/           
├── src/                
├── app/                
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Language:** Python
- **Core Libraries:** Pandas, NumPy, Scikit-learn
- **Notebook Environment:** Jupyter

---

## Development Phases

### Phase 1 — Data Ingestion
- Loaded the MovieLens 100K dataset
- Merged ratings and movie metadata into a unified dataframe for downstream processing

### Phase 2 — Popularity-Based Recommender
- Built a baseline recommender surfacing the top 10 movies by average user rating
- Serves as a cold-start fallback for new users with no interaction history

**Sample Output:**
```
Top 10 Recommended Movies:
1. The Shawshank Redemption     Avg Rating: 4.4
2. Star Wars                    Avg Rating: 4.3
3. The Godfather                Avg Rating: 4.3
```

### Phase 3 — Collaborative Filtering
- Constructed a user-item interaction matrix
- Applied cosine similarity to identify users with overlapping taste profiles
- Implemented a weighted scoring model to generate personalized recommendations

**Scoring Formula:**
```
score = similarity_weight * user_rating
```

This approach prioritizes recommendations from users most similar to the target user, improving precision over naive averaging.

---

## Getting Started

```bash
git clone https://github.com/Manika-Saini/Recommender-System.git
cd Recommender-System
pip install -r requirements.txt
```

To explore the notebooks:
```bash
jupyter notebook notebooks/
```

---

## Roadmap

- [ ] Matrix factorization (SVD / ALS)
- [ ] Hybrid filtering (content-based + collaborative)
- [ ] Evaluation metrics (Precision@K, NDCG, MAP)
- [ ] REST API layer for serving recommendations

---

## Dataset

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) provided by GroupLens Research at the University of Minnesota.
