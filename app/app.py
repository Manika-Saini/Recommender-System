import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hybrid_recommender import hybrid_recommend   
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System")
st.write("Get personalized movie recommendations")
user_id = st.number_input("Enter User ID", min_value=1, step=1)
if st.button("Recommend"):
    try:
        recs = hybrid_recommend(user_id)

        if recs and len(recs) > 0:
            st.subheader("Recommended Movies:")
            for i, movie in enumerate(recs, 1):
                st.write(f"{i}. {movie}")
        else:
            st.warning("No recommendations found for this user.")

    except Exception as e:
        st.error(f"Error: {str(e)}")