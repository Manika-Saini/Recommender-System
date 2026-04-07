import streamlit as st
from hybrid_recommender import hybrid_recommend
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("Movie Recommender System")
st.write("Get personalized movie recommendations")
user_id=st.number_input("Enter user ID",min_value=1,step=1)
if st.button("Recommend"):
    try:
        recs=hybrid_recommend(user_id)
        if recs:
            st.subheader("Recommended Movies:")
            for i,movie in enumerate(recs,1):
                st.write(f"{i}.{movie}")
        else:
            st.warning("No recommendations found")
    except Exception as e:
        st.error(f"Error: {e}")
