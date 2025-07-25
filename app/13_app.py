from collections import UserDict
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ----------------------
# Paths & Loaders
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_RECS_PATH = os.path.join(BASE_DIR, '../outputs/12_hybrid_combined_classifier_bert_xgboost_ncf.csv')

@st.cache_data
def load_final_recommendations():
    return pd.read_csv(FINAL_RECS_PATH)

df = load_final_recommendations()

# ----------------------
# UI Setup
# ----------------------
st.set_page_config(page_title="Amazon Recommender", page_icon="📚")

st.title("📚 Amazon Product Recommender")
st.markdown("Choose a user or product to get recommendations. This app uses your final hybrid model outputs.")

# Sidebar filters
st.sidebar.header("🔎 Filter Options")
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5)

category_options = df['category'].dropna().unique() if 'category' in df.columns else []
selected_category = st.sidebar.selectbox("Category", ['All'] + sorted(category_options.tolist())) if len(category_options) > 0 else 'All'

# ----------------------
# Tabs
# ----------------------


# ----------------------
# Tab 1: User Recs
# ----------------------
user_id = st.text_input("Enter User ID")

model_choice = st.selectbox("Choose Model", ["Hybrid", "SVD", "BERT", "Sentiment", "XGBoost", "NCF"])


if st.button("Get Recommendations"):
    while True:
        if user_id == "":
            st.write("No user found in database.")
            break

        user_recs = df[df['user_id'] == user_id]
        user_recs = user_recs.rename(
            columns={"svd_rating": "SVD",
                "bert_similarity": "BERT",
                "sentiment_score": "Sentiment",
                "xgb_pred_score": "XGBoost",
                "ncf_score": "NCF"})
        
        user_recs['Hybrid'] = 0.3 * user_recs['XGBoost'] + 0.25 * user_recs['NCF'] + 0.15 * user_recs['BERT'] + 0.15 * user_recs['SVD'] + 0.15 * user_recs ['Sentiment']

        if model_choice in user_recs.columns:
            user_recs = user_recs[user_recs[model_choice] >= min_rating]
            top_recs = user_recs.sort_values(model_choice, ascending=False).head(5)
            if len(user_recs[model_choice]) == 0:
                st.write(f"No recommendations found with {model_choice} scores ≥ {min_rating}. Try lowering the minimum rating.")
                break

        st.subheader(f"Top 5 Recommendations for User {user_id}")
        st.markdown(f" **Min: {user_recs[model_choice].min():.5f}, Max: {user_recs[model_choice].max():.5f}**")
        for _, row in top_recs.iterrows():
            st.markdown(f"**Product ID: {row['asin']}**")
            # st.write(f"🐹 Selected {model_choice} Score: {row.get(model_choice, 0):.5f}")
            st.write(f"📊 Hybrid Score: {row.get('Hybrid', 0):.5f}")
            st.write(f"⭐ SVD: {row.get('SVD', 0):.5f}")
            st.write(f"💬 Sentiment: {row.get('Sentiment', 0):.5f}")
            st.write(f"👅 BERT: {row.get('BERT', 0):.5f}")
            st.write(f"🌲 XGBoost: {row.get('XGBoost', 0):.5f}")
            st.write(f"🧠 NCF: {row.get('NCF', 0):.5f}")
            
            st.write("---")

        # 🔍 Visualization
        st.subheader("Score Comparison")
        plt.figure(figsize=(10, 4))
        plt.bar(top_recs['asin'], top_recs[model_choice], color='skyblue')
        plt.title(f"{model_choice} Scores for Top 5 Products")
        plt.xlabel("Product ID")
        plt.ylabel("Score")
        st.pyplot(plt)

        break