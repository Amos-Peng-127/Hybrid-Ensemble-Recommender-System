import streamlit as st
st.set_page_config(page_title="Amazon Recommender", page_icon="📚")

def main():
    st.title("My Streamlit App")
    
    import os, sys, pickle, warnings
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import gdown

    import sys, os
    import pathlib
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    sys.path.append(str(BASE_DIR))

    from notebooks.helper import (
        load_reviews_df,
        get_svd_predictions_for_user_history,
        calculate_sentiment_for_items,
        calculate_bert_content_similarity,
        get_ncf_predictions,
        calculate_xgboost
    )

    # ---------------------- #
    # 1. Basic Configuration
    # ---------------------- #
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

    st.title("📚 Amazon Product Recommender")
    st.markdown("Choose a user or product to get recommendations. This app uses the final hybrid model outputs.")

    # ---------------------- #
    # 2. File Download & Check
    # ---------------------- #
    FILES = {
        "10_%_raw_data.csv": "https://drive.google.com/uc?id=11VltwluUJR87OvO9v-E6ZRebMTNTq-ge",
        "bert_asins.csv": "https://drive.google.com/uc?id=1hZNdi7EjCyQsdo5USDyMxdSNWQbW5gfb",
        "bert_embeddings.npy": "https://drive.google.com/uc?id=1l5_VKscgAulltGSgZqbwFjqNhB86g-KH",
        "ncf_model.pt": "https://drive.google.com/uc?id=1rDBZp7t-XsJlpp6Kj4upuO2RAkMxaxql",
        "trained_svd_model.pkl": "https://drive.google.com/uc?id=1qoL7nBaIQqTimBZStVld8rCr0oekqd9m",
        "user_item_mappings.pkl": "https://drive.google.com/uc?id=1WipR3wt_XIwsydaSHvlGyEt_rHlglyak",
        "xgboost_model.pkl": "https://drive.google.com/uc?id=1XI-yQ-wu8CSgZhQM91GgH3Ny3XdP0Ixz",
    }
    TARGET_DIR = "resources"

    @st.cache_resource
    def ensure_all_files():
        if not os.path.exists(TARGET_DIR):
            os.makedirs(TARGET_DIR)

        import time
        downloaded_files = []
        for filename, url in FILES.items():
            path = os.path.join(TARGET_DIR, filename)
            if not os.path.exists(path):
                # msg = st.empty()
                # msg.success(f"⬇️ Downloading {filename} ...")
                # time.sleep(3)  # Display for 3 seconds
                # msg.empty()

                gdown.download(url, path, quiet=False)
                downloaded_files.append(filename)

        if downloaded_files:
            msg = st.empty()
            msg.success(f"Downloaded files: {', '.join(downloaded_files)}")

            time.sleep(3)  # Display for 3 seconds
            msg.empty()
        else:
            msg = st.empty()
            msg.success("✅ All files already exist. No download needed.")

            time.sleep(3)  # Display for 3 seconds
            msg.empty()

        return TARGET_DIR

    download_dir = ensure_all_files()

    # ---------------------- #
    # 3. Load Data and Models
    # ---------------------- #
    df = pd.read_csv(os.path.join(download_dir, "10_%_raw_data.csv"), low_memory=False).rename(
        columns={"reviewerID": "user_id", "asin": "item_id", "reviewText": "review_text"}
    )

    svd_model_path = os.path.join(download_dir, "trained_svd_model.pkl")
    bert_vectors = np.load(os.path.join(download_dir, "bert_embeddings.npy"))
    bert_asins = pd.read_csv(os.path.join(download_dir, "bert_asins.csv"))["asin"].tolist()
    bert_item_id_to_idx = {asin: i for i, asin in enumerate(bert_asins)}

    with open(os.path.join(download_dir, "xgboost_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)

    # ---------------------- #
    # 4. UI Setup
    # ---------------------- #
    st.sidebar.header("🔎 Filter Options")
    min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5)
    category_options = df["category"].dropna().unique() if "category" in df.columns else []
    selected_category = (
        st.sidebar.selectbox("Category", ["All"] + sorted(category_options.tolist()))
        if len(category_options) > 0
        else "All"
    )

    user_id = st.text_input("Enter User ID: (For example, 'AAP7PPBU72QFM')")
    model_choice = st.selectbox("Choose Model", ["Hybrid", "SVD", "BERT", "Sentiment", "XGBoost", "NCF"])

    # ---------------------- #
    # 5. Generate Recommendations
    # ---------------------- #
    if st.button("Get Recommendations"):
        
        try:
            
            # --- SVD Predictions ---
            svd_recommended_items_df = get_svd_predictions_for_user_history(user_id, df, df, svd_model_path, n=10)

            # --- Sentiment Analysis ---
            sentiment_df = calculate_sentiment_for_items(df, svd_recommended_items_df)

            # --- BERT Similarity ---
            bert_similarity_df = calculate_bert_content_similarity(
                df, sentiment_df, user_id, bert_vectors, bert_item_id_to_idx
            )
            # st.write(bert_similarity_df.head())

            # --- User Average Ratings ---
            user_avg_rating = df[df["user_id"] == user_id]["overall"].mean()
            bert_similarity_df["user_ave_rating"] = user_avg_rating

            # --- Item Average Ratings ---
            item_avg_ratings = []
            for item_id in bert_similarity_df["item_id"].tolist():
                item_avg_ratings.append(df[df["item_id"] == item_id]["overall"].mean())
            
            bert_similarity_df["product_ave_rating"] = item_avg_ratings
            
            # --- XGBoost Predictions ---
            xgb_predictions_df = calculate_xgboost(bert_similarity_df, xgb_model)
            
            # --- NCF Predictions ---
            ncf_predictions_df = get_ncf_predictions(
                xgb_predictions_df,
                os.path.join(download_dir, "ncf_model.pt"),
                os.path.join(download_dir, "user_item_mappings.pkl"),
                embedding_dim=64,
            )
            
            # --- Rename Columns for Display ---
            user_recs = ncf_predictions_df.rename(
                columns={
                    "svd_rating": "SVD",
                    "bert_similarity": "BERT",
                    "sentiment_score": "Sentiment",
                    "xgb_pred_score": "XGBoost",
                    "ncf_score": "NCF",
                }
            )
            
            if True:
                
                def scale_to_0_5(series):
                    # z-score
                    mean_val, std_val = series.mean(), series.std()
                    if std_val == 0:
                        standardized = series - mean_val
                    else:
                        standardized = (series - mean_val) / std_val
                    
                    # scale 0-5
                    if standardized.max() > standardized.min():
                        return 5 * (standardized - standardized.min()) / (standardized.max() - standardized.min())
                    else:
                        return 2.5
                    
                user_recs['XGBoost_scaled'] = scale_to_0_5(user_recs['XGBoost'])
                user_recs['NCF_scaled'] = scale_to_0_5(user_recs['NCF'])
                user_recs['BERT_scaled'] = scale_to_0_5(user_recs['BERT'])
                user_recs['SVD_scaled'] = scale_to_0_5(user_recs['SVD'])
                user_recs['Sentiment_scaled'] = scale_to_0_5(user_recs['Sentiment'])

                user_recs['Hybrid'] = (
                    0.3 * user_recs['XGBoost_scaled'] +
                    0.25 * user_recs['NCF_scaled'] +
                    0.15 * user_recs['BERT_scaled'] +
                    0.15 * user_recs['SVD_scaled'] +
                    0.15 * user_recs['Sentiment_scaled']
                )

                user_recs['Hybrid'] = user_recs['Hybrid']
                
                # --- Filter Recommendations by Minimum Rating ---
                if model_choice in user_recs.columns:
                    user_recs = user_recs[user_recs[model_choice] >= min_rating]
                    top_recs = user_recs.sort_values(model_choice, ascending=False).head(5)
                    if len(user_recs[model_choice]) == 0:
                        st.write(f"No recommendations found with {model_choice} scores ≥ {min_rating}. Try lowering the minimum rating.")
                    else:    
                        st.subheader(f"Top 5 Recommendations for User {user_id}")
                        st.markdown(f" **Min: {user_recs[model_choice].min():.5f}, Max: {user_recs[model_choice].max():.5f}**")
                        for _, row in top_recs.iterrows():
                            st.markdown(f"**Product ID: {row['item_id']}**")
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
                        plt.bar(top_recs['item_id'], top_recs[model_choice], color='skyblue')
                        plt.title(f"{model_choice} Scores for Top 5 Products")
                        plt.xlabel("Product ID")
                        plt.ylabel("Score")
                        st.pyplot(plt)

        except Exception as e:
            import traceback
            st.error(f"❌ Error: {e}")
            st.code(traceback.format_exc())
        
if __name__ == "__main__":
    main()
