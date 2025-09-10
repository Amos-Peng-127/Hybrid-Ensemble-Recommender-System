import streamlit as st # type: ignore
from streamlit_extras.badges import badge # type: ignore
st.set_page_config(page_title="Amazon Recommender", page_icon="📚")

def main():
    # st.title("My Streamlit App")
    
    st.markdown(
        """
        <style>
            * {
                font-family: cursive;
                color: #000000;
                font-size: 16px;
            }
            .stApp {
                background-color: #ADD8E6; /* Light blue background */
            }
            .stSidebar {
                background-color: white; /* Lighter blue for sidebar */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
   
    st.markdown(
        """
        <div style="display: flex; gap: 10px; align-items: center;">
            <a href="https://github.com/TINYRAINYLIN" target="_blank" style="text-decoration: none;">
                <span style="background-color: #E4CFD3; color: black; padding: 5px 10px; border-radius: 5px; font-size: 14px; display: inline-block;">
                💧 Rain Lin
                </span>
            </a>
            <a href="https://github.com/Amos-Peng-127" target="_blank" style="text-decoration: none;">
                <span style="background-color: #007bff; color: white; padding: 5px 10px; border-radius: 5px; font-size: 14px; display: inline-block;">
                🐉 Zhixiang Peng
                </span>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    import os, sys, pickle, warnings
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import gdown # type: ignore
    import ast

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

    st.markdown("<h1 style='font-size: 40px;'>📚 Amazon Product Recommender</h1>", unsafe_allow_html=True)
    st.markdown("Choose a user ID to get recommendations for this user ID. You can filter recommendations by minimum rating and category.")

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
        "Metadata.csv": "https://drive.google.com/uc?id=1Gghd2ZkyU2HZXzOMZjpvkiieHMaFwI-q"
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
            # msg = st.empty()
            # msg.success(f"Downloaded files: {', '.join(downloaded_files)}")

            # time.sleep(3)  # Display for 3 seconds
            # msg.empty()
            pass
        else:
            # msg = st.empty()
            # msg.success("✅ All files already exist. No download needed.")

            # time.sleep(3)  # Display for 3 seconds
            # msg.empty()
            pass

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
    metadata_df = pd.read_csv(os.path.join(download_dir, "Metadata.csv"), low_memory=False)

    with open(os.path.join(download_dir, "xgboost_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)

    metadata_df = metadata_df.rename(columns={"asin": "item_id"}) if metadata_df is not None else None

    if 'price' in metadata_df.columns: # type: ignore
        # Remove currency symbols, then convert to a numeric type.
        # 'coerce' will turn any errors (like text) into NaN (Not a Number).
        metadata_df['price'] = metadata_df['price'].astype(str).str.replace(r'[$,]', '', regex=True) # type: ignore
        metadata_df['price'] = pd.to_numeric(metadata_df['price'], errors='coerce') # type: ignore

    # ---------------------- #
    # 4. UI Setup
    # ---------------------- #
    st.sidebar.header("🔎 Filter Options")
    min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.5)
    category_options = df["category"].dropna().unique() if "category" in df.columns else []
    selected_category = (
        st.sidebar.selectbox("Category", ["All"] + sorted(category_options.tolist())) # type: ignore
        if len(category_options) > 0
        else "All"
    )

    # user_id = st.text_input("Enter User ID: (For example, 'AAP7PPBU72QFM')")
    user_id = st.selectbox(
        "Choose from 10 Random User ID",
        df.sample(10)["user_id"].unique().tolist(),
    )

    model_choice = st.selectbox("Choose Model", ["Hybrid", "SVD", "BERT", "Sentiment", "XGBoost", "NCF"])

    with st.expander("Calculation Strategy Explanation"):
                st.write('''
                The hybrid recommendation system combines multiple models to generate a comprehensive score for each product. Here's how each component contributes:

                - **SVD**: The SVD model provides a basic recommendation based on user-item interactions and their ratings.
                - **BERT**: The BERT model captures semantic similarity between products based on their text content.
                - **Sentiment**: Sentiment analysis is used to filter out negatively reviewed products.
                - **XGBoost**: The XGBoost model is used to fine-tune the final recommendation score.
                - **NCF**: The Neural Collaborative Filtering (NCF) model is a deep learning approach to generate personalized recommendations.
                ''')
            
                st.latex(r'''
                    Hybrid = 0.3 \times XGBoost_{scaled} + 0.25 \times NCF_{scaled} + 0.15 \times BERT_{scaled} + 0.15 \times SVD_{scaled} + 0.15 \times Sentiment_{scaled}
                    ''')
            
    # ---------------------- #
    # 5. Generate Recommendations
    # ---------------------- #
    if st.button("Get Recommendations"):
        placeholder = st.empty()
        try:
            
            # Display meme image in placeholder
            with placeholder:
                st.image(
                    "https://media.licdn.com/dms/image/v2/C5612AQFtBw-tXa5Saw/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1554742888227?e=1760572800&v=beta&t=7UtXE9QnqhmDURtPJ3zYAfC62op2HLyuBVaP7SDD0yc",
                    width=300
                )  # A relevant, fun meme
            with st.spinner('🤖 Calculating your personalized recommendations...'):

                # The results will replace the spinner once the code inside is finished.
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

                if metadata_df is not None:
                    user_recs = pd.merge(user_recs, metadata_df, on="item_id", how="left")
                
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
                    
                    placeholder.empty()

                    # --- Filter Recommendations by Minimum Rating ---
                    if model_choice in user_recs.columns:
                        user_recs = user_recs[user_recs[model_choice] >= min_rating]
                        top_recs = user_recs.sort_values(model_choice, ascending=False).head(5)
                        top_recs.drop_duplicates(keep='first', inplace=True)
                        
                        if len(user_recs[model_choice]) == 0:
                            st.write(f"No recommendations found with {model_choice} scores ≥ {min_rating}. Try lowering the minimum rating.")
                        else:    
                            st.subheader(f"Top {len(top_recs)} Recommendations for User ID:{user_id}")
                            st.markdown(f" **Min: {user_recs[model_choice].min():.5f}, Max: {user_recs[model_choice].max():.5f}**")
                            for _, row in top_recs.iterrows():
                                # --- Get all the metadata with corrected column names ---
                                product_title = row.get('title', f"Product ID: {row['item_id']}")
                                image_urls = row.get('imageURLHighRes') # Use the correct image column
                                price = row.get('price')
                                category = row.get('main_cat', 'No Category') # Use the correct category column
                                
                                url_lists = []
                                if image_urls and isinstance(image_urls, str) and image_urls.startswith('['):
                                    try:
                                        # Convert the string "['url1']" into a real list ['url1']
                                        url_lists = ast.literal_eval(image_urls)
                                        # Get the first URL from the list
                                        image_urls = url_lists[0] if url_lists else None
                                    except:
                                        # If parsing fails, treat as no image
                                        image_urls = None
                                        url_lists = []
                                
                                # --- Display the metadata ---
                                st.markdown(f"##### {product_title}")

                                img_rows = (len(url_lists) + 4) // 5
                                for r in range(img_rows):
                                    # Create 5 columns
                                    cols = st.columns(5)

                                    # Get image indices for current row
                                    start = r * 5
                                    end = min(start + 5, len(url_lists))  # Ensure end does not exceed length of url_lists
                                    for i, img in enumerate(url_lists[start:end]):
                                        with cols[i]:
                                            st.image(img, width=100)
                                else:
                                    st.text("No image available")
                                    
                                if price and pd.notna(price):
                                    st.markdown(f"**Price:** ${price:.2f} | **Category:** {category}")
                                else:
                                    st.markdown(f"**Price:** No price available | **Category:** {category}")
                                
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
            placeholder.empty()
            import traceback
            st.error(f"❌ Error: {e}")
            st.code(traceback.format_exc())
        
if __name__ == "__main__":
    main()
