# --------------------------------------
# Section 1: Imports
# --------------------------------------
# Standard Library
import gzip
import json
import pickle

# Third-Party Libraries
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------
# Section 2: NLTK Setup (Run Once)
# --------------------------------------
# The following downloads are for the NLTK library. 
# They only need to be run once per environment.
# 'stopwords': A list of common words (e.g., "a", "the") to filter out.
# 'wordnet': A lexical database required for lemmatization (reducing words to their base form).
# 'punkt': A pre-trained tokenizer for splitting text into sentences or words.
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     print("Downloading NLTK stopwords...")
#     nltk.download('stopwords')

# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     print("Downloading NLTK wordnet...")
#     nltk.download('wordnet')

# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('tokenizers/punkt_tab')
# except LookupError:
#     print("Downloading NLTK punkt tokenizer...")
#     nltk.download('punkt')
#     nltk.download('punkt_tab')

for resource in ["punkt", "punkt_tab", "wordnet", "stopwords"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# --------------------------------------
# Section 3: Data Loading Functions
# --------------------------------------
def parse_gzipped_json(path):
    """A generator to read gzipped JSON file line by line."""
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)
            
@st.cache_data
def load_reviews_df(path):
    """Loads the entire gzipped JSON file into a pandas DataFrame."""
    print(f"Loading data from {path}...")
    df = pd.DataFrame(list(parse_gzipped_json(path)))
    # Standardize column names for consistency
    df = df.rename(columns={'reviewerID': 'user_id', 'asin': 'item_id', 'reviewText': 'review_text'})
    print("Data loading complete.")
    return df

# --------------------------------------
# Section 4: Core Analysis Functions
# --------------------------------------
# def get_svd_predictions_for_user_history(user_history_df, user_id, model_path):
#     """
#     Loads a pre-trained SVD model to predict ratings for items a user has
#     already interacted with and returns the top 5 predictions.

#     Args:
#         user_history_df (pd.DataFrame): DataFrame with 'user_id' and 'item_id' columns.
#         user_id (str): The ID of the user for whom to generate predictions.
#         model_path (str): The file path to the trained SVD model (.pkl).

#     Returns:
#         pd.DataFrame: A 5-row DataFrame with 'user_id', 'item_id', and 'svd_prediction',
#                       sorted by the predicted rating.
#     """
#     print(f"Loading SVD model from {model_path}...")
#     with open(model_path, 'rb') as file:
#         svd_model = pickle.load(file)

#     # Filter the DataFrame for the specified user's interaction history
#     target_user_df = user_history_df[user_history_df['user_id'] == user_id].copy()

#     print(f"Generating SVD predictions for user: {user_id}...")
#     # Use the model to predict ratings for each item in the user's history
#     # The 'est' attribute of the prediction object holds the estimated rating.
#     target_user_df['svd_rating'] = target_user_df.apply(
#         lambda row: svd_model.predict(uid=row['user_id'], iid=row['item_id']).est,
#         axis=1
#     )
    
#     # Sort results and return the top 5
#     top_predictions_df = target_user_df.sort_values(by='svd_rating', ascending=False).head()
    
#     return top_predictions_df[['user_id', 'item_id', 'svd_rating']]

def get_svd_predictions_for_user_history(user_id, all_items_df, user_history_df, model_path, n=5):
    """
    Recommend Top-N items for a given user using a pre-trained SVD model.

    Args:
        user_id (str or int): The target user ID.
        all_items_df (pd.DataFrame): DataFrame containing all available items.
                                     Must include an 'item_id' column.
        user_history_df (pd.DataFrame): DataFrame containing user interactions.
                                        Must include 'user_id' and 'item_id' columns.
        model_path (str): Path to the pre-trained SVD model (.pkl file).
        n (int): Number of items to recommend (default = 5).

    Returns:
        pd.DataFrame: A DataFrame with columns ['user_id', 'item_id', 'svd_rating']
                      representing the top-N recommendations.
    """
    # 1 Load the trained SVD model
    with open(model_path, 'rb') as f:
        svd_model = pickle.load(f)

    # 2️ Get items the user has already interacted with
    user_items = set(user_history_df[user_history_df['user_id'] == user_id]['item_id'])

    # 3️ Determine candidate items (all items minus items already interacted with)
    candidate_items = set(all_items_df['item_id']) - user_items

    if not candidate_items:
        return pd.DataFrame(columns=['user_id', 'item_id', 'svd_rating'])

    # 4️ Predict ratings for each candidate item
    predictions = []
    for item in candidate_items:
        est_rating = svd_model.predict(uid=user_id, iid=item).est
        predictions.append((user_id, item, est_rating))

    # 5️ Convert predictions to DataFrame and select Top-N highest scores
    pred_df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'svd_rating'])
    top_n_df = pred_df.sort_values(by='svd_rating', ascending=False).head(n)

    return top_n_df

def calculate_sentiment_for_items(reviews_df, recommended_items_df):
    """
    Calculates the average sentiment score for a list of recommended items.

    Args:
        reviews_df (pd.DataFrame): The full reviews DataFrame, containing 'item_id' and 'review_text'.
        recommended_items_df (pd.DataFrame): DataFrame of recommended items, containing an 'item_id' column.

    Returns:
        pd.DataFrame: The recommended_items_df with an added 'avg_sentiment_score' column.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    # This is an inner helper function, its scope is limited to calculate_sentiment_for_items.
    def get_sentiment_score(text):
        # --- Annotation Start ---
        # The VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool
        # is specifically attuned to sentiments expressed in social media and reviews.
        # It returns a dictionary with 'neg', 'neu', 'pos', and 'compound' scores.
        
        # We are interested in the 'compound' score, which is a normalized, weighted composite score.
        # It ranges from -1 (most extreme negative) to +1 (most extreme positive).
        # A score around 0 indicates neutrality.
        # --- Annotation End ---
        if isinstance(text, str):
            return analyzer.polarity_scores(text)['compound']
        return 0 # Return neutral score for non-string (e.g., NaN) reviews

    # Get the list of unique item IDs from the recommendations
    item_ids_to_analyze = recommended_items_df['item_id'].tolist()
    sentiment_scores = []
    
    print("Calculating sentiment scores for recommended items...")
    for item_id in item_ids_to_analyze:
        # Filter all reviews for the current item
        item_reviews = reviews_df[reviews_df['item_id'] == item_id]
        
        # Calculate sentiment score for each review and get the average
        # We filter out cases where review_text might be missing (NaN)
        if not item_reviews.empty and item_reviews['review_text'].notna().any():
            scores = item_reviews['review_text'].apply(get_sentiment_score)
            average_score = scores.mean()
        else:
            average_score = 0 # Assign a neutral score if no reviews are found
            
        sentiment_scores.append(average_score)
    
    # Add the calculated scores as a new column to the recommendations DataFrame
    output_df = recommended_items_df.copy()
    output_df['sentiment_score'] = sentiment_scores
    
    return output_df

def calculate_bert_content_similarity(
    reviews_df: pd.DataFrame,
    recommended_items_df: pd.DataFrame,
    user_id: str,
    bert_vectors: np.ndarray,
    bert_item_id_to_idx: dict
) -> pd.DataFrame:
    """
    Calculates the average BERT content similarity between recommended items
    and the items a user has previously interacted with. This helps to
    ensure recommended items are similar in content to what the user likes.

    Args:
        reviews_df (pd.DataFrame): The full reviews DataFrame, containing 'user_id' and 'item_id'.
                                   Used to get the user's historical items.
        recommended_items_df (pd.DataFrame): DataFrame of recommended items (e.g., from SVD),
                                            containing an 'item_id' column.
        user_id (str): The ID of the user for whom to calculate similarity.
        bert_vectors (np.ndarray): A NumPy array where each row is the BERT embedding
                                   for an item's description (e.g., product description).
        bert_item_id_to_idx (dict): A dictionary mapping 'item_id' to its
                                    corresponding index in `bert_vectors`.

    Returns:
        pd.DataFrame: The recommended_items_df with an added 'avg_bert_similarity' column.
                      Items not found in the BERT embeddings will have a similarity of 0.
    """
    if bert_vectors is None or not bert_item_id_to_idx:
        print("BERT vectors or item ID mapping are not loaded. Skipping BERT similarity calculation.")
        # Return the recommended_items_df with a column of zeros or NaNs for avg_bert_similarity
        output_df = recommended_items_df.copy()
        output_df['bert_similarity'] = 0.0
        return output_df

    # Get the user's historical items
    user_historical_items_df = reviews_df[reviews_df['user_id'] == user_id]
    if user_historical_items_df.empty:
        print(f"No historical items found for user: {user_id}. BERT similarity will be 0.")
        output_df = recommended_items_df.copy()
        output_df['bert_similarity'] = 0.0
        return output_df

    # Get BERT vectors for the user's historical items
    user_historical_vectors = []
    historical_item_ids_found = []
    for item_id in user_historical_items_df['item_id'].unique():
        if item_id in bert_item_id_to_idx:
            user_historical_vectors.append(bert_vectors[bert_item_id_to_idx[item_id]])
            historical_item_ids_found.append(item_id)
    
    if not user_historical_vectors:
        print(f"No BERT embeddings found for historical items of user: {user_id}. BERT similarity will be 0.")
        output_df = recommended_items_df.copy()
        output_df['bert_similarity'] = 0.0
        return output_df

    user_historical_vectors = np.array(user_historical_vectors)

    # Calculate similarity for each recommended item
    bert_similarity_scores = []
    print("Calculating BERT content similarity for recommended items...")
    for item_id in recommended_items_df['item_id'].tolist():
        if item_id in bert_item_id_to_idx:
            recommended_item_vector = bert_vectors[bert_item_id_to_idx[item_id]].reshape(1, -1)
            
            # Calculate cosine similarity between the recommended item and all historical items
            # The result is an array of similarities, one for each historical item
            similarities = cosine_similarity(recommended_item_vector, user_historical_vectors)[0]
            avg_similarity = np.mean(similarities)
            bert_similarity_scores.append(avg_similarity)
        else:
            # If a recommended item's BERT vector is not found, assign a neutral or zero similarity
            bert_similarity_scores.append(0.0)
            print(f"Warning: BERT embedding not found for recommended item: {item_id}. Assigning 0.0 similarity.")

    output_df = recommended_items_df.copy()
    output_df['bert_similarity'] = bert_similarity_scores

    return output_df

def calculate_xgboost(df, model):
    """
    Given a dataframe of user-product features and a trained XGBoost model,
    this function adds predicted probabilities (likelihood that the user will like the product)
    to the dataframe as a new column: 'xgb_pred_score'.

    Parameters:
    -----------
    df : pd.DataFrame
        A dataframe containing the following columns:
        - user_id
        - asin
        - svd_rating
        - sentiment_score
        - bert_similarity
        - user_ave_rating
        - product_ave_rating
        - target_overall (label for training, dropped for prediction)

    model : xgboost.Booster or XGBClassifier
        A trained XGBoost model loaded from a pickle file.

    Returns:
    --------
    df : pd.DataFrame
        The input dataframe with 'target_overall' dropped and a new column 'xgb_pred_score'
        containing predicted probabilities that the user will like the product.
    """

    # Drop non-feature columns before prediction: 'user_id', 'asin', 'target_overall'
    X = df.drop(columns=['user_id', 'item_id'])

    # Predict the probability that each user will like each product (label = 1)
    # predict_proba returns [prob_0, prob_1]; we want prob_1 (i.e., "like")
    df['XGBoost'] = model.predict_proba(X)[:, 1]

    # Drop the target label column, since it's no longer needed
    return df

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # output predicted rating
        )

    def forward(self, user_idx, item_idx):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.fc_layers(x).squeeze()
    
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_idx'].values, dtype = torch.long)
        self.items = torch.tensor(df['item_idx'].values, dtype=torch.long)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model architecture.
    It learns embeddings for users and items and combines them through a
    multi-layer perceptron (MLP) to predict ratings.
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single predicted rating
        )

    def forward(self, user_idx, item_idx):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        # Concatenate user and item embeddings to feed into the MLP
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.fc_layers(x).squeeze()

def get_ncf_predictions(
    target_df: pd.DataFrame, 
    model_path: str, 
    mapping_path: str, 
    embedding_dim: int = 64
) -> pd.DataFrame:
    """
    Loads a pre-trained NCF model and generates rating predictions for user-item pairs.

    Args:
        target_df (pd.DataFrame): DataFrame containing 'user_id' and 'item_id' columns 
                                  for which predictions are needed.
        model_path (str): The file path to the saved PyTorch model state dictionary (.pth).
        mapping_path (str): The file path to the pickled user/item to index mappings (.pkl).
        embedding_dim (int): The dimensionality of the user and item embeddings. Must match
                             the trained model's embedding dimension.

    Returns:
        pd.DataFrame: A DataFrame containing predictions for the valid user-item pairs
                      found in the model, with a new 'ncf_prediction' column.
    """
    
    # --- Annotation Start ---
    # Neural Collaborative Filtering (NCF) is a deep learning model for recommendations.
    # It learns latent feature vectors (embeddings) for users and items from their
    # past interactions. Unlike traditional matrix factorization (like SVD), NCF uses a
    # neural network to learn a potentially more complex and non-linear relationship
    # between user and item embeddings to predict ratings.
    # --- Annotation End ---

    def _load_model_and_mappings(model_path, mapping_path, embedding_dim):
        """Inner function to load the model and mappings from disk."""
        print(f"Loading NCF model from {model_path}...")
        # Load the user and item ID to integer index mappings.
        # These are essential because embedding layers in PyTorch work with integer indices.
        with open(mapping_path, 'rb') as f:
            user2idx, item2idx = pickle.load(f)

        # Initialize the model architecture
        model = NCF(num_users=len(user2idx), num_items=len(item2idx), embedding_dim=embedding_dim)
        
        # Load the pre-trained weights into the model
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        
        # Set the model to evaluation mode. This is crucial as it disables layers
        # like Dropout or BatchNorm, ensuring deterministic output for predictions.
        model.eval()
        
        return model, user2idx, item2idx

    def _predict_ratings(model, user2idx, item2idx, df_to_predict):
        """Inner function to perform the prediction on a DataFrame."""
        # The model only knows about users and items it was trained on.
        # Filter the input DataFrame to include only these known entities.
        valid_rows = df_to_predict[df_to_predict['user_id'].isin(user2idx) & df_to_predict['item_id'].isin(item2idx)].copy()
        
        if valid_rows.empty:
            print("Warning: No valid user-item pairs found in the target DataFrame for NCF prediction.")
            return pd.DataFrame(columns=list(df_to_predict.columns) + ['NCF'])

        # Convert string IDs to the integer indices the model expects
        users_tensor = torch.tensor([user2idx[u] for u in valid_rows['user_id']], dtype=torch.long)
        items_tensor = torch.tensor([item2idx[i] for i in valid_rows['item_id']], dtype=torch.long)

        # Use torch.no_grad() to disable gradient calculations, which is unnecessary
        # for inference and saves memory and computation.
        print("Generating NCF predictions...")
        with torch.no_grad():
            predictions = model(users_tensor, items_tensor).numpy()

        valid_rows['NCF'] = predictions
        return valid_rows

    # Main execution flow of get_ncf_predictions
    model, user_to_idx, item_to_idx = _load_model_and_mappings(model_path, mapping_path, embedding_dim)
    predictions_df = _predict_ratings(model, user_to_idx, item_to_idx, target_df)
    
    return predictions_df