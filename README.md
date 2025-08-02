---
title: Hybrid Ensemble Recommender System
emoji: 📚
colorFrom: pink
colorTo: green
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
license: mit
short_description: A hybrid recommender system using SVD, BERT, sentiment analysis, XGBoost, and NCF.
---

# 📚 Hybrid Ensemble Recommender System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ZPENG127/Hybrid-Ensemble-Recommender-System)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated **hybrid product recommendation system** that ensembles multiple state-of-the-art models to provide accurate and robust user recommendations. The entire system is deployed as an interactive **Streamlit** web application.

---

## 🚀 Live Demo & Repository

| Link                                                                                            | Description                                         |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| 🔴 **[Live Demo (Hugging Face Spaces)]**                                                        | Interact with the deployed Streamlit application.   |
| 🟡 **[GitHub Repository](https://github.com/Amos-Peng-127/Hybrid-Ensemble-Recommender-System)** | Browse the source code, notebooks, and setup files. |

---

## ✨ Core Features

- ✅ **Ensemble Methodology**: Combines predictions from multiple models to improve accuracy and handle data sparsity.
- ✅ **Collaborative Filtering**: Utilizes **SVD** (`Singular Value Decomposition`) to capture latent user-item interactions.
- ✅ **Content-Based Filtering**: Employs **BERT** embeddings to understand product semantics and recommend similar items.
- ✅ **Sentiment-Enhanced Recommendations**: Integrates sentiment analysis scores as a feature to filter out negatively reviewed products.
- ✅ **Advanced Ranking Model**: Uses an **XGBoost** model to rank candidate recommendations based on a rich set of features.
- ✅ **Deep Learning Approach**: Implements **Neural Collaborative Filtering (NCF)** for a deep learning perspective on recommendations.
- ✅ **Interactive UI**: A user-friendly interface built with **Streamlit** allowing for easy interaction and recommendation filtering.

---

## 🧠 Methodology Overview

The recommendation pipeline follows a multi-stage process to generate the final ranked list for a user:

1.  **Candidate Generation**: A set of initial product candidates is generated using both **SVD** (for collaborative signals) and **BERT-based content similarity** (for item semantics).
2.  **Feature Engineering**: For each user-candidate pair, a feature vector is constructed. This includes:
    - The prediction scores from SVD and BERT.
    - Sentiment analysis scores for the product.
    - Other user and item metadata.
3.  **Ranking**: The **XGBoost** model takes these feature vectors as input and predicts a final relevance score for each candidate product.
4.  **Final Output**: The products are sorted based on their XGBoost scores, and the top-N recommendations are presented to the user. The **NCF** model serves as an alternative deep-learning-based pipeline for comparison and ensembling.

---

## 🛠 Tech Stack

- **Frontend & UI**:
  - `Streamlit`
- **Machine Learning & Deep Learning**:
  - `PyTorch`
  - `XGBoost`
  - `scikit-learn`
  - `Transformers`
  - `scikit-surprise`
- **Data Handling & Utilities**:
  - `Pandas`, `NumPy`
  - `gdown`
- **Visualization**:
  - `Matplotlib`

---

## 📂 Project Structure

```
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── notebooks/helper.py  # Helper functions for recommendation logic
├── resources/           # Model files and embeddings (auto-downloaded)
└── README.md            # Project documentation
```

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://[your-github-repo-url]
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Note: On the first run, the application will automatically download the necessary models and data files from Google Drive._

---

## 🚀 How to Run

1.  **Launch the Streamlit app from your terminal:**

    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser** and navigate to `http://localhost:8501`.

---

## 📦 Model Files

The app **automatically downloads required files** (CSV, embeddings, and models) from Google Drive on the first launch.

---

## 👥 Authors

- **Zhixiang Peng** ([@Amos-Peng-127](https://github.com/Amos-Peng-127))
- **Rain Lin** ([@TINYRAINYLIN](https://github.com/TINYRAINYLIN))

---
