"""
rank_comparables.py: Use LLM technique to screen potential comparable companies to get
                     comparable companies for Transfer Pricing audit reports

Detailed Description: Use SentenceTransformer for text embeddings, compare and calculate similarity of companies

Author: Maoyi Fan
Date: 2026/3/3
Version: 0.1
License: None
Contact: maoyi.fan@yapro.com.tw
Dependencies: <List of external dependencies if any>
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------- CONFIG ----------------
EXCEL_FILE = "Potential_Comparables.xlsx"
TESTED_PARTY_FILE = "Liaiseng.txt"
TEXT_COLUMN = "Description"
MODEL_NAME = "all-mpnet-base-v2"                # strong general-purpose model
# ---------------------------------------

# -------- PENALTIES RULES DICTIONARY ---------
PENALTIES = {
    "IP_HEAVY": 0.10,
    "MANUFACTURING": 0.15,
    "REAL_ESTATE": 0.4
}


def rank_comparable():
    # Load data
    df = pd.read_excel(EXCEL_FILE)
    # Basic validation to ensure there is required information in the list of potential comparables
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in Excel file")
    df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)

    # Load tested party description
    with open(TESTED_PARTY_FILE, "r", encoding="utf-8") as f:
        tested_party_text = f.read().strip()

    # Initialize embedding model (feature extractor)
    model = SentenceTransformer(MODEL_NAME)

    # Create embeddings
    print("Embedding tested party...")
    tested_embedding = model.encode([tested_party_text], normalize_embeddings=True)

    print("Embedding comparable companies...")
    company_embeddings = model.encode(
        df[TEXT_COLUMN].tolist(),
        normalize_embeddings=True
    )

    # Base process: Compute cosine similarity
    similarities = cosine_similarity(tested_embedding, company_embeddings)[0]
    # Add scores to DataFrame
    df["similarity_score"] = similarities
    # Sort the DataFrame by similarity scores
    df_ranked = df.sort_values(by="similarity_score", ascending=False)

    # Add company category tags by walking through TEXT_COLUMN series and applying the tag function 'tag_company'
    df_ranked["flags"] = df_ranked[TEXT_COLUMN].apply(tag_company)

    # Add shared terms by rows according to company descriptions
    df_ranked["shared_terms"] = df_ranked[TEXT_COLUMN].apply(
        lambda x: ", ".join(extract_shared_terms(tested_party_text, x))
    )

    # Add adjusted scores column by applying penalty function
    df_ranked["adjusted_score"] = df_ranked.apply(apply_penalty, axis=1)

    # Sort the DataFrame by 'adjusted score'
    df_ranked = df_ranked.sort_values(by="adjusted_score", ascending=False)

    # Save output with rankings, company tags,...
    df_ranked.to_excel("ranked_comparables.xlsx", index=False)

#
# Description: assigning category tags according to business description of the companies
#
def tag_company(text):
    text = text.lower()
    tags = []
    if any(k in text for k in ["patent", "proprietary", "license", "royalty"]):
        tags.append("IP_HEAVY")
    if any(k in text for k in ["manufacture", "factory", "production"]):
        tags.append("MANUFACTURING")
    if any(k in text for k in ["bank", "insurance", "broker"]):
        tags.append("FINANCIAL")
    if any(k in text for k in ["real estate", "property management", "property investment", "asset management"]):
        tags.append("REAL_ESTATE")
    if any(k in text for k in ["platform", "saas", "subscription software"]):
        tags.append("SOFTWARE_PRODUCT")

    return tags
    # return ", ".join(tags)

#
# Description: extract shared key phrases using TF-IDF(Term Frequency-Inverse Document Frequency) algorithms
#
def extract_shared_terms(text_a, text_b, top_n=8):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=2000
    )

    tfidf = vectorizer.fit_transform([text_a, text_b])
    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf.toarray()
    shared_scores = scores[0] * scores[1]

    top_indices = shared_scores.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices if shared_scores[i] > 0]

#
# Description: applies penalties to companies with specific business activity tags
#
def apply_penalty(row):
    score = row["similarity_score"]
    for flag in row["flags"]:
        # print(f"applying penalty for {flag}....before: {score}", end=" ")
        score -= PENALTIES.get(flag, 0)
        # print(f"after: {score}")
    return max(score, 0)


if __name__ == "__main__":
    rank_comparable()
