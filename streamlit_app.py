
import streamlit as st
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import os

# Assuming processed_bm25, processed_corpus, agg_df, and destinations are already loaded

# Define the path to the app_models folder
models_path = 'app_models/'

# Load BM25 model
with open(os.path.join(models_path, "processed_bm25.pkl"), "rb") as f:
    processed_bm25 = pickle.load(f)

# Load TF-IDF vectorizer and matrix
with open(os.path.join(models_path, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(models_path, "tfidf_matrix.pkl"), "rb") as f:
    tfidf_matrix = pickle.load(f)

# Load BERT model
with open(os.path.join(models_path, "bert_model.pkl"), "rb") as f:
    bert_model = pickle.load(f)

# Load embeddings
with open(os.path.join(models_path, "embeddings.pkl"), "rb") as f:
    embeddings = pickle.load(f)

# Load processed_corpus and agg_df.destination (if required)
with open(os.path.join(models_path, "processed_corpus.pkl"), "rb") as f:
    processed_corpus = pickle.load(f)

with open(os.path.join(models_path, "agg_df_destination.pkl"), "rb") as f:
    agg_df_destination = pickle.load(f)

# Load the train emissions data
with open(os.path.join(models_path, "agg_df_train_emissions.pkl"), "rb") as f:
    agg_df_train_emissions = pickle.load(f)

# Preprocessing function for user input
def preprocess_input(user_input):
    return user_input.split(" ")

# BM25 model
def predict_bm25(user_input, bm25_model, corpus, destinations):
    tokenized_query = preprocess_input(user_input)
    doc_scores = bm25_model.get_scores(tokenized_query)
    top_3_idx = np.argsort(doc_scores)[-3:][::-1]
    
    top_3_results = []
    for idx in top_3_idx:
        best_match_document = corpus[idx]
        best_match_destination = destinations[idx]
        top_3_results.append((best_match_destination, best_match_document, doc_scores[idx]))
    
    return top_3_results

# TF-IDF model
def predict_tfidf(user_input, tfidf_matrix, tfidf, corpus, destinations):
    query_vector = tfidf.transform([user_input])
    doc_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_3_idx = np.argsort(doc_scores)[-3:][::-1]
    
    top_3_results = []
    for idx in top_3_idx:
        best_match_document = corpus[idx]
        best_match_destination = destinations[idx]
        top_3_results.append((best_match_destination, best_match_document, doc_scores[idx]))
    
    return top_3_results

# BERT model
def predict_bert(user_input, model, embeddings, corpus, destinations):
    query_embedding = model.encode([user_input])
    doc_scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_3_idx = np.argsort(doc_scores)[-3:][::-1]
    
    top_3_results = []
    for idx in top_3_idx:
        best_match_document = corpus[idx]
        best_match_destination = destinations[idx]
        top_3_results.append((best_match_destination, best_match_document, doc_scores[idx]))
    
    return top_3_results

# Ensemble model
def ensemble_predict(user_input, bm25_model, tfidf_matrix, tfidf, model, embeddings, corpus, destinations):
    # Get predictions from each model
    bm25_results = predict_bm25(user_input, bm25_model, corpus, destinations)
    tfidf_results = predict_tfidf(user_input, tfidf_matrix, tfidf, corpus, destinations)
    bert_results = predict_bert(user_input, model, embeddings, corpus, destinations)
    
    # Combine the results from all models
    combined_results = {}
    for result in bm25_results + tfidf_results + bert_results:
        destination, document, score = result
        if destination not in combined_results:
            combined_results[destination] = []
        combined_results[destination].append(score)
    
    # Calculate the average score for each destination
    averaged_results = []
    for destination, scores in combined_results.items():
        avg_score = np.mean(scores)
        averaged_results.append((destination, avg_score))
    
    # Sort by average score and return top 3
    averaged_results.sort(key=lambda x: x[1], reverse=True)
    top_3_ensemble = averaged_results[:3]
    
    return top_3_ensemble

# Recommendation with Best F1 and Lowest Emissions Balance
def recommend_best_destination_with_balance(top_3_ensemble_with_emissions):
    # Directly calculate F1 score - emissions for each recommendation
    recommendations = []
    for destination, f1_score, emissions in top_3_ensemble_with_emissions:
        combined_score = f1_score - emissions  # F1 score - emissions (no scaling/normalization)
        recommendations.append((destination, combined_score))

    # Get the destination with the highest combined score
    best_destination = max(recommendations, key=lambda x: x[1])[0]
    return best_destination

# Streamlit app code
def run_streamlit_app_with_best_recommendation():
    st.title("Travel Review Prediction with Ensemble Models")
    
    # Input field for user query
    user_input = st.text_input("Enter your travel review query:")
    
    if user_input:
        st.write("User input received:", user_input)  # Debugging line

        # Get the ensemble prediction with emissions
        top_3_ensemble_with_emissions = ensemble_predict(user_input, processed_bm25, tfidf_matrix, tfidf, bert_model, embeddings, processed_corpus, agg_df_destination)

        # Add emissions to the results
        top_3_ensemble_with_emissions_and_emissions = []
        for (destination, score) in top_3_ensemble_with_emissions:
            # Instead of using .index(), use .loc to fetch the emission value
            emissions = agg_df_train_emissions.loc[agg_df_destination == destination].values[0]
            top_3_ensemble_with_emissions_and_emissions.append((destination, score, emissions))

        # Display top 3 predictions with F1 score and emissions
        st.write("### Top 3 Predictions:")
        for destination, score, emissions in top_3_ensemble_with_emissions_and_emissions:
            st.write(f"**Destination:** {destination}")
            st.write(f"**F1 Score:** {score:.2f}")
            st.write(f"**Emissions:** {emissions:.2f} kg CO2")
            st.write("---")
        
        # Get the best destination based on F1 score and emissions balance
        best_destination = recommend_best_destination_with_balance(top_3_ensemble_with_emissions_and_emissions)

        st.write("### Best Recommendation based on F1 Score - Emissions:")
        st.write(f"**Best Destination:** {best_destination}")
        

run_streamlit_app_with_best_recommendation()


