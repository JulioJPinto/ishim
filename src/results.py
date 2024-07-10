import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Define file paths
csv_file_path = 'sorted_orcamento.csv'
embeddings_file_path = 'embeddings_orcamento.pkl'
fine_tuned_model_path = 'fine-tuned-model'

# Load the fine-tuned model if available, otherwise load the pre-trained model
if os.path.exists(fine_tuned_model_path):
    model = SentenceTransformer(fine_tuned_model_path)
else:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Function to check if the embeddings need to be computed
def embeddings_need_update(csv_file, embeddings_file):
    if not os.path.exists(embeddings_file):
        return True
    return os.path.getmtime(csv_file) > os.path.getmtime(embeddings_file)

# Function to compute and save embeddings
def compute_and_save_embeddings(csv_file, embeddings_file):
    # Load phrases from CSV
    df = pd.read_csv(csv_file)
    phrases = df['design'].tolist()
    
    # Encode the phrases
    phrase_embeddings = model.encode(phrases)
    
    # Save the embeddings and phrases
    with open(embeddings_file, 'wb') as f:
        pickle.dump((phrases, phrase_embeddings), f)
    
    return phrases, phrase_embeddings

# Load or compute embeddings as needed
if embeddings_need_update(csv_file_path, embeddings_file_path):
    phrases, phrase_embeddings = compute_and_save_embeddings(csv_file_path, embeddings_file_path)
else:
    with open(embeddings_file_path, 'rb') as f:
        phrases, phrase_embeddings = pickle.load(f)

# Function to find top N most similar phrases to the input phrase
def find_top_similar_phrases(input_phrase, phrases, phrase_embeddings, top_n=5):
    # Encode the input phrase
    input_embedding = model.encode([input_phrase])
    
    # Calculate cosine similarity between the input phrase and each phrase in the list
    similarities = cosine_similarity(input_embedding, phrase_embeddings)[0]
    
    # Get the indices of the top N most similar phrases
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Get the top N most similar phrases and their similarity scores
    top_phrases = [(phrases[i], similarities[i]) for i in top_indices]
    
    return top_phrases

# Example user input
input_phrase = input("> Enter an article: ")

# Find top 5 most similar phrases
top_similar_phrases = find_top_similar_phrases(input_phrase, phrases, phrase_embeddings)

# Print the top 5 most similar phrases
print("Top 5 most similar phrases:")
for phrase, score in top_similar_phrases:
    print(f"Phrase: {phrase}, Similarity Score: {score:.4f}")

# Save top similar phrases for feedback
with open('top_similar_phrases.pkl', 'wb') as f:
    pickle.dump((input_phrase, top_similar_phrases), f)
