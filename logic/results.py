import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Define file paths
csv_file_path = 'data/sorted_orcamento.csv'
embeddings_file_path = 'data/embeddings_orcamento.pkl'
csv_feedback_file_path = 'data/feedback.csv'
fine_tuned_model_path = 'data/fine-tuned-model'

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

def process_phrase(phrase):
    # Tokenize by whitespace
    tokens = phrase.split()
    # Sort tokens alphabetically
    tokens.sort()
    # Join tokens back into a single string
    return ' '.join(tokens)

# Function to compute and save embeddings
def compute_and_save_embeddings(csv_file, embeddings_file):
    # Load phrases from CSV
    df = pd.read_csv(csv_file)
    phrases = df['design'].tolist()
    
    phrases = map(process_phrase, phrases)
    
    # Encode the phrases
    phrase_embeddings = model.encode(phrases)
    
    # Save the embeddings and phrases
    with open(embeddings_file, 'wb') as f:
        pickle.dump((phrases, phrase_embeddings), f)
    
    return phrases, phrase_embeddings

# Load or compute embeddings as needed
if embeddings_need_update(csv_file_path, embeddings_file_path):
    phrases, phrase_embeddings = compute_and_save_embeddings(csv_file_path, embeddings_file_path)
    print("Entered")
else:
    with open(embeddings_file_path, 'rb') as f:
        phrases, phrase_embeddings = pickle.load(f)

# Function to find top N most similar phrases to the input phrase

def find_top_similar_phrases(input_phrase, phrases, phrase_embeddings, csv_path, top_n=5):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    # Filter the CSV for the input phrase
    input_df = df[df['Input Phrase'] == input_phrase]

    # Aggregate ratings and count for each Suggested Phrase
    grouped_df = input_df.groupby('Suggested Phrase').agg({'Rating': 'sum', 'Count': 'sum'}).reset_index()

    # Calculate average rating per Suggested Phrase
    grouped_df['Average Rating'] = grouped_df['Rating'] / (grouped_df['Count']*5)

    # Create a dictionary to map each suggested phrase to its average rating
    rating_map = dict(zip(grouped_df['Suggested Phrase'], grouped_df['Average Rating']))

    # Encode the input phrase (assuming model.encode function is defined elsewhere)
    input_embedding = model.encode([input_phrase])

    # Calculate cosine similarity between the input phrase and each phrase in the list
    similarities = cosine_similarity(input_embedding, phrase_embeddings)[0]

    # Adjust similarities by multiplying with the rating
    adjusted_similarities = [
        similarities[i] * rating_map.get(phrases[i], 1) for i in range(len(phrases))
    ]
    
    # Get the indices of the top N most similar phrases
    top_indices = sorted(range(len(adjusted_similarities)), key=lambda i: adjusted_similarities[i], reverse=True)[:top_n]
    
    # Get the top N most similar phrases and their adjusted similarity scores
    top_phrases = [(phrases[i], adjusted_similarities[i]) for i in top_indices]
    
    return top_phrases

#binary search for phrase
def find_phrase(sorted_list, phrase):
    index = bisect.bisect_left(sorted_list, phrase)
    if index != len(sorted_list) and sorted_list[index] == phrase:
        return index
    return -1

# Example user input
input_phrase = input("> Enter an article: ")
input_phrase = process_phrase(input_phrase)

# Print the instacne of phrases
phrases_set = set(phrases)

if not (input_phrase in phrases_set):
    # Find top 5 most similar phrases
    top_similar_phrases = find_top_similar_phrases(input_phrase, phrases, phrase_embeddings, csv_path=csv_feedback_file_path, top_n=10)

    # Print the top 5 most similar phrases
    print("Top 5 most similar phrases:")
    for phrase, score in top_similar_phrases:
        print(f"Phrase: {phrase}, Similarity Score: {score:.4f}")

    # Save top similar phrases for feedback
    with open('top_similar_phrases.pkl', 'wb') as f:
        pickle.dump((input_phrase, top_similar_phrases), f)
        
else:
    #Case where it exists
    print("This phrase exists in the database")
