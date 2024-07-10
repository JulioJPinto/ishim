import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from torch.utils.data import DataLoader
from datasets import Dataset  # Add this import

# Define file paths
csv_file_path = 'sorted_orcamento.csv'
embeddings_file_path = 'embeddings_orcamento.pkl'
feedback_file_path = 'feedback.csv'
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

# Function to record feedback
def record_feedback(input_phrase, top_phrases):
    feedback_data = []
    print("Please provide feedback for the following suggestions (rate 1 to 5):")
    for phrase, score in top_phrases:
        rating = input(f"Phrase: {phrase}, Similarity Score: {score:.4f}, Your Rating: ")
        feedback_data.append((input_phrase, phrase, score, rating))
    
    # Save feedback to CSV
    feedback_df = pd.DataFrame(feedback_data, columns=['Input Phrase', 'Suggested Phrase', 'Similarity Score', 'Rating'])
    if not os.path.exists(feedback_file_path):
        feedback_df.to_csv(feedback_file_path, index=False)
    else:
        feedback_df.to_csv(feedback_file_path, mode='a', header=False, index=False)

# Function to fine-tune the model based on feedback
def fine_tune_model():
    if not os.path.exists(feedback_file_path):
        print("No feedback available for fine-tuning.")
        return
    
    feedback_df = pd.read_csv(feedback_file_path)
    
    # Create training examples based on feedback
    train_examples = []
    for _, row in feedback_df.iterrows():
        input_phrase = row['Input Phrase']
        suggested_phrase = row['Suggested Phrase']
        rating = float(row['Rating'])
        
        # Convert rating to a score (e.g., normalize between 0 and 1)
        score = rating / 5.0
        
        # Create input example
        train_examples.append(InputExample(texts=[input_phrase, suggested_phrase], label=score))
    
    # Define a DataLoader for the training examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Use a loss function suitable for fine-tuning based on similarity
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Fine-tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    
    # Save the fine-tuned model
    model.save(fine_tuned_model_path)

# Example user input
input_phrase = "Tritubo Ø 120"

# Find top 5 most similar phrases
top_similar_phrases = find_top_similar_phrases(input_phrase, phrases, phrase_embeddings)

# Print the top 5 most similar phrases
print("Top 5 most similar phrases:")
for phrase, score in top_similar_phrases:
    print(f"Phrase: {phrase}, Similarity Score: {score:.4f}")

# Record feedback
record_feedback(input_phrase, top_similar_phrases)

# Periodically fine-tune the model based on feedback
# This can be called periodically, for example in a scheduled task
fine_tune_model()
