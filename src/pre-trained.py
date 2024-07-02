import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# JSON array of phrases (you can replace this with an actual JSON input if needed)
phrases_json = '''
[
    "Revestimento de pedra natural RUBICER FERRARA 15X60CM",
    "Revestimento cerâmico Artens Peak Brown 17x52 cm",
    "Revestimento decorativo preto",
    "Revestimento decorativo Artens Stone Mikeno preto"
]
'''

# Parse the JSON array to get the list of phrases
phrases = json.loads(phrases_json)

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encode the phrases
phrase_embeddings = model.encode(phrases)

# Function to find top 5 most similar phrases to the input phrase
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
input_phrase = "Revestimento Artens 17x52 cm"

# Find top 5 most similar phrases
top_similar_phrases = find_top_similar_phrases(input_phrase, phrases, phrase_embeddings)

# Print the top 5 most similar phrases
print("Top 5 most similar phrases:")
for phrase, score in top_similar_phrases:
    print(f"Phrase: {phrase}, Similarity Score: {score:.4f}")
