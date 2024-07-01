import gensim
import numpy as np
from numpy.linalg import norm
import json


# For this file im using w2v.vectors.vk
# These are pre-trained models with multiple portuguese words
# Source: https://github.com/rdenadai/WordEmbeddingPortugues

# Load the KeyedVectors
word_vectors = gensim.models.KeyedVectors.load("models/w2v.vectors.kv", mmap='r')

def phrase_vector(phrase, word_vectors):
    words = phrase.split()
    word_vecs = [word_vectors[word] for word in words if word in word_vectors]
    if not word_vecs:
        return np.zeros(word_vectors.vector_size)
    return np.mean(word_vecs, axis=0)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def precompute_phrase_vectors(phrases, word_vectors):
    phrase_vecs = {}
    for phrase in phrases:
        phrase_vecs[phrase] = phrase_vector(phrase, word_vectors).tolist()  # Convert to list for JSON serialization
    return phrase_vecs

def save_vectors_to_json(phrase_vecs, filename):
    with open(filename, 'w') as f:
        json.dump(phrase_vecs, f)

def load_vectors_from_json(filename):
    with open(filename, 'r') as f:
        phrase_vecs = json.load(f)
    return {phrase: np.array(vec) for phrase, vec in phrase_vecs.items()}

def most_similar_phrase(target_phrase, precomputed_vectors, word_vectors):
    target_vec = phrase_vector(target_phrase, word_vectors)
    similarities = []
    for phrase, vec in precomputed_vectors.items():
        similarity = cosine_similarity(target_vec, vec)
        similarities.append((phrase, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

# Example phrases
phrases = [
    "sample material description",
    "different material type",
    "another example of material",
    "yet another description of a material",
]

# # Precompute vectors for the phrases
# precomputed_vectors = precompute_phrase_vectors(phrases, word_vectors)

# # Save the precomputed vectors to a JSON file
# save_vectors_to_json(precomputed_vectors, 'precomputed_vectors.json')

# Load the precomputed vectors from the JSON file
loaded_vectors = load_vectors_from_json('precomputed_vectors.json')

# Find the most similar phrases
target_phrase = "example material description"
similar_phrases = most_similar_phrase(target_phrase, loaded_vectors, word_vectors)
for phrase, similarity in similar_phrases:
    print(f"Phrase: {phrase}, Similarity: {similarity:.4f}")
