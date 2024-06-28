import gensim
import numpy as np
from numpy.linalg import norm

# For this file im using w2v.vectors.vk
# These are pre-trained models with multiple portuguese words
# Source: https://github.com/rdenadai/WordEmbeddingPortugues

# Load the KeyedVectors
word_vectors = gensim.models.KeyedVectors.load("models/w2v.vectors.kv", mmap='r')

def phrase_vector(phrase, word_vectors):
    words = phrase.split()
    word_vecs = []
    for word in words:
        if word in word_vectors:
            word_vecs.append(word_vectors[word])
    if not word_vecs:
        return np.zeros(word_vectors.vector_size)
    return np.mean(word_vecs, axis=0)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def most_similar_phrase(target_phrase, phrases, word_vectors):
    target_vec = phrase_vector(target_phrase, word_vectors)
    similarities = []
    for phrase in phrases:
        phrase_vec = phrase_vector(phrase, word_vectors)
        similarity = cosine_similarity(target_vec, phrase_vec)
        similarities.append((phrase, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

# Example phrases
target_phrase = "example material description"
phrases = [
    "sample material description",
    "different material type",
    "another example of material",
    "yet another description of a material",
]

# Find the most similar phrases
similar_phrases = most_similar_phrase(target_phrase, phrases, word_vectors)
for phrase, similarity in similar_phrases:
    print(f"Phrase: {phrase}, Similarity: {similarity:.4f}")
