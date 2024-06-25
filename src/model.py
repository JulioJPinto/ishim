import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from material import Material, Details
from material import read_materials_from_json

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocesses a text: tokenizes, lowers, removes stopwords and non-alphanumeric tokens."""
    stop_words = set(stopwords.words('english'))
    return [word for word in word_tokenize(text.lower()) if word.isalnum() and word not in stop_words]

def load_materials_from_json(file_path):
    """Loads materials from a JSON file."""
    try:
        materials = read_materials_from_json(file_path=file_path)
        return materials
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        exit(1)

def tokenize_materials(materials):
    """Tokenizes and preprocesses materials."""
    texts = [f'{material.material} {str(material.details).replace(", ", " ")}' for material in materials]
    return [preprocess_text(doc) for doc in texts]

def build_lda_model(texts_processed, num_topics=3):
    """Builds an LDA model."""
    id2word = corpora.Dictionary(texts_processed)
    texts_corpus = [id2word.doc2bow(text) for text in texts_processed]
    
    lda_model = LdaModel(
        corpus=texts_corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=10,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    return lda_model, texts_corpus

def compute_topic_vectors(lda_model, texts_corpus):
    """Computes topic vectors for all materials."""
    all_topic_distributions = [lda_model.get_document_topics(bow, minimum_probability=0) for bow in texts_corpus]
    
    def topic_distribution_to_vector(topic_distribution, num_topics):
        vec = np.zeros(num_topics)
        for topic_num, prob in topic_distribution:
            vec[topic_num] = prob
        return vec
    
    num_topics = lda_model.num_topics
    all_topic_vectors = [topic_distribution_to_vector(dist, num_topics) for dist in all_topic_distributions]
    
    return all_topic_vectors

def find_similar_materials(all_topic_vectors, materials, material_name, num_topics, top_n=5):
    """Finds most similar materials based on topic vectors."""
    try:
        material_index = next(i for i, material in enumerate(materials) if material.material == material_name)
    except StopIteration:
        print(f"Error: Material named '{material_name}' not found in the dataset.")
        exit(1)
    
    target_vector = all_topic_vectors[material_index]
    similarities = cosine_similarity([target_vector], all_topic_vectors)[0]
    similar_indices = similarities.argsort()[::-1]
    similar_indices = similar_indices[similar_indices != material_index]  # Exclude the material itself
    
    return similar_indices[:top_n], similarities[similar_indices[:top_n]]

# Main program

def main():
    file_path = input("Enter the path to the materials JSON file: ")
    
    materials = load_materials_from_json(file_path)
    
    texts_processed = tokenize_materials(materials)
    
    lda_model, texts_corpus = build_lda_model(texts_processed)
    
    all_topic_vectors = compute_topic_vectors(lda_model, texts_corpus)
    
    material_name = input("Enter the name of the material to find similarities: ")
    
    similar_material_indices, similarity_scores = find_similar_materials(all_topic_vectors, materials, material_name, lda_model.num_topics)
    
    print(f'Materials most similar to {material_name}:')
    for idx, score in zip(similar_material_indices, similarity_scores):
        print(f'{materials[idx].material} with similarity score of {score:.4f}')
        
if __name__ == "__main__":
    main()

