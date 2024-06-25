import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

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

def build_lda_model(texts_processed, num_topics=3, model_path='models/lda_model.model'):
    """Builds and saves an LDA model."""
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
    
    lda_model.save(model_path)
    return lda_model, texts_corpus

def build_lsi_model(texts_processed, num_topics=3, model_path='models/lsi_model.model'):
    """Builds and saves an LSI model."""
    id2word = corpora.Dictionary(texts_processed)
    texts_corpus = [id2word.doc2bow(text) for text in texts_processed]
    
    lsi_model = LsiModel(
        corpus=texts_corpus,
        id2word=id2word,
        num_topics=num_topics
    )
    
    lsi_model.save(model_path)
    return lsi_model, texts_corpus

def build_word2vec_model(texts_processed, vector_size=100, window=5, min_count=1, workers=4, model_path='models/word2vec.model'):
    """Builds and saves a Word2Vec model."""
    word2vec_model = Word2Vec(
        sentences=texts_processed,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    
    word2vec_model.save(model_path)
    return word2vec_model

def build_doc2vec_model(texts_processed, vector_size=100, window=5, min_count=1, workers=4, model_path='models/doc2vec.model'):
    """Builds and saves a Doc2Vec model."""
    tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(texts_processed)]
    
    doc2vec_model = Doc2Vec(
        documents=tagged_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    
    doc2vec_model.save(model_path)
    return doc2vec_model

def load_model(model_type, model_path):
    """Loads a saved model."""
    if model_type == 'lda':
        return LdaModel.load(model_path)
    elif model_type == 'lsi':
        return LsiModel.load(model_path)
    elif model_type == 'word2vec':
        return Word2Vec.load(model_path)
    elif model_type == 'doc2vec':
        return Doc2Vec.load(model_path)
    else:
        raise ValueError("Invalid model type. Please choose from 'lda', 'lsi', 'word2vec', or 'doc2vec'.")

def compute_topic_vectors(lda_model, texts_corpus):
    """Computes topic vectors for all materials using LDA model."""
    all_topic_distributions = [lda_model.get_document_topics(bow, minimum_probability=0) for bow in texts_corpus]
    
    def topic_distribution_to_vector(topic_distribution, num_topics):
        vec = np.zeros(num_topics)
        for topic_num, prob in topic_distribution:
            vec[topic_num] = prob
        return vec
    
    num_topics = lda_model.num_topics
    all_topic_vectors = [topic_distribution_to_vector(dist, num_topics) for dist in all_topic_distributions]
    
    return all_topic_vectors

def compute_lsi_vectors(lsi_model, texts_corpus):
    """Computes topic vectors for all materials using LSI model."""
    all_topic_vectors = [lsi_model[bow] for bow in texts_corpus]
    
    def lsi_vector_to_array(lsi_vector, num_topics):
        vec = np.zeros(num_topics)
        for topic_num, value in lsi_vector:
            vec[topic_num] = value
        return vec
    
    num_topics = lsi_model.num_topics
    all_topic_vectors = [lsi_vector_to_array(vec, num_topics) for vec in all_topic_vectors]
    
    return all_topic_vectors

def compute_doc2vec_vectors(doc2vec_model, texts_processed):
    """Computes document vectors for all materials using Doc2Vec model."""
    all_vectors = [doc2vec_model.infer_vector(doc) for doc in texts_processed]
    return all_vectors

def find_similar_materials(all_vectors, materials, material_name, top_n=5):
    """Finds most similar materials based on vectors."""
    try:
        material_index = next(i for i, material in enumerate(materials) if material.material == material_name)
    except StopIteration:
        print(f"Error: Material named '{material_name}' not found in the dataset.")
        exit(1)
    
    target_vector = all_vectors[material_index]
    similarities = cosine_similarity([target_vector], all_vectors)[0]
    similar_indices = similarities.argsort()[::-1]
    similar_indices = similar_indices[similar_indices != material_index]  # Exclude the material itself
    
    return similar_indices[:top_n], similarities[similar_indices[:top_n]]

# Main program

def main():
    file_path = input("Enter the path to the materials JSON file: ")
    
    materials = load_materials_from_json(file_path)
    
    texts_processed = tokenize_materials(materials)
    
    model_type = input("Enter the model type (lda/lsi/word2vec/doc2vec): ").strip().lower()
    model_path = f'{model_type}_model.model'
    
    if os.path.exists(model_path):
        print(f"Loading existing {model_type} model...")
        model = load_model(model_type, model_path)
    else:
        print(f"Building new {model_type} model...")
        if model_type == 'lda':
            model, texts_corpus = build_lda_model(texts_processed, model_path=model_path)
            all_vectors = compute_topic_vectors(model, texts_corpus)
        elif model_type == 'lsi':
            model, texts_corpus = build_lsi_model(texts_processed, model_path=model_path)
            all_vectors = compute_lsi_vectors(model, texts_corpus)
        elif model_type == 'word2vec':
            model = build_word2vec_model(texts_processed, model_path=model_path)
            all_vectors = [model.wv[preprocess_text(material.material)] for material in materials]
        elif model_type == 'doc2vec':
            model = build_doc2vec_model(texts_processed, model_path=model_path)
            all_vectors = compute_doc2vec_vectors(model, texts_processed)
        else:
            print("Error: Invalid model type. Please choose from 'lda', 'lsi', 'word2vec', or 'doc2vec'.")
            exit(1)
    
    material_name = input("Enter the name of the material to find similarities: ")
    
    similar_material_indices, similarity_scores = find_similar_materials(all_vectors, materials, material_name)
    
    print(f'Materials most similar to {material_name}:')
    for idx, score in zip(similar_material_indices, similarity_scores):
        print(f'{materials[idx].material} with similarity score of {score:.4f}')
        
if __name__ == "__main__":
    main()
