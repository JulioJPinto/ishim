import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import functions from model.py
from model import (
    preprocess_text,
    load_materials_from_json,
    tokenize_materials,
    build_lda_model,
    build_lsi_model,
    build_doc2vec_model,
    load_model,
    compute_topic_vectors,
    compute_lsi_vectors,
    compute_doc2vec_vectors,
    find_similar_materials
)

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def compare_models(materials, texts_processed, material_name):
    # Building or loading LDA model
    lda_model_path = 'models/lda_model.model'
    if os.path.exists(lda_model_path):
        print("Loading existing LDA model...")
        lda_model = load_model('lda', lda_model_path)
        id2word = corpora.Dictionary(texts_processed)
        lda_texts_corpus = [id2word.doc2bow(text) for text in texts_processed]
    else:
        print("Building new LDA model...")
        lda_model, lda_texts_corpus = build_lda_model(texts_processed, model_path=lda_model_path)
    lda_vectors = compute_topic_vectors(lda_model, lda_texts_corpus)
    
    # Building or loading LSI model
    lsi_model_path = 'models/lsi_model.model'
    if os.path.exists(lsi_model_path):
        print("Loading existing LSI model...")
        lsi_model = load_model('lsi', lsi_model_path)
        id2word = corpora.Dictionary(texts_processed)
        lsi_texts_corpus = [id2word.doc2bow(text) for text in texts_processed]
    else:
        print("Building new LSI model...")
        lsi_model, lsi_texts_corpus = build_lsi_model(texts_processed, model_path=lsi_model_path)
    lsi_vectors = compute_lsi_vectors(lsi_model, lsi_texts_corpus)
    
    # Building or loading Doc2Vec model
    doc2vec_model_path = 'models/doc2vec.model'
    if os.path.exists(doc2vec_model_path):
        print("Loading existing Doc2Vec model...")
        doc2vec_model = load_model('doc2vec', doc2vec_model_path)
    else:
        print("Building new Doc2Vec model...")
        doc2vec_model = build_doc2vec_model(texts_processed, model_path=doc2vec_model_path)
    doc2vec_vectors = compute_doc2vec_vectors(doc2vec_model, texts_processed)
    
    # Finding similar materials using LDA model
    lda_similar_indices, lda_similarity_scores = find_similar_materials(lda_vectors, materials, material_name)
    print(f'\nLDA: Materials most similar to {material_name}:')
    for idx, score in zip(lda_similar_indices, lda_similarity_scores):
        print(f'{materials[idx].material} with similarity score of {score:.4f}')
    
    # Finding similar materials using LSI model
    lsi_similar_indices, lsi_similarity_scores = find_similar_materials(lsi_vectors, materials, material_name)
    print(f'\nLSI: Materials most similar to {material_name}:')
    for idx, score in zip(lsi_similar_indices, lsi_similarity_scores):
        print(f'{materials[idx].material} with similarity score of {score:.4f}')
    
    # Finding similar materials using Doc2Vec model
    doc2vec_similar_indices, doc2vec_similarity_scores = find_similar_materials(doc2vec_vectors, materials, material_name)
    print(f'\nDoc2Vec: Materials most similar to {material_name}:')
    for idx, score in zip(doc2vec_similar_indices, doc2vec_similarity_scores):
        print(f'{materials[idx].material} with similarity score of {score:.4f}')

def main():
    file_path = input("Enter the path to the materials JSON file: ")
    
    materials = load_materials_from_json(file_path)
    
    texts_processed = tokenize_materials(materials)
    
    material_name = input("Enter the name of the material to find similarities: ")
    
    compare_models(materials, texts_processed, material_name)
    
if __name__ == "__main__":
    main()
