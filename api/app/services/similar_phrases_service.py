import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

csv_file_path = 'app/data/sorted_orcamento.csv'
embeddings_file_path = 'app/data/embeddings_orcamento.pkl'
fine_tuned_model_path = 'app/data/fine-tuned-model'
feedback_file_path = 'app/data/feedback.csv'

# Load the model
if os.path.exists(fine_tuned_model_path):
    model = SentenceTransformer(fine_tuned_model_path)
else:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def load_embeddings():
    if not os.path.exists(embeddings_file_path):
        return compute_and_save_embeddings()
    with open(embeddings_file_path, 'rb') as f:
        return pickle.load(f)
    
def process_phrase(phrase):
    tokens = phrase.split()
    tokens.sort()
    return ' '.join(tokens)


def compute_and_save_embeddings():
    df = pd.read_csv(csv_file_path)
    phrases = df['design'].tolist()
    phrases = map(process_phrase, phrases)
    phrase_embeddings = model.encode(phrases)
    with open(embeddings_file_path, 'wb') as f:
        pickle.dump((phrases, phrase_embeddings), f)
    return phrases, phrase_embeddings

phrases, phrase_embeddings = load_embeddings()

def find_top_similar_phrases(input_phrase: str, top_n=5):
    input_embedding = model.encode([input_phrase])
    similarities = cosine_similarity(input_embedding, phrase_embeddings)[0]
    
    if os.path.exists(feedback_file_path):
        df = pd.read_csv(feedback_file_path)
        input_df = df[df['Input Phrase'] == input_phrase]
        grouped_df = input_df.groupby('Suggested Phrase').agg({'Rating': 'sum', 'Count': 'sum'}).reset_index()
        grouped_df['Average Rating'] = grouped_df['Rating'] / (grouped_df['Count'] * 5)
        rating_map = dict(zip(grouped_df['Suggested Phrase'], grouped_df['Average Rating']))
    else:
        rating_map = {}

    adjusted_similarities = [
        similarities[i] * rating_map.get(phrases[i], 1) for i in range(len(phrases))
    ]
    
    top_indices = sorted(range(len(adjusted_similarities)), key=lambda i: adjusted_similarities[i], reverse=True)[:top_n]
    top_phrases = [(phrases[i], adjusted_similarities[i]) for i in top_indices]
    
    return top_phrases
