import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
import pickle
from torch.utils.data import DataLoader
from datasets import Dataset

# Define file paths
feedback_file_path = 'feedback.csv'
fine_tuned_model_path = 'fine-tuned-model'

# Load the fine-tuned model if available, otherwise load the pre-trained model
if os.path.exists(fine_tuned_model_path):
    model = SentenceTransformer(fine_tuned_model_path)
else:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load top similar phrases from results
with open('top_similar_phrases.pkl', 'rb') as f:
    input_phrase, top_similar_phrases = pickle.load(f)

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

    print("Feedback recorded.")

# Record feedback
record_feedback(input_phrase, top_similar_phrases)

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


# Periodically fine-tune the model based on feedback
fine_tune_model()