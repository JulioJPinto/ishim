import os
import pandas as pd
import pickle

# Define file paths
feedback_file_path = 'data/feedback.csv'

# Load top similar phrases from results
with open('top_similar_phrases.pkl', 'rb') as f:
    input_phrase, top_similar_phrases = pickle.load(f)

# Function to record feedback
def record_feedback(input_phrase, top_phrases):
    feedback_data = []
    print("Please provide feedback for the following suggestions (rate 1 to 5):")
    for phrase, _ in top_phrases:  # Ignore similarity score
        rating = None
        while rating not in range(1, 6):
            try:
                rating = int(input(f"Phrase: {phrase}, Your Rating (1-5): "))
                if rating not in range(1, 6):
                    print("Rating must be between 1 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        feedback_data.append((input_phrase, phrase, rating))
    
    # Load existing feedback or create new DataFrame
    if os.path.exists(feedback_file_path):
        feedback_df = pd.read_csv(feedback_file_path)
    else:
        feedback_df = pd.DataFrame(columns=['Input Phrase', 'Suggested Phrase', 'Rating', 'Count'])
    
    # Update or append feedback
    for input_phrase, phrase, rating in feedback_data:
        existing_entry = feedback_df[(feedback_df['Input Phrase'] == input_phrase) & 
                                     (feedback_df['Suggested Phrase'] == phrase)]
        if not existing_entry.empty:
            index = existing_entry.index[0]
            feedback_df.at[index, 'Rating'] += rating
            feedback_df.at[index, 'Count'] += 1
        else:
            new_entry = pd.DataFrame([{'Input Phrase': input_phrase, 
                                       'Suggested Phrase': phrase, 
                                       'Rating': rating, 
                                       'Count': 1}])
            feedback_df = pd.concat([feedback_df, new_entry], ignore_index=True)
    
    # Save feedback to CSV
    feedback_df.to_csv(feedback_file_path, index=False)
    print("Feedback recorded.")

# Record feedback
record_feedback(input_phrase, top_similar_phrases)
