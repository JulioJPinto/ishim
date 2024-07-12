import os
import pandas as pd
from app.models.feedback import Feedback

feedback_file_path = 'app/data/feedback.csv'

def record_feedback(feedback: Feedback):
    feedback_data = [(feedback.input_phrase, item.suggested_phrase, item.rating) for item in feedback.feedback]
    
    if os.path.exists(feedback_file_path):
        feedback_df = pd.read_csv(feedback_file_path)
    else:
        feedback_df = pd.DataFrame(columns=['Input Phrase', 'Suggested Phrase', 'Rating', 'Count'])
    
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
    
    feedback_df.to_csv(feedback_file_path, index=False)
