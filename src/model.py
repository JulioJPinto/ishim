import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel

from material import Material, Details
from material import read_materials_from_json

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load JSON data from file
materials = read_materials_from_json("../materials.json")

# Preprocess data
stop_words = set(stopwords.words('english'))

# Combine all details into a single string for each material
texts = []
for material in materials:
    texts.append(
        f'{material.material} {str(material.details).replace(", ", " ")}'
    )

# Tokenize and remove stopwords
texts_processed = [
    [word for word in word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words]
    for doc in texts
]

# Create Dictionary
id2word = corpora.Dictionary(texts_processed)

# Create Corpus
texts_corpus = [id2word.doc2bow(text) for text in texts_processed]

# Build LDA model
lda_model = LdaModel(
    corpus=texts_corpus,
    id2word=id2word,
    num_topics=3,  # Adjust the number of topics as needed
    random_state=100,
    update_every=1,
    chunksize=10,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic: {idx} \nWords: {topic}')
