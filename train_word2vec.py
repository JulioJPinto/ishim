import nltk
nltk.download('punkt')
nltk.download('stopwords')

import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample data
texts = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"
]

# Preprocess data
stop_words = set(stopwords.words('english'))
texts_processed = [[word for word in word_tokenize(doc.lower()) if word.isalnum() and word not in stop_words] for doc in texts]

# Create Dictionary
id2word = corpora.Dictionary(texts_processed)

# Create Corpus
texts_corpus = [id2word.doc2bow(text) for text in texts_processed]

# Build LDA model
lda_model = LdaModel(corpus=texts_corpus,
                     id2word=id2word,
                     num_topics=3,
                     random_state=100,
                     update_every=1,
                     chunksize=10,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

