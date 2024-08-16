import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Load the dataset
df = pd.read_csv('job_description_data.csv')

# Define the parameters for the Word2Vec model
min_count = 10
size = 300
window = 5

# Define a function to preprocess the text
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return filtered_tokens

# Preprocess the job descriptions
job_descriptions = df['job_description'].apply(preprocess_text).tolist()

# Train the Word2Vec model
w2v_model = Word2Vec(job_descriptions, min_count=min_count, size=size, window=window)

# Save the Word2Vec model
w2v_model.save('w2v_model.bin')
