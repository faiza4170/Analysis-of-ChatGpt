import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the dataframe
df = pd.read_csv('/path/to/your/dataframe.csv')

# Define a function to remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Group the dataframe by industry and concatenate job descriptions
grouped_df = df.groupby('industry')['job_description'].apply(lambda x: ' '.join(x))

# Remove stopwords from job descriptions
grouped_df = grouped_df.apply(lambda x: remove_stopwords(x))

# Perform topic modeling for each industry
n_topics = 5
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online', random_state=0)
top_words = {}
for industry, text in grouped_df.iteritems():
    print(f'Topic modeling for {industry}:')
    text = re.sub(r'\d+', '', text)  # remove digits
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = text.lower()  # convert to lowercase
    text = text.strip()  # remove leading/trailing whitespaces
    text = vectorizer.fit_transform([text])
    lda.fit(text)
    top_words[industry] = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words[industry].append([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-6:-1]])
        print(f'Topic {topic_idx}: {" ".join(top_words[industry][-1])}')

# Add the top_words column to the dataframe
df['top_words'] = df['industry'].apply(lambda x: top_words[x])
