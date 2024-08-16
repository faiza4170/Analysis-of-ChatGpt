import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('job_description_data.csv')

# Define the number of clusters and K value for KNN
n_clusters = 5
K = 5

# Load the pre-trained Word2Vec model
w2v_model = Word2Vec.load('w2v_model.bin')

# Define a function to preprocess the text
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return filtered_tokens

# Loop through each unique industry
industry_embeddings = []
industry_clusters = {}
for industry in df['industry'].unique():
    # Extract the job descriptions for the industry
    job_descriptions = df[df['industry'] == industry]['job_description'].tolist()
    
    # Perform topic modeling on the job descriptions
    # ...
    # code to perform topic modeling goes here
    
    # Convert the topics to word embeddings
    topic_embeddings = []
    for topic in topics:
        topic_tokens = preprocess_text(topic)
        topic_embedding = np.mean([w2v_model[token] for token in topic_tokens if token in w2v_model], axis=0)
        topic_embeddings.append(topic_embedding)
    topic_embeddings = np.array(topic_embeddings)
    
    # Store the industry, topics, and embeddings in a dictionary
    industry_data = {'industry': industry, 'topics': topics, 'embeddings': topic_embeddings}
    industry_embeddings.append(topic_embeddings)
    
    # Add the industry data to the industry clusters dictionary
    for i, cluster_label in enumerate(kmeans.labels_):
        if cluster_label not in industry_clusters:
            industry_clusters[cluster_label] = []
        if cluster_label == i:
            industry_clusters[cluster_label].append(industry_data)
            
# Concatenate the embeddings for all industries to form a matrix of embeddings
embeddings_matrix = np.concatenate(industry_embeddings, axis=0)

# Use KMeans clustering to cluster the embeddings
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(embeddings_matrix)

# For each KMeans cluster, find the KNN clusters of industries based on the cosine similarity between the embeddings
for i in range(n_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    cluster_embeddings = embeddings_matrix[cluster_indices]
    similarity_matrix = cosine_similarity(cluster_embeddings)
    knn_indices = np.argsort(-similarity_matrix, axis=1)[:, :K]
    knn_ind
