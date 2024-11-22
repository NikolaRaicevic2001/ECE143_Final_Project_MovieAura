import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Datasets/Movies_Merged.csv'
movies_data = pd.read_csv(file_path)

#Content Based Filtering
# Enhance combined features with weights
#Weights: Genre = 2, Keywords = 3, Summary, Director = 1
def combine_features_with_weights(row, genre_weight=2, keyword_weight=3):
    description = row['Description'] if not pd.isnull(row['Description']) else ''
    genres = (row['Genres'] + ' ') * genre_weight if not pd.isnull(row['Genres']) else ''
    keywords = (row['keywords'] + ' ') * keyword_weight if not pd.isnull(row['keywords']) else ''
    tagline = row['tagline'] if not pd.isnull(row['tagline']) else ''
    written_by = row['Written by'] if not pd.isnull(row['Written by']) else ''
    directed_by = row['Directed by'] if not pd.isnull(row['Directed by']) else ''
    
    return f"{description} {genres} {keywords} {tagline} {written_by} {directed_by}"

# Fill missing values and apply enhanced feature combination
movies_data['combined_features'] = movies_data.apply(
    combine_features_with_weights, axis=1
)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['combined_features'])

# Enhanced recommendation function
def get_recommendations_content_based(movie_title, tfidf_matrix=tfidf_matrix, top_n=10):
    # Find the index of the movie that matches the title
    indices = pd.Series(movies_data.index, index=movies_data['Title']).drop_duplicates()
    idx = indices.get(movie_title)
    
    if idx is None:
        return f"Movie '{movie_title}' not found in the dataset."
    
    # Compute cosine similarity
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get top N similar movies
    sim_scores_indices = sim_scores.argsort()[-top_n - 1 : -1][::-1]
    
    # Prepare recommendations as a dictionary
    recommendations = {
        movies_data.iloc[i]['Title']: round(sim_scores[i], 3) for i in sim_scores_indices
    }
    
    return recommendations

example_movie1 = "The Godfather"
recommendations = get_recommendations_content_based(example_movie1)
print(f"Recommendation for {example_movie1} are {recommendations}")

example_movie2 = "The Martian"
recommendations = get_recommendations_content_based(example_movie2)
print(f"Recommendation for {example_movie2} are {recommendations}")