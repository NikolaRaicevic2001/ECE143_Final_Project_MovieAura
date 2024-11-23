import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = './Datasets/Movies_Merged.csv'
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


movies_data['combined_features'] = movies_data.apply(
    combine_features_with_weights, axis=1
)

# Vectorize the combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['combined_features'])

#Function can receive any number of inputs and returns 10 moviews 
def get_movie_scores(movie_titles, tfidf_matrix=tfidf_matrix):
    """
    Calculate similarity scores for all movies in the dataset based on the input movie titles.

    Parameters:
        movie_titles (list or str): A single movie title or a list of movie titles.
        tfidf_matrix (sparse matrix): TF-IDF matrix for all movies.

    Returns:
        np.ndarray: A vector of similarity scores for each movie in the dataset, in the same order as the dataset.
    """
    # Ensure movie_titles is a list
    if isinstance(movie_titles, str):
        movie_titles = [movie_titles]
    
    # Find indices of the input movies
    indices = pd.Series(movies_data.index, index=movies_data['Title']).drop_duplicates()
    input_indices = [indices.get(title) for title in movie_titles if title in indices]
    
    if not input_indices:
        raise ValueError("None of the provided movies are found in the dataset.")
    
    # Aggregate similarity scores for all provided movies
    aggregated_scores = sum(cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten() for idx in input_indices)
    
    # Return the vector of scores
    return aggregated_scores

# example_movie = ['The Godfather', 'The Traitor']
# recommendations = get_movie_scores(example_movie)
# print(recommendations)

# example_movie = ['The Martian', 'Red Planet']
# recommendations = get_movie_scores(example_movie)
# print(recommendations)