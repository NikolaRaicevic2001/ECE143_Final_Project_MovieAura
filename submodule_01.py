# submodule1.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import requests

data = pd.read_csv('Datasets/Movies_Merged.csv')

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def load_and_preprocess_image(url_path=None, target_size=(224, 224)):
    try:
        # Load and preprocess the image from a URL
        response = requests.get(url_path, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        img = Image.open(response.raw).convert('RGB')
        
        # Resize and preprocess the image
        img = img.resize(target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def extract_features(data, num_movies, TMDB_prefix="https://image.tmdb.org/t/p/w500"):
    movie_sequence = []
    features = []
    success_count = 0
    poster_col = data['poster_path']
    # id_col = data['id']
    
    for row_idx in range(num_movies):
        # movie_idx = id_col[row_idx]

        if poster_col[row_idx] and poster_col[row_idx] != float('nan'):
            try:
                url = TMDB_prefix + poster_col[row_idx]
                preprocessed_image = load_and_preprocess_image(url_path=url)
            except Exception as e:
                print(f"Error {poster_col[row_idx]=}: {e}")
                
        else:
            preprocessed_image = None

        if preprocessed_image is not None:
            try:
                feature = model.predict(preprocessed_image)
                features.append(feature.flatten())
                success_count += 1  # Increment success counter
            except Exception as e:
                print(f"Error predicting features for index {row_idx}: {e}")
                features.append(None)
        else:
            features.append(None)
        
        movie_sequence.append(data['Title'])
    
    print(f"Successfully processed {success_count}/{num_movies} images.")
    return np.array(features), success_count, movie_sequence

def save_features(features, file_path):
    np.save(file_path, features)
    print(f"Features saved to {file_path}")

features, success_count, movie_sequence = extract_features(data, len(data))
save_features(features, f"PosterFeatures/movie_features_merge.npy")

print(movie_sequence)
def load_features(file_path):
    if os.path.exists(file_path):
        print(f"Loading features from {file_path}")
        return np.load(file_path, allow_pickle=True)
    else:
        print(f"No saved features found at {file_path}. Please extract features first.")
        return None

def get_movie_scores(movie_titles, top_k):
    """
    Recommend top_k movies based on the average similarity to multiple input movies.
    
    Args:
    movie_indices (list): List of movie indices for which to find recommendations.
    top_k (int): Number of recommendations to return.

    Returns:
    DataFrame: Top-k recommended movies.
    
    """
    features = load_features("PosterFeatures/movie_features_merge.npy")
    features = np.array([feat for feat in features if feat is not None])
    similarity_matrix = cosine_similarity(features)
    print(similarity_matrix)

    # Aggregate similarity scores for the input movies
    aggregated_similarity = np.mean(similarity_matrix[movie_titles], axis=0)
    similar_indices = np.argsort(aggregated_similarity)[::-1]
    similar_indices = [idx for idx in similar_indices if idx not in movie_titles]
    
    top_similar_indices = similar_indices[:top_k]
    
    recommendations = data.iloc[top_similar_indices]
    return recommendations


#================================

# Filter rows with missing poster_path
missing_poster_indices = data[data['poster_path'].isna()].index

# Load previously saved features
existing_features = np.load('PosterFeatures/movie_features_merge.npy', allow_pickle=True)

# Process only missing posters
def fetch_missing_posters(missing_indices, data, existing_features, TMDB_prefix="https://image.tmdb.org/t/p/w500"):
    updated_features = existing_features.tolist()  # Convert to list for easier updates
    
    for row_idx in missing_indices:
        title = data.loc[row_idx, 'Title']  # Movie title (optional, for debugging)
        print(f"Processing missing poster for: {title} (Index {row_idx})")
        
        # Fetch the missing poster
        poster_path = data.loc[row_idx, 'poster_path']
        if pd.notna(poster_path):  # Ensure the path is valid
            try:
                url = TMDB_prefix + poster_path
                preprocessed_image = load_and_preprocess_image(url_path=url)
                if preprocessed_image is not None:
                    # Predict features for the poster
                    feature = model.predict(preprocessed_image)
                    updated_features[row_idx] = feature.flatten()
                    print(f"Poster features updated for index {row_idx}")
            except Exception as e:
                print(f"Error processing poster for {title}: {e}")
        else:
            print(f"No valid poster path for index {row_idx}")
    
    return np.array(updated_features, dtype=object)

# Update the features array with missing posters
updated_features = fetch_missing_posters(missing_poster_indices, data, existing_features)

# Save the updated features array
save_features(updated_features, "PosterFeatures/movie_features_merge_updated.npy")



if __name__ == 'main':
    # Example: Recommend movies for the first movie in the dataset
    movie_index = []
    recommended_movies, sim_list = get_movie_scores(movie_index, 5)

    print("Recommended movies:")
    print(recommended_movies[['title']])
