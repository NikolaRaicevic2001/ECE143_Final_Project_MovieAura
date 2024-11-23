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

data = pd.read_csv('Dataset/Movies_Merged.csv')

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
    
    print(f"Successfully processed {success_count}/{num_movies} images.")
    return np.array(features), success_count

def save_features(features, file_path):
    np.save(file_path, features)
    print(f"Features saved to {file_path}")


def load_features(file_path):
    if os.path.exists(file_path):
        print(f"Loading features from {file_path}")
        return np.load(file_path, allow_pickle=True)
    else:
        print(f"No saved features found at {file_path}. Please extract features first.")
        return None

def get_movie_scores():
    return [0.7, 0.4, 0.6, 0.8, 0.5, 0.9, 0.3, 0.2, 0.4, 0.6]


features = load_features("PosterFeatures/movie_features_0.npy")


# Handle missing features (e.g., due to failed image loading)
features = np.array([feat for feat in features if feat is not None])

# Compute cosine similarity
similarity_matrix = cosine_similarity(features)

# Function to recommend movies based on poster similarity
def recommend_movies(movie_index, top_k=20):
    similar_indices = np.argsort(similarity_matrix[movie_index])[::-1][1:top_k + 1]
    recommendations = data.iloc[similar_indices]
    sim_list = []
    for i in recommendations:
        sim_list.append(similarity_matrix[i])
    return recommendations, sim_list

# Example: Recommend movies for the first movie in the dataset
movie_index = 0
recommended_movies, sim_list = recommend_movies(movie_index)

print("Recommended movies:")
print(recommended_movies[['title']])
