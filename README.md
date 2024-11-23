# ECE143_Final_Project
This project aims to develop a personalized movie recommender system that delivers tailored recommendations based on user viewing history, genre preferences, and similar user preferences. This recommender can enhance user satisfaction by simplifying the content discovery process and minimizing search time.

# Main.py
Gathers all the movie scores from four submodules and weights them according to the relevance of each submodule to derive a final score to be sorted. From the sorted list the first five movies are chosen as the user recommendations.

# Data_Processing.py
Contains a class that processes dataset returning different fields that could be relevant for submodules 

# Submodule_01.py - 

# Submodule_02_Content_Based.py -

This submodule implements a **Content-Based Filtering** system to recommend movies based on weighted features. It enhances movie attributes and calculates similarity scores to identify similar movies.

### Key Features

1. **Feature Combination with Weights**  
   Combines multiple attributes of a movie (e.g., genres, keywords, description) into a weighted string for better similarity calculations. Default weights are:
   - **Genre**: 2  
   - **Keywords**: 3  
   - **Summary, Director, and others**: 1  

2. **TF-IDF Vectorization**  
   Processes combined features using TF-IDF vectorization to represent movies as feature vectors.

3. **Similarity Calculation**  
   Computes cosine similarity scores between movies based on their feature vectors. Supports aggregated similarity scores for multiple input movie titles.

4. **Flexible Input**  
   Accepts one or more movie titles and computes similarity scores against all movies in the dataset.

### How It Works
1. Combine weighted features for each movie using attributes like genres, keywords, description, tagline, director, etc.
2. Vectorize the combined features using TF-IDF.
3. Compute cosine similarity scores between input movies and the dataset.
4. Return aggregated similarity scores for user-specified movie titles.

# Submodule_03.py - 

# Submodule_04.py - 


