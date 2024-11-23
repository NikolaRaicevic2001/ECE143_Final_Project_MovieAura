# ECE143_Final_Project
This project aims to develop a personalized movie recommender system that delivers tailored recommendations based on user viewing history, genre preferences, and similar user preferences. This recommender can enhance user satisfaction by simplifying the content discovery process and minimizing search time.

# Main.py
Gathers all the movie scores from four submodules and weights them according to the relevance of each submodule to derive a final score to be sorted. From the sorted list the first five movies are chosen as the user recommendations.

# Data_Processing.py
Contains a class that processes dataset returning different fields that could be relevant for submodules 

# Submodule_01.py - 

# Submodule_02_Content_Based.py -

This submodule implements a Content-Based Filtering recommendation system that uses weighted combined features to calculate similarity scores for movies. The system enhances movie features with adjustable weights for genres, keywords, summaries, directors, and more. Key functionalities include:

Feature Combination with Weights: Merges multiple attributes (e.g., genres, keywords, and descriptions) into a weighted string for each movie, with default weights:

Genre: 2
Keywords: 3
Summary, Director, and others: 1
Similarity Calculation: Processes movie data using TF-IDF vectorization and calculates cosine similarity scores based on the weighted features.

Input Flexibility: Accepts one or multiple movie titles as input, identifies their indices and computes aggregated similarity scores against the dataset.

This system provides a customizable way to find movies similar to user-specified titles using enriched content-based recommendations.

# Submodule_03.py - 

# Submodule_04.py - 


