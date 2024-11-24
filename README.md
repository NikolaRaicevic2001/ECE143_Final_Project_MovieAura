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

# Submodule_03_Graph_Based.py -

This submodule implements a **hybrid Graph-Based Filtering system** that combines **BM25/TF-IDF similarity** with the **PageRank algorithm** to recommend movies. It integrates both **textual** and **numeric features**, leveraging graph theory to refine and enhance recommendation quality.

## **1. Features**

### **Text Features**
Incorporates a variety of movie attributes for textual similarity, with customizable weights:  
- **Genres**: Weight = 2  
- **Keywords**: Weight = 3  
- **Description, Tagline, Written by, Directed by, Original Language, Production Companies**: Weight = 1 each  

### **Numeric Features**  
Includes numeric attributes for similarity calculations, normalized for consistency:  
- **Release Year**: Weight = 0.1  
- **Runtime**: Weight = 0.1  

## **2. Text Information Retrieval**

### **TF-IDF Vectorization**  
Uses TF-IDF to process combined movie features, similar to Submodule_02.  

### **BM25 Scoring**  
Leverages BM25, a robust ranking function that emphasizes term relevance by accounting for term frequency and document length.  

## **3. Numeric Information Processing**

- **Normalization**: Applies `StandardScaler` from `sklearn.preprocessing` to normalize numeric features.  
- **Similarity Calculation**: Computes **cosine similarity** between normalized numeric features to quantify similarity.

## **4. Graph-Based Filtering with PageRank**

- **Graph Construction**:  
  - Treats movies as **nodes** and assigns **edge weights** based on combined BM25/TF-IDF and numeric similarity scores.  
  - Ensures weights are normalized for balanced scoring.  

- **PageRank Algorithm**:  
  - Computes **global importance scores** for each movie.  
  - Iteratively refines rankings based on graph connectivity, emphasizing both direct similarity and network-wide relationships.  

# Submodule_04.py - 

This submodule implements **Transformer Encoder** system to recommend movies based on numerical features. The model used is SASRec/BERT4Rec based Transformer-based arhitecture designed specifically for recommendation tasks. The model uses self-attention mechanism to model user sequence and can handle both sequential dependancies (past movies) and incorporate information or context based embeddings. 

 - **Input Format**: The list of movies with their specific numerical embeddings including Adult movie, Genres, Production Country, Keywords, Description, Rating, Popularity, and Release Year. The Adult category was binary either true/false; Genres and Production Country was one-hot encoded; Rating, Popularity, and Release Year was normalized so that the features are in the range of 0 to 1; Keywords and Description features were extracted using pre-trained language model BERT to create a dense vector representation.

 - **Output Format**: The model generates sequence of logits which are scores for each movie to be corrolated with the input sequence of the movies. The softmax function is later used to generate a probability distribution over the potential next movies based on the sequence and contextual information.

