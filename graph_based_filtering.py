import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# Load movie data
file_path = './Datasets/Movies_Merged.csv'
movies_data = pd.read_csv(file_path)


# Preprocess and normalize numerical features
def preprocess_numerical_features(row):
    release_year = row["Release Year"] if not pd.isnull(row["Release Year"]) else 0
    runtime = row["runtime"] if not pd.isnull(row["runtime"]) else 0

    return release_year, runtime


# Combine text and numerical features
def combine_features_with_weights(row, weights=(1, 2, 3, 1, 1, 1, 1, 1)):
    (
        description_weight,
        genre_weight,
        keyword_weight,
        tagline_weight,
        writer_weight,
        director_weight,
        language_weight,
        company_weight,
    ) = weights
    
    description = (row["Description"] * description_weight if not pd.isnull(row["Description"]) else "")
    genres = ((row["Genres"] + " ") * genre_weight if not pd.isnull(row["Genres"]) else "")
    keywords = ((row["keywords"] + " ") * keyword_weight if not pd.isnull(row["keywords"]) else "")
    tagline = row["tagline"] * tagline_weight if not pd.isnull(row["tagline"]) else ""
    written_by = (row["Written by"] * writer_weight if not pd.isnull(row["Written by"]) else "")
    directed_by = (row["Directed by"] * director_weight if not pd.isnull(row["Directed by"]) else "")
    language = (row["original_language"] * language_weight if not pd.isnull(row["original_language"]) else "")
    production_companies = ((row["production_companies"] + " ") * company_weight 
                            if not pd.isnull(row["production_companies"]) else "")

    release_year, runtime = preprocess_numerical_features(row)

    text_features = f"{description} {genres} {keywords} {tagline} {written_by} {directed_by} {language} {production_companies}"
    return f"{text_features} {release_year} {runtime}"


# Apply feature combination
movies_data["combined_features"] = movies_data.apply(combine_features_with_weights, axis=1)

# Tokenize for BM25
tokenized_features = [doc.split() for doc in movies_data["combined_features"]]
bm25 = BM25Okapi(tokenized_features)

# Standardize numerical features
scaler = StandardScaler()
numerical_features = movies_data[["Release Year", "runtime"]].fillna(0)
scaled_numerical_features = scaler.fit_transform(numerical_features)


# Recommendation using BM25 and numerical similarity
def get_recommendations_bm25(movie_titles, top_n=10, numeric_weight=10):
    indices = pd.Series(movies_data.index, index=movies_data["Title"]).drop_duplicates()
    aggregated_scores = {}
    
    for movie_title in movie_titles:
        idx = indices.get(movie_title)
        if idx is None:
            return f"Warning: Movie '{movie_title}' not found in the dataset."
        
        query = movies_data.loc[idx, "combined_features"].split()
        bm25_scores = bm25.get_scores(query)
        movie_numerical_features = scaled_numerical_features[idx]
        numerical_similarity = cosine_similarity([movie_numerical_features], scaled_numerical_features).flatten()
        
        # Combine BM25 and numerical similarity scores
        final_scores = bm25_scores + numerical_similarity * numeric_weight      
        final_scores[idx] = -1  

        # Normalize the top N scores to probabilities
        top_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i],
                             reverse=True)[:top_n]
        top_scores = [final_scores[i] for i in top_indices]     
        total_score_top_n = sum(top_scores)
        normalized_scores = ([score / total_score_top_n for score in top_scores] 
                             if total_score_top_n > 0 else top_scores)
        
        # Accumulate the scores into the aggregated dictionary
        for i, movie_idx in enumerate(top_indices):
            movie_title_rec = movies_data.iloc[movie_idx]["Title"]
            aggregated_scores[movie_title_rec] = (aggregated_scores.get(movie_title_rec, 0) + normalized_scores[i])
    
    # Normalize and sort the aggregated scores, keep the top N
    total_aggregated_score = sum(aggregated_scores.values())
    if total_aggregated_score > 0:
        aggregated_scores = {
            title: score / total_aggregated_score 
            for title, score in aggregated_scores.items()}
    sorted_aggregated_scores = dict(sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    # Final normalization for top N
    final_total_score = sum(sorted_aggregated_scores.values())
    final_normalized_scores = {
        title: round(score / final_total_score, 3)
        for title, score in sorted_aggregated_scores.items()
    }
    
    return final_normalized_scores


# Example usage
example_movie1 = ["The Godfather", "The Martian"]
recommendations = get_recommendations_bm25(example_movie1)
print(f"Recommendations for '{example_movie1}' are: {recommendations}")

