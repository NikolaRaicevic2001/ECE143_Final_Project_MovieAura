import submodule_01
import submodule_02
import submodule_03
import submodule_04

# Define the movie titles (indexed by position)
movie_titles = [
    "Movie A", "Movie B", "Movie C", "Movie D", "Movie E",
    "Movie F", "Movie G", "Movie H", "Movie I", "Movie J"
]

# Define weights for each module based on importance
weights = {
    'submodule_01': 0.5,
    'submodule_02': 0.5,
    'submodule_03': 0.5,
    'submodule_04': 0.5,
}

def main():
    # Retrieve scores from each module
    scores1 = submodule_01.get_movie_scores()
    scores2 = submodule_02.get_movie_scores()
    scores3 = submodule_03.get_movie_scores()
    scores4 = submodule_04.get_movie_scores()
    
    # Compute weighted score for each movie
    weighted_scores = [
        scores1[i] * weights['submodule_01'] +
        scores2[i] * weights['submodule_02'] +
        scores3[i] * weights['submodule_03'] +
        scores4[i] * weights['submodule_04']
        for i in range(len(scores1))
    ]
    
    # Pair movie titles with their scores
    movie_recommendations = list(zip(movie_titles, weighted_scores))
    
    # Sort movies by score in descending order
    top_movies = sorted(movie_recommendations, key=lambda x: x[1], reverse=True)[:5]
    
    # Display the top five recommended movies
    print("Top 5 Movie Recommendations:")
    for title, score in top_movies:
        print(f"{title}: {score:.2f}")

if __name__ == "__main__":
    main()
