import submodule_01
import submodule_02_Content_Based
# import submodule_03
import submodule_04_Transformer 

# Define weights for each module based on importance
weights = {
    'submodule_01': 0.5,
    'submodule_02': 0.5,
    'submodule_03': 0.5,
    'submodule_04': 0.5,
}

def main():
    # Example sequence of movie titles (representing a user's watch history)
    movie_sequence = ['Three Colors: Red', 'The Godfather', 'The Shawshank Redemption', 'The Dark Knight', 'Pulp Fiction',
                       'The Lord of the Rings: The Return of the King', 'The Lord of the Rings: The Fellowship of the Ring', 
                       'The Lord of the Rings: The Two Towers', 'The Matrix', 'The Dark Knight Rises', 'Inception', 'Interstellar',
                       'Django Unchained', 'The Prestige', 'The Departed', 'The Green Mile', 'The Lion King','The Truman Show',
                       'The Silence of the Lambs', 'The Usual Suspects']
    
    # Load movie embeddings
    _, movie_embeddings = submodule_04_Transformer.get_movie_embeddings('Dataset_Processed/Movie_Embeddings.pkl',movie_sequence)
    movie_titles = movie_embeddings['Title'].values

    # Retrieve scores from each module
<<<<<<< HEAD
    scores1 = submodule_01.get_movie_scores(movie_sequence)
    # scores2 = submodule_02.get_movie_scores(movie_sequence)
=======
    # scores1 = submodule_01.get_movie_scores(movie_sequence)
    scores2 = submodule_02_Content_Based.get_movie_scores(movie_sequence, path='./Datasets/Movies_Merged.csv')
>>>>>>> dc6b0f586685203e5c095fb2ff7de81ed72807e1
    # scores3 = submodule_03.get_movie_scores(movie_sequence)
    scores4 = submodule_04_Transformer.get_movie_scores(movie_sequence)
    
    print("Shapes of the scores:")
    print("Scores ({}) for Content Based:{}".format(scores2.shape, scores2[:10]))
    print("Scores ({}) for Transformer:{}".format(scores4.shape, scores4[:10]))

    # Compute weighted score for each movie
    weighted_scores = [
<<<<<<< HEAD
        scores1[i] * weights['submodule_01'] +
        # scores2[i] * weights['submodule_02'] +
=======
        # scores1[i] * weights['submodule_01'] +
        scores2[i] * weights['submodule_02'] +
>>>>>>> dc6b0f586685203e5c095fb2ff7de81ed72807e1
        # scores3[i] * weights['submodule_03'] +
        scores4[i] * weights['submodule_04']
        for i in range(len(scores4))
    ]
    
    # Pair movie titles with their scores and filter out movies in the input sequence
    movie_recommendations = list(zip(movie_titles, weighted_scores))
    movie_recommendations = [movie for movie in movie_recommendations if movie[0] not in movie_sequence]

    # Sort movies by score in descending order
    top_movies = sorted(movie_recommendations, key=lambda x: x[1], reverse=True)[:5]
    
    # Display the top five recommended movies
    print("Top 5 Movie Recommendations:")
    for title, score in top_movies:
        print(f"{title}: {score:.2f}")

if __name__ == "__main__":
    main()

