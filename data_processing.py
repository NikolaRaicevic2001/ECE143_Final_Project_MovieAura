import kagglehub
import pandas as pd

#########################
######## Dataset ########
#########################
# 16000+ Movies 1910-2024 (Metacritic)
path = kagglehub.dataset_download("kashifsahil/16000-movies-1910-2024-metacritic")
print("Path to dataset files:", path)

#########################
##### Processing ########
#########################
class MovieDataset:
    def __init__(self, dataset_path=None):
        """
        Initialize the MovieDataset class by loading the dataset from the provided path.
        
        :param dataset_path: str, optional path to the downloaded dataset CSV file.
        """
        # Download the dataset using kagglehub if a path is not provided
        if dataset_path is None:
            dataset_path = kagglehub.dataset_download("kashifsahil/16000-movies-1910-2024-metacritic") + "/16k_Movies.csv"
        self.df = pd.read_csv(dataset_path)

        # Convert 'Release Date' column to datetime, handling any errors by setting invalid dates to NaT (Not a Time)
        self.df['Release Date'] = pd.to_datetime(self.df['Release Date'], errors='coerce')

    def get_column_names(self):
        """
        Get a list of column names in the dataset.
        
        :return: list of column names
        """
        return self.df.columns.tolist()

    def get_movie_titles(self, num_titles=10):
        """
        Get a specified number of movie titles.
        
        :param num_titles: int, number of movie titles to return
        :return: list of movie titles
        """
        return self.df['Title'].head(num_titles).tolist()

    def filter_by_year(self, start_year, end_year):
        """
        Filter movies within a specified year range.
        
        :param start_year: int, start of the year range
        :param end_year: int, end of the year range
        :return: DataFrame containing movies in the specified year range
        """

        # Extract the release year from the 'Release Date' column
        self.df['Release Year'] = self.df['Release Date'].dt.year

        # Filter the dataset by year range
        return self.df[(self.df['Release Year'] >= start_year) & (self.df['Release Year'] <= end_year)]

    def get_movie_info(self, title):
        """
        Get detailed information about a movie by its title.
        
        :param title: str, the title of the movie
        :return: Series containing the movie details or None if not found
        """
        movie = self.df[self.df['Title'] == title]
        return movie.iloc[0] if not movie.empty else None

# Example usage
path_Nikola = '/root/.cache/kagglehub/datasets/kashifsahil/16000-movies-1910-2024-metacritic/versions/1/16k_Movies.csv'
movies = MovieDataset(dataset_path=path_Nikola)

# Display column names
print("Column Names:", movies.get_column_names())

# Display first 10 movie titles
print("First 10 Movie Titles:", movies.get_movie_titles())

# Filter movies released between 2000 and 2010
filtered_movies = movies.filter_by_year(2000, 2005)
print("Movies from 2000 to 2010:\n", filtered_movies.head())

# Get details for a specific movie
movie_info = movies.get_movie_info("Inception")
print("Movie Information:\n", movie_info)
