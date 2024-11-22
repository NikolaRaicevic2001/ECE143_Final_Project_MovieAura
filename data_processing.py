import kagglehub
import pandas as pd
from thefuzz import fuzz, process  

#########################
####### Dowload #########
#########################
# dataset_path_metacritic = kagglehub.dataset_download("kashifsahil/16000-movies-1910-2024-metacritic") 
# dataset_path_TMDB = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies") 

# print("Path to Metacritic files:", dataset_path_metacritic)
# print("Path to TMDB files:", dataset_path_TMDB)

#########################
##### Processing ########
#########################
class MovieDataset:
    def __init__(self):
        """
        Initialize the MovieDataset class by loading the dataset from the provided path.
        
        :param dataset_path: str, optional path to the downloaded dataset CSV file.
        """
        # Obtaining csv files and converting them to dataframe
        self.df_01 = pd.read_csv("./Datasets/16k_Movies.csv")
        self.df_02 = pd.read_csv("./Datasets/TMDB_movie_dataset_v11.csv")

        # print('Column names:',self.df_01.columns.tolist())
        # print('Column names:',self.df_02.columns.tolist())

        # Convert 'Release Date' column to datetime, handling any errors by setting invalid dates to NaT (Not a Time)
        self.df_01['Release Date'] = pd.to_datetime(self.df_01['Release Date'], errors='coerce')

    def get_column_names(self):
        """
        Get a list of column names in the dataset.
        
        :return: list of column names
        """
        return self.df_01.columns.tolist()

    def get_movie_titles(self, num_titles=10):
        """
        Get a specified number of movie titles.
        
        :param num_titles: int, number of movie titles to return
        :return: list of movie titles
        """
        return self.df_01['Title'].head(num_titles).tolist()

    def filter_by_year(self, start_year, end_year):
        """
        Filter movies within a specified year range.
        
        :param start_year: int, start of the year range
        :param end_year: int, end of the year range
        :return: DataFrame containing movies in the specified year range
        """

        # Extract the release year from the 'Release Date' column
        self.df_01['Release Year'] = self.df_01['Release Date'].dt.year

        # Filter the dataset by year range
        return self.df_01[(self.df_01['Release Year'] >= start_year) & (self.df_01['Release Year'] <= end_year)]

    def get_movie_info(self, title):
        """
        Get detailed information about a movie by its title.
        
        :param title: str, the title of the movie
        :return: Series containing the movie details or None if not found
        """
        movie = self.df_01[self.df_01['Title'] == title]
        return movie.iloc[0] if not movie.empty else None

    def merge_datasets(self, threshold=90):
        """
        Merge the two datasets by matching movie titles with fuzzy matching.
        
        :param threshold: int, the similarity score threshold for matching movie titles.
        :return: DataFrame, the merged dataset.
        """
        # Normalize titles in both datasets to lowercase for better matching
        self.df_01['Title_norm'] = self.df_01['Title'].str.lower()
        self.df_02['Title_norm'] = self.df_02['title'].str.lower()
        
        # Create a dictionary to store matching titles and their indexes
        matches = {}

        # Iterate over titles in the first dataset
        for idx, title in enumerate(self.df_01['Title_norm']):
            # Find the best match in the second dataset
            match, score = process.extractOne(title, self.df_02['Title_norm'], scorer=fuzz.ratio)
            
            # If the match score is above the threshold, store the match
            if score >= threshold:
                matches[idx] = self.df_02[self.df_02['Title_norm'] == match].index[0]
        
        # Create a DataFrame to map matched indices and perform the merge
        matched_df_01 = self.df_01.loc[matches.keys()].reset_index(drop=True)
        matched_df_02 = self.df_02.loc[matches.values()].reset_index(drop=True)
        
        # Drop temporary normalized title columns
        matched_df_01 = matched_df_01.drop(columns=['Title_norm'])
        matched_df_02 = matched_df_02.drop(columns=['Title_norm'])
        
        # Merge on the original 'Title' column after matching
        merged_df = pd.concat([matched_df_01, matched_df_02], axis=1)
        
        print(f"Merged {len(matches)} movies based on fuzzy title matching with threshold {threshold}")
        return None
        # return merged_df
    

# from fuzzywuzzy import fuzz, process

# dataset1 = pd.read_csv("dataset1.csv")
# dataset2 = pd.read_csv("dataset2.csv")

# titles_1 = dataset1['Title'].tolist()
# titles_2 = dataset2['title'].tolist()

# matches = {}
# for title in titles_1:
#     match, score = process.extractOne(title, titles_2, scorer=fuzz.token_sort_ratio)
#     if score > 80:
#         matches[title] = match

# dataset1['Matched Title'] = dataset1['Title'].map(matches)

# merged_dataset = pd.merge(
#     dataset1, dataset2,
#     left_on='Matched Title', right_on='title',
#     how='inner'
# )

# merged_dataset.drop(columns=['Unnamed: 0', 'title', 'Matched Title'], inplace=True)

# merged_dataset.to_csv("merged_dataset.csv", index=False)


if __name__ == "__main__":
    #########################
    ######## Dataset ########
    #########################
    # Example usage
    movies = MovieDataset()

    # # Display column names
    # print("Column Names:", movies.get_column_names())

    # # Display first 10 movie titles
    # print("First 10 Movie Titles:", movies.get_movie_titles())

    # # Filter movies released between 2000 and 2010
    # filtered_movies = movies.filter_by_year(2000, 2005)
    # print("Movies from 2000 to 2010:\n", filtered_movies.head())

    # # Get details for a specific movie
    # movie_info = movies.get_movie_info("Inception")
    # print("Movie Information:\n", movie_info)

    # Merge datasets with fuzzy matching
    merged_movies = movies.merge_datasets(threshold=90)
    print("Merged Movies DataFrame:\n", merged_movies.head())