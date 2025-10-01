import pandas as pd

# Load the datasets
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
users = pd.read_csv('data/users.csv')

# --- Explore the first few rows ---
print("Movies dataset:")
print(movies.head())
print("\nRatings dataset:")
print(ratings.head())
print("\nUsers dataset:")
print(users.head())

# --- Info about columns and types ---
print("\nMovies info:")
print(movies.info())
print("\nRatings info:")
print(ratings.info())
print("\nUsers info:")
print(users.info())

# --- Basic statistics ---
print("\nRatings statistics:")
print(ratings.describe())
print("\nUsers statistics:")
print(users.describe())
# --- Check for missing values ---
print("\nMissing values in Movies dataset:")
print(movies.isnull().sum())

print("\nMissing values in Ratings dataset:")
print(ratings.isnull().sum())

print("\nMissing values in Users dataset:")
print(users.isnull().sum())

# --- Check for duplicates ---
print("\nDuplicate rows in Movies dataset:", movies.duplicated().sum())
print("Duplicate rows in Ratings dataset:", ratings.duplicated().sum())
print("Duplicate rows in Users dataset:", users.duplicated().sum())

# --- Remove duplicates if any ---
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)
users.drop_duplicates(inplace=True)
# List of genre columns in movies.csv
genre_columns = ['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary',
                 'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance',
                 'Sci-Fi','Thriller','War','Western']

# Sum number of movies in each genre
genre_counts = movies[genre_columns].sum().sort_values(ascending=False)
print("\nNumber of movies per genre:")
print(genre_counts)
import matplotlib.pyplot as plt

# Bar chart of number of movies per genre
plt.figure(figsize=(12,6))
genre_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Movies per Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Histogram of ratings
plt.figure(figsize=(8,5))
ratings['rating'].hist(bins=10, color='orange', edgecolor='black')
plt.title('Ratings Distribution')
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')
plt.xticks(range(1,6))
plt.show()

from sklearn.metrics.pairwise import cosine_similarity

# Use genre columns as features
features = movies[genre_columns]

# Compute cosine similarity between movies
similarity_matrix = cosine_similarity(features)

# Create a Series to map movie titles to index
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
def recommend_movies(title, num_recommendations=5):
    # Get index of the movie
    idx = movie_indices[title]
    
    # Get similarity scores for this movie with all others
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort movies based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get indices of top recommendations (skip first one, which is the movie itself)
    movie_indices_recommend = [i[0] for i in sim_scores[1:num_recommendations+1]]
    
    # Return movie titles
    return movies['title'].iloc[movie_indices_recommend]
print("Movies similar to 'Toy Story (1995)':")
print(recommend_movies('Toy Story (1995)', 5))