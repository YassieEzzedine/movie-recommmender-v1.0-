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
