import pandas as pd

# --- Convert movies (u.item) ---
# u.item columns (movieId | movieTitle | releaseDate | videoReleaseDate | IMDbURL | genres...)
movie_columns = [
    'movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

movies = pd.read_csv('data/u.item', sep='|', encoding='latin-1', names=movie_columns)
movies.to_csv('data/movies.csv', index=False)
print("movies.csv created!")

# --- Convert ratings (u.data) ---
# u.data columns: userId | movieId | rating | timestamp
ratings = pd.read_csv('data/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
ratings.to_csv('data/ratings.csv', index=False)
print("ratings.csv created!")

# --- Convert users (u.user) ---
# u.user columns: userId | age | gender | occupation | zip_code
users = pd.read_csv('data/u.user', sep='|', names=['userId', 'age', 'gender', 'occupation', 'zip_code'])
users.to_csv('data/users.csv', index=False)
print("users.csv created!")