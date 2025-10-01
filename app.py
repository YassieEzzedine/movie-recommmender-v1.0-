import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# --------------------------
# Streamlit page config
# --------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --------------------------
# Load dataset
# --------------------------
movies = pd.read_csv("../data/movies.csv")

# Create a 'tags' column (combine title + genres)
genre_cols = movies.columns[5:]  # all genre columns
movies['tags'] = movies['title'] + " " + movies[genre_cols].apply(lambda row: " ".join(row.index[row == 1]), axis=1)

# --------------------------
# Sidebar: Genre filter
# --------------------------
genre = st.sidebar.selectbox("Select Genre", ["All"] + list(movies.columns[5:]))

if genre != "All":
    filtered_movies = movies[movies[genre] == 1].reset_index(drop=True)
else:
    filtered_movies = movies.reset_index(drop=True)

# --------------------------
# Feature extraction on filtered movies
# --------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(filtered_movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# --------------------------
# Recommendation function
# --------------------------
def recommend(movie_title, movie_list, similarity_matrix):
    if movie_title not in movie_list['title'].values:
        return ["Movie not found!"]

    idx = movie_list[movie_list['title'] == movie_title].index[0]
    distances = list(enumerate(similarity_matrix[idx]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]  # top 5
    return [movie_list.iloc[i[0]].title for i in movies_list]

# --------------------------
# Streamlit UI
# --------------------------
st.title("🎬 Movie Recommender")

# Movie selection dropdown
selected_movie = st.selectbox(
    "Select a movie you like:",
    filtered_movies['title'].values
)

# Button to trigger recommendations
if st.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        recommendations = recommend(selected_movie, filtered_movies, similarity)
    
    st.subheader("Recommended Movies:")

    # Display recommendations in columns with posters
    cols = st.columns(len(recommendations))
    for i, col in enumerate(cols):
        movie_data = filtered_movies[filtered_movies['title'] == recommendations[i]]
        # Show poster if available
        if 'poster_url' in movie_data.columns:
            col.image(movie_data['poster_url'].values[0], width=150)
        col.write(recommendations[i])