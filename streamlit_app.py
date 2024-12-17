import streamlit as st
import pandas as pd
import numpy as np

# Load required data
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv('https://liangfgithub.github.io/MovieData/ratings.dat?raw=true', sep='::', engine='python', header=None, 
                             names=['MovieID', 'Title', 'Genres'], encoding='utf-8')
    except UnicodeDecodeError:
        movies = pd.read_csv('https://liangfgithub.github.io/MovieData/ratings.dat?raw=true', sep='::', engine='python', header=None, 
                             names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')

    try:
        movie_rankings = pd.read_csv('movie_rankings.csv')
    except Exception as e:
        movie_rankings = pd.DataFrame()
        st.error(f"Error loading movie_rankings.csv: {e}")

    try:
        S = pd.read_csv('modified_similarity_matrix.csv', index_col=0)
    except Exception as e:
        S = pd.DataFrame()
        st.error(f"Error loading modified_similarity_matrix.csv: {e}")

    try:
        R = pd.read_csv('rmat.csv', index_col=0)
    except Exception as e:
        R = pd.DataFrame()
        st.error(f"Error loading rmat.csv: {e}")

    return movies, movie_rankings, S, R

# Function: myIBCF
def myIBCF(newuser, R, S, top_ratings):
    w = newuser.copy()
    predictions = []
    movie_ids = R.columns.tolist()

    for i in range(len(w)):
        if pd.isna(w[i]):
            movie_id = movie_ids[i]
            similar_movies = S.loc[movie_id].dropna()
            rated_indices = ~pd.isna(w)
            common_movies = similar_movies.index.intersection(
                [movie_ids[j] for j in np.where(rated_indices)[0]] )
            numerator, denominator = 0, 0
            for sim_movie in common_movies:
                j = movie_ids.index(sim_movie)
                similarity = S.loc[movie_id, sim_movie]
                numerator += similarity * w[j]
                denominator += abs(similarity)
            predicted_rating = numerator / denominator if denominator > 0 else np.nan
            predictions.append((movie_id, predicted_rating))

    predictions_df = pd.DataFrame(predictions, columns=['MovieID', 'Predicted Rating']).dropna()
    predictions_df = predictions_df.sort_values(by='Predicted Rating', ascending=False).head(10)

    if top_ratings is not None and len(predictions_df) < 10:
        already_rated = set(movie_ids[j] for j in np.where(~pd.isna(w))[0])
        remaining_movies = top_ratings['MovieID'].loc[~top_ratings['MovieID'].isin(already_rated)]
        remaining_movies = remaining_movies.head(10 - len(predictions_df))
        remaining_df = pd.DataFrame({'MovieID': remaining_movies, 'Predicted Rating': np.nan})
        predictions_df = pd.concat([predictions_df, remaining_df])

    return predictions_df

# Load data
movies, movie_rankings, S, R = load_data()

# Streamlit App Title
st.title("PSL: Movie Recommendation App")
st.subheader("vanzile2@illinois.edu, Fall 2024")
st.text("Rate the below set of movies, then click 'Get Recommendations' to get your movie recommendations!")
st.text("Click 'Find New Movies' to get a new set of movies to rate.")

if st.button("Find New Movies"):
        # Filter out movies with valid ratings (1-5)
    previously_rated_ids = {
        movie_id for movie_id, rating in st.session_state.rated_movies.items() if not pd.isna(rating)
    }
    available_movies = movie_rankings[~movie_rankings['MovieID'].isin(previously_rated_ids)]

    if len(available_movies) >= 10:
        st.session_state.sample_movies = available_movies.sample(10).reset_index(drop=True)
    else:
        st.error("Not enough unrated movies left to generate a new set!")

# Persist rated movies in session state
if "rated_movies" not in st.session_state:
    st.session_state.rated_movies = {}  # Store ratings of all movies

if "sample_movies" not in st.session_state:
    st.session_state.sample_movies = movie_rankings.sample(10).reset_index(drop=True)

# Display movies to rate
user_ratings = {}
for _, row in st.session_state.sample_movies.iterrows():
    movie_id = row['MovieID']
    title = row['Title']
    image_url = f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg"

    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(image_url, width=100)
    with col2:
        st.write(f"**{title}**")

        # Rating buttons
        if f"rating_{movie_id}" not in st.session_state:
            if movie_id in st.session_state.rated_movies:
                st.session_state[f"rating_{movie_id}"] = st.session_state.rated_movies[movie_id]
            else:
                st.session_state[f"rating_{movie_id}"] = np.nan

        col_buttons = st.columns(7)
        for i, label in enumerate(["0", "1", "2", "3", "4", "5", "N/A"]):
            with col_buttons[i]:
                if st.button(label, key=f"{movie_id}_{label}"):
                    st.session_state[f"rating_{movie_id}"] = np.nan if label == "N/A" else int(label)

        user_ratings[movie_id] = st.session_state[f"rating_{movie_id}"]
        st.session_state.rated_movies[movie_id] = st.session_state[f"rating_{movie_id}"]

        current_rating = st.session_state[f"rating_{movie_id}"]
        st.write(f"**Current Rating: {int(current_rating) if not pd.isna(current_rating) else 'N/A'}**")

# Display buttons on the same row
col1, col2 = st.columns([1, 1])  # Create two equally spaced columns

with col1:
    if st.button("Get Recommendations"):
        w = np.full(S.shape[0], np.nan)

        # Include the ratings of previously rated movies
        for movie_id, rating in st.session_state.rated_movies.items():
            if movie_id in movies['MovieID'].values:
                movie_idx = movies[movies['MovieID'] == movie_id].index[0]
                w[movie_idx] = rating

        # Create recommendations based on the updated `w` array
        recommendations = myIBCF(w, R, S, movie_rankings)

        st.subheader("Your Top 10 Movie Recommendations")
        if recommendations is not None and not recommendations.empty:
            for movie_id in recommendations['MovieID']:
                movie_id_stripped = str(movie_id).lstrip('m')  # Remove 'm' from movie ID
                movie = movies[movies['MovieID'] == int(movie_id_stripped)]  # Convert back to integer for lookup

                if not movie.empty:
                    title = movie['Title'].values[0]
                    image_url = f"https://liangfgithub.github.io/MovieImages/{movie_id_stripped}.jpg"

                    # Display image and title
                    col_a, col_b = st.columns([1, 4])  # Adjust layout as needed
                    with col_a:
                        st.image(image_url, width=100)  # Display movie poster
                    with col_b:
                        st.write(f"**{title}**")
                else:
                    st.write(f"- MovieID {movie_id} not found in the database.")
        else:
            st.write("No recommendations available. Please try rating more movies!")

with col2:
    if st.button("Show Rated Movies"):
        # Display movies the user has already rated (excluding N/A)
        rated_movies_df = pd.DataFrame([
            {'MovieID': movie_id, 'Title': movies[movies['MovieID'] == movie_id]['Title'].values[0], 'Rating': rating}
            for movie_id, rating in st.session_state.rated_movies.items() if not pd.isna(rating)
        ])

        if not rated_movies_df.empty:
            st.subheader("Your Rated Movies")
            for _, row in rated_movies_df.iterrows():
                st.write(f"- **{row['Title']}**: {row['Rating']} stars")
        else:
            st.write("You haven't rated any movies yet.")




st.markdown("---")
st.write("Powered by Streamlit")
