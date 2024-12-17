import streamlit as st
import pandas as pd
import numpy as np

# Load required data
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv(
            'movies.dat', sep='::', engine='python', header=None,
            names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1'
        )
        movie_rankings = pd.read_csv('movie_rankings.csv')
        S = pd.read_csv('modified_similarity_matrix.csv', index_col=0)
        R = pd.read_csv('rmat.csv', index_col=0)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return movies, movie_rankings, S, R

# Function: Precompute unrated movies
@st.cache_data
def precompute_unrated(movie_rankings, rated_movies):
    rated_ids = set(movie_id for movie_id, rating in rated_movies.items() if not pd.isna(rating))
    return movie_rankings[~movie_rankings['MovieID'].isin(rated_ids)]

# Function: Optimize IBCF
def myIBCF(newuser, R, S, top_ratings):
    w = np.array(newuser)
    unrated_indices = np.where(pd.isna(w))[0]
    rated_indices = np.where(~pd.isna(w))[0]
    movie_ids = R.columns

    # Precompute predictions using matrix multiplication for similarity
    predictions = []
    for i in unrated_indices:
        movie_id = movie_ids[i]
        similar_movies = S.iloc[i, rated_indices]
        rated_ratings = w[rated_indices]

        # Compute weighted prediction
        numerator = (similar_movies * rated_ratings).sum()
        denominator = similar_movies.abs().sum()
        predicted_rating = numerator / denominator if denominator > 0 else np.nan
        predictions.append((movie_id, predicted_rating))

    predictions_df = pd.DataFrame(predictions, columns=['MovieID', 'Predicted Rating']).dropna()
    predictions_df = predictions_df.sort_values(by='Predicted Rating', ascending=False).head(10)

    # Fill with top_ratings if needed
    if top_ratings is not None and len(predictions_df) < 10:
        already_rated = set(movie_ids[j] for j in rated_indices)
        remaining_movies = top_ratings[~top_ratings['MovieID'].isin(already_rated)]
        remaining_df = pd.DataFrame({'MovieID': remaining_movies['MovieID'].head(10 - len(predictions_df)),
                                     'Predicted Rating': np.nan})
        predictions_df = pd.concat([predictions_df, remaining_df])

    return predictions_df

# Load data
movies, movie_rankings, S, R = load_data()

# Streamlit App Title
st.title("Movie Recommendation App")

# Persist rated movies in session state
if "rated_movies" not in st.session_state:
    st.session_state.rated_movies = {}

if "sample_movies" not in st.session_state:
    st.session_state.sample_movies = movie_rankings.sample(10).reset_index(drop=True)

# Display movies to rate
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
        current_rating = st.session_state.rated_movies.get(movie_id, np.nan)
        col_buttons = st.columns(7)
        for i, label in enumerate(["0", "1", "2", "3", "4", "5", "N/A"]):
            with col_buttons[i]:
                if st.button(label, key=f"{movie_id}_{label}"):
                    st.session_state.rated_movies[movie_id] = np.nan if label == "N/A" else int(label)

        st.write(f"**Current Rating: {int(current_rating) if not pd.isna(current_rating) else 'N/A'}**")

# Buttons Row
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Get Recommendations"):
        w = [st.session_state.rated_movies.get(movie_id, np.nan) for movie_id in R.columns]
        recommendations = myIBCF(w, R, S, movie_rankings)

        st.subheader("Your Top 10 Movie Recommendations")
        for _, row in recommendations.iterrows():
            movie = movies[movies['MovieID'] == row['MovieID']]
            if not movie.empty:
                title = movie['Title'].values[0]
                image_url = f"https://liangfgithub.github.io/MovieImages/{row['MovieID']}.jpg"
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.image(image_url, width=100)
                with col_b:
                    st.write(f"**{title}**")
            else:
                st.write(f"- MovieID {row['MovieID']} not found in the database.")

with col2:
    if st.button("Show Rated Movies"):
        rated_movies_df = pd.DataFrame([
            {'MovieID': movie_id, 'Title': movies[movies['MovieID'] == movie_id]['Title'].values[0],
             'Rating': rating}
            for movie_id, rating in st.session_state.rated_movies.items() if not pd.isna(rating)
        ])
        st.subheader("Your Rated Movies")
        for _, row in rated_movies_df.iterrows():
            st.write(f"- **{row['Title']}**: {row['Rating']} stars")

with col3:
    if st.button("Find New Movies"):
        available_movies = precompute_unrated(movie_rankings, st.session_state.rated_movies)
        if len(available_movies) >= 10:
            st.session_state.sample_movies = available_movies.sample(10).reset_index(drop=True)
        else:
            st.error("Not enough unrated movies left!")
