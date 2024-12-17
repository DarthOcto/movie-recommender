import streamlit as st
import pandas as pd
import numpy as np

# Load required data with caching
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv('movies.dat', sep='::', engine='python', header=None, 
                             names=['MovieID', 'Title', 'Genres'], encoding='utf-8')
    except UnicodeDecodeError:
        movies = pd.read_csv('movies.dat', sep='::', engine='python', header=None, 
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

# Optimized IBCF function using vectorization
def myIBCF_optimized(newuser, R, S, top_ratings, movie_id_to_idx, idx_to_movie_id):
    """
    Computes the (transformed) cosine similarity-based predictions for unrated movies.
    
    Args:
        newuser (np.ndarray): Array of user ratings where indices correspond to movie indices.
        R (np.ndarray): Ratings matrix (movies x users).
        S (np.ndarray): Similarity matrix (movies x movies).
        top_ratings (pd.DataFrame): DataFrame containing top-rated movies.
        movie_id_to_idx (dict): Mapping from MovieID to index.
        idx_to_movie_id (dict): Mapping from index to MovieID.
    
    Returns:
        pd.DataFrame: Top 10 movie recommendations with predicted ratings.
    """
    # Identify rated and unrated movies
    rated_indices = np.where(~np.isnan(newuser))[0]
    unrated_indices = np.where(np.isnan(newuser))[0]

    if len(rated_indices) == 0:
        # If no movies are rated, return top-rated movies
        recommendations = top_ratings[['MovieID']].head(10).copy()
        recommendations['Predicted Rating'] = np.nan
        return recommendations

    # Extract similarities for unrated movies to rated movies
    S_unrated_rated = S[unrated_indices][:, rated_indices]  # Shape: (num_unrated, num_rated)

    # Extract ratings for rated movies
    ratings_rated = newuser[rated_indices]  # Shape: (num_rated,)

    # Compute numerator and denominator
    numerator = S_unrated_rated.dot(ratings_rated)  # Shape: (num_unrated,)
    denominator = np.abs(S_unrated_rated).sum(axis=1)  # Shape: (num_unrated,)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        predicted_ratings = np.where(denominator > 0, numerator / denominator, np.nan)
    
    # Transform to [0, 1] range
    predicted_ratings = 0.5 + 0.5 * predicted_ratings

    # Create recommendations DataFrame
    recommendations = pd.DataFrame({
        'MovieID': [idx_to_movie_id[idx] for idx in unrated_indices],
        'Predicted Rating': predicted_ratings
    }).dropna()

    # Sort and select top 10
    recommendations = recommendations.sort_values(by='Predicted Rating', ascending=False).head(10)

    # Fill remaining recommendations if needed
    if top_ratings is not None and len(recommendations) < 10:
        already_rated_ids = set([idx_to_movie_id[idx] for idx in rated_indices])
        remaining_movies = top_ratings['MovieID'].loc[~top_ratings['MovieID'].isin(already_rated_ids)]
        remaining_movies = remaining_movies.head(10 - len(recommendations))
        remaining_df = pd.DataFrame({'MovieID': remaining_movies, 'Predicted Rating': np.nan})
        recommendations = pd.concat([recommendations, remaining_df], ignore_index=True)

    return recommendations

# Load data
movies, movie_rankings, S_df, R_df = load_data()

# Validate loaded data
if movies.empty or movie_rankings.empty or S_df.empty or R_df.empty:
    st.stop()

# Create mappings from MovieID to index and vice versa
movie_ids = movies['MovieID'].tolist()
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
idx_to_movie_id = {idx: movie_id for idx, movie_id in enumerate(movie_ids)}

# Convert similarity matrix S and ratings matrix R to NumPy arrays
S = S_df.loc[movie_ids, movie_ids].to_numpy()  # Ensure order matches movie_ids
R = R_df[movie_ids].to_numpy()  # Ensure columns match movie_ids

# Streamlit App Title
st.title("Movie Recommendation App")

# Persist rated movies in session state
if "rated_movies" not in st.session_state:
    st.session_state.rated_movies = {}  # Store ratings of all movies

if "sample_movies" not in st.session_state:
    st.session_state.sample_movies = movie_rankings.sample(10, random_state=42).reset_index(drop=True)

# Display buttons on the same row
button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    if st.button("Get Recommendations"):
        # Create user rating vector
        w = np.full(S.shape[0], np.nan)
        
        for movie_id, rating in st.session_state.rated_movies.items():
            if movie_id in movie_id_to_idx:
                movie_idx = movie_id_to_idx[movie_id]
                w[movie_idx] = rating

        # Get recommendations
        recommendations = myIBCF_optimized(
            newuser=w,
            R=R,
            S=S,
            top_ratings=movie_rankings,
            movie_id_to_idx=movie_id_to_idx,
            idx_to_movie_id=idx_to_movie_id
        )

        st.subheader("Your Top 10 Movie Recommendations")
        if not recommendations.empty:
            for _, row in recommendations.iterrows():
                movie_id = row['MovieID']
                title = movies.loc[movies['MovieID'] == movie_id, 'Title'].values[0]
                image_url = f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg"

                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.image(image_url, width=100, use_column_width=False)
                with col_b:
                    st.write(f"**{title}**")
        else:
            st.write("No recommendations available. Please try rating more movies!")

with button_col2:
    if st.button("Show Rated Movies"):
        # Display movies the user has already rated (excluding N/A)
        if st.session_state.rated_movies:
            rated_movies = [
                {
                    'MovieID': movie_id,
                    'Title': movies.loc[movies['MovieID'] == movie_id, 'Title'].values[0],
                    'Rating': rating
                }
                for movie_id, rating in st.session_state.rated_movies.items() if not pd.isna(rating)
            ]

            if rated_movies:
                rated_movies_df = pd.DataFrame(rated_movies)
                st.subheader("Your Rated Movies")
                for _, row in rated_movies_df.iterrows():
                    st.write(f"- **{row['Title']}**: {int(row['Rating'])} stars")
            else:
                st.write("You haven't rated any movies yet.")
        else:
            st.write("You haven't rated any movies yet.")

with button_col3:
    if st.button("Find New Movies"):
        # Precompute unrated movie IDs
        previously_rated_ids = set(st.session_state.rated_movies.keys())
        available_movies = movie_rankings[~movie_rankings['MovieID'].isin(previously_rated_ids)]

        # Efficient sampling
        if len(available_movies) >= 10:
            st.session_state.sample_movies = available_movies.sample(10, random_state=42).reset_index(drop=True)
        elif len(available_movies) > 0:
            st.session_state.sample_movies = available_movies.sample(len(available_movies), random_state=42).reset_index(drop=True)
            st.warning(f"Only {len(available_movies)} unrated movies left.")
        else:
            st.error("No unrated movies left to generate a new set!")

# Display movies to rate in 2 rows of 5
num_movies = len(st.session_state.sample_movies)
rows = [st.session_state.sample_movies.iloc[i:i + 5] for i in range(0, num_movies, 5)]

for row in rows:
    cols = st.columns(5)
    for col, (_, movie) in zip(cols, row.iterrows()):
        movie_id = movie['MovieID']
        title = movie['Title']
        image_url = f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg"

        with col:
            st.image(image_url, width=100, use_column_width=False)
            st.write(f"**{title}**")

            # Rating buttons
            if f"rating_{movie_id}" not in st.session_state:
                st.session_state[f"rating_{movie_id}"] = st.session_state.rated_movies.get(movie_id, np.nan)

            # Display rating buttons horizontally
            button_labels = ["0", "1", "2", "3", "4", "5", "N/A"]
            button_cols = st.columns(len(button_labels))
            for button_col, label in zip(button_cols, button_labels):
                with button_col:
                    if st.button(label, key=f"{movie_id}_{label}"):
                        st.session_state[f"rating_{movie_id}"] = np.nan if label == "N/A" else int(label)
                        st.session_state.rated_movies[movie_id] = st.session_state[f"rating_{movie_id}"]

            # Display current rating
            current_rating = st.session_state[f"rating_{movie_id}"]
            st.write(f"**Current Rating: {int(current_rating) if not pd.isna(current_rating) else 'N/A'}**")

st.markdown("---")
st.write("Powered by Streamlit")
