import streamlit as st
import pandas as pd
import numpy as np

# Load required data
@st.cache_data
def load_data():
    movie_rankings = pd.read_csv('movie_rankings.csv')  # Replace with your file path
    S = pd.read_csv('similarity_matrix.csv', index_col=0)  # Precomputed similarity matrix
    R = pd.read_csv('rmat.csv')
    return movie_rankings, S

# Load data
movie_rankings, S = load_data()

# Function: myIBCF
def myIBCF(newuser, R, S):
    # Placeholder for IBCF implementation
    return np.random.choice(movie_rankings['MovieID'], 10, replace=False)  # Dummy recommendation

# Title
st.title("Movie Recommendation System")

# Step 1: Persist Sample Movies for Rating
st.subheader("Rate These Movies")
st.write("Rate the following movies to receive personalized recommendations:")

# Persist sampled movies using session state
if "sample_movies" not in st.session_state:
    st.session_state.sample_movies = movie_rankings.sample(10).reset_index(drop=True)

# Display the same list of movies
user_ratings = {}
for _, row in st.session_state.sample_movies.iterrows():
    movie_id = row['MovieID']
    title = row['Title']
    user_ratings[movie_id] = st.slider(f"{title}", 0, 5, 3)  # Default rating is 3

# Step 2: Process User Input into Vector
if st.button("Get Recommendations"):
    # Initialize a ratings vector
    w = np.full(S.shape[0], np.nan)  # NaN for unrated movies
    
    # Update ratings vector with user input
    for movie_id, rating in user_ratings.items():
        movie_idx = movie_rankings[movie_rankings['MovieID'] == movie_id].index[0]
        w[movie_idx] = rating

    # Placeholder Rating Matrix (build or load your R matrix)
    R = pd.DataFrame(np.nan, index=['u1181'], columns=movie_rankings['MovieID'])  # Example format
    
    # Generate Recommendations
    recommendations = myIBCF(w, R, S)
    
    # Display Recommendations
    st.subheader("Your Top 10 Recommendations")
    if recommendations is not None:
        for movie_id in recommendations:
            movie = movie_rankings[movie_rankings['MovieID'] == movie_id]
            title = movie['Title'].values[0]
            st.write(f"- {title}")
    else:
        st.write("No recommendations available. Try rating more movies!")

st.markdown("---")
st.write("Powered by Streamlit")
