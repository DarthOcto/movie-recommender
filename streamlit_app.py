import streamlit as st
import pandas as pd
import numpy as np

# Load required data
@st.cache_data
def load_data():
    movie_rankings = pd.read_csv('movie_rankings.csv')  # Replace with your file path
    S = pd.read_csv('modified_similarity_matrix.csv', index_col=0)  # Precomputed similarity matrix
    R = pd.read_csv('rmat.csv')
    return movie_rankings, S, R

# Load data
movie_rankings, S, R = load_data()

# Function: myIBCF
def myIBCF(newuser, R, S, top_ratings):
    """
    Input:
    - newuser: A 3706x1 vector of movie ratings (some NA values).
    - R: The rating matrix (users x movies).
    - S: The similarity matrix (movies x movies).
    - top_ratings: Optional DataFrame with popular movies for backup recommendations.
    
    Output:
    - Top 10 movie recommendations for the new user with predicted ratings.
    """
    w = newuser.copy()  # User ratings (input)
    predictions = []  # To store (movie_id, predicted_rating)

    # Step 1: Get movie IDs (from R's columns)
    movie_ids = R.columns.tolist()
    
    # Step 2: Iterate over all movies to compute predictions
    for i in range(len(w)):
        if pd.isna(w[i]):  # Only predict for unrated movies
            movie_id = movie_ids[i]
            
            # Step 3: Identify similar movies with ratings
            similar_movies = S.loc[movie_id].dropna()
            rated_indices = ~pd.isna(w)  # Indices of rated movies
            common_movies = similar_movies.index.intersection([movie_ids[j] for j in np.where(rated_indices)[0]])
            
            # Step 4: Compute prediction using the formula
            numerator = 0
            denominator = 0
            for sim_movie in common_movies:
                j = movie_ids.index(sim_movie)  # Find index in newuser
                similarity = S.loc[movie_id, sim_movie]
                numerator += similarity * w[j]
                denominator += abs(similarity)
                
            if denominator > 0:
                predicted_rating = numerator / denominator
            else:
                predicted_rating = np.nan
            
            predictions.append((movie_id, predicted_rating))
    
    # Step 5: Rank predictions and select top 10
    predictions_df = pd.DataFrame(predictions, columns=['MovieID', 'Predicted Rating']).dropna()
    predictions_df = predictions_df.sort_values(by='Predicted Rating', ascending=False).head(10)
    
    # Step 6: Fill remaining slots with popular movies, if needed
    if top_ratings is not None and len(predictions_df) < 10:
        already_rated = set(movie_ids[j] for j in np.where(~pd.isna(w))[0])
        remaining_movies = top_ratings['MovieID'].loc[~top_ratings['MovieID'].isin(already_rated)]
        remaining_movies = remaining_movies.head(10 - len(predictions_df))
        remaining_df = pd.DataFrame({'MovieID': remaining_movies, 'Predicted Rating': np.nan})
        predictions_df = pd.concat([predictions_df, remaining_df])

    return predictions_df  # Return as DataFrame

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

    print(w)

    # Placeholder Rating Matrix (build or load your R matrix)
    #R = pd.DataFrame(np.nan, index=['u1181'], columns=movie_rankings['MovieID'])  # Example format
    
    # Generate Recommendations
    recommendations = myIBCF(w, R, S, movie_rankings)

    #print(recommendations)
    
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
