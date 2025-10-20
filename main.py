import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load dataset ---
# NOTE: Ensure the movies.csv file is in the 'data/' directory relative to this script,
# or change the path to '../movies.csv' or 'movies.csv' if it's in the same directory.
df = pd.read_csv('data/movies.csv', sep='\t')

# --- Preprocess genres ---
# FIXED: Changed 'genres' to the correct column name: 'Genre'
df['Genre'] = df['Genre'].fillna('').apply(lambda x: x.replace('|', ' '))

# --- Convert genres into feature vectors ---
vectorizer = CountVectorizer(stop_words='english')
# FIXED: Changed 'genres' to the correct column name: 'Genre'
genre_matrix = vectorizer.fit_transform(df['Genre'])

# --- Compute cosine similarity ---
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# --- Helper: Get recommendations ---
def recommend_movie(title, n=5):
    # Find movie index
    # FIXED: Changed 'title' to the correct column name: 'Title'
    if title not in df['Title'].values:
        print(f"Movie '{title}' not found in dataset.")
        return

    # FIXED: Changed 'title' to the correct column name: 'Title'
    idx = df[df['Title'] == title].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar movies (excluding itself)
    # FIXED: Changed 'title' to the correct column name: 'Title'
    top_movies = [df.iloc[i[0]]['Title'] for i in sim_scores[1:n+1]]

    print(f"\n🎬 Because you liked *{title}*, you might also like these movies:")
    for m in top_movies:
        print(f"👉 {m}")

# --- Example run ---
if __name__ == "__main__":
    recommend_movie("Inception")
