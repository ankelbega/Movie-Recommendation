🎬 Movie Recommendation System

A Python script that recommends movies based on genre similarity using pandas and scikit-learn.
_______________________________________________________________________________________________

Features

-Recommend top N movies based on genre similarity

-Handles missing genres

-Simple CSV-based dataset
________________________________________________________________________________________________

Requirements

1. Python 3.8+
2. pandas
3. scikit-learn

Install dependencies: pip install pandas scikit-learn
__________________________________________________________________________________________________

Dataset

CSV movies.csv (tab-separated) with columns:
MovieID, Title, Genre, Release_Year, IMDb_Rating, Director
Place it in the data/ folder.

___________________________________________________________________________________________________

**How It Works :**

1. Load dataset with pandas
2. Preprocess genres
3. Convert genres to vectors (CountVectorizer)
4. Compute cosine similarity
5. Recommend top N similar movies
