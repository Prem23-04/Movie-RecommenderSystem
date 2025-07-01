import pickle
import pandas as pd
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load title and vote average data
movies = pd.read_pickle('title_votes3.pkl')
movies = movies.rename(columns={'original_title': 'title', 'vote_average': 'rating'})

# Load similarity matrix
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Load genres and merge
genres_df = pd.read_csv('movie_genre.csv')  # Ensure this file has 'title' and 'genres' columns
genres_df = genres_df.rename(columns={'genres': 'genre'})
movies = pd.merge(movies, genres_df, on='title', how='left')

# Recommendation function
def recommend(movie_name):
    try:
        index = np.where(movies['title'] == movie_name)[0][0]
    except IndexError:
        return []
    distances = similarity[index]
    movie_list = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:6]
    return movies.iloc[[i[0] for i in movie_list]][['title', 'rating', 'genre']].to_dict(orient='records')

# Homepage
@app.route('/')
def index():
    display_movies = movies[['title', 'rating', 'genre']].iloc[:100]
    return render_template('index.html', movies=display_movies.to_dict(orient='records'))

# Recommendation route
@app.route('/recommend', methods=['POST'])
def show_recommendations():
    movie_name = request.form['movie']
    recommended_movies = recommend(movie_name)
    return render_template('recommend.html', selected=movie_name, recommended=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
