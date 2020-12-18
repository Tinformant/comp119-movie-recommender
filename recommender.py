import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ast import literal_eval
from surprise import Reader, Dataset, SVD, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# [cast, crew, id]
credits = pd.read_csv('/Users/sir/Desktop/movie_data/the_movies_dataset/credits.csv')

# column: [userId, movieId, rating, timestamp]
ratings = pd.read_csv('/Users/sir/Desktop/movie_data/the_movies_dataset/ratings_small.csv')

# [movieId, imdbId, tmdbId]
links = pd.read_csv('/Users/sir/Desktop/movie_data/the_movies_dataset/links_small.csv')

# [id, keywords]
keywords = pd.read_csv('/Users/sir/Desktop/movie_data/the_movies_dataset/keywords.csv')

# [id, keywords, adult, belongs_to_collection, budget, genres, homepage, id, imdb_id, original_language, original title]
# overview, ...
metadata = pd.read_csv('/Users/sir/Desktop/movie_data/the_movies_dataset/movies_metadata.csv')


"""
~~~~~~~~~~~~~~~~~Demographic Filtering~~~~~~~~~~~~~~~~~
Cannot use average rating to rate moving: some movies only have a handful of ratings
IMDB weighted rating = v x r / (v + m) + m x c / (v + m)
* v: number of votes for the movie
* m is the minimum votes required to be listed in the chart
* r is the average rating of the movie
* c is the mean vote across the whole report
"""
# IMDB's weighted rating
c = metadata['vote_average'].mean()

m = metadata['vote_count'].quantile(0.8)


def get_weighted_rating(x, m=m, c=c):
    v = x['vote_count']
    r = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * r) + (m/(m+v) * c)


q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies['score'] = q_movies.apply(get_weighted_rating, axis=1)


"""
~~~~~~~~~~~~~~~~~Content Based Filtering~~~~~~~~~~~~~~~~~
"""
# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def get_recommendations(title, indices=indices, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

# Define new director, cast, genres and keywords features that are in a suitable form.
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


metadata['soup'] = metadata.apply(create_soup, axis=1)


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])


cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

get_recommendations('The Dark Knight Rises', cosine_sim2)


"""
~~~~~~~~~~~~~~~~~Collaborative Filtering~~~~~~~~~~~~~~~~~

User-based collaborative filtering. Model is SVD and estimator is root-mean-square-error
"""

reader = Reader()

# 100004 max movie id
max_movieId = 100004
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

svd = SVD()
model_selection.cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)

trainset = data.build_full_trainset()
svd.fit(trainset)

predict = svd.predict(5, 1, 3)

predicts = []
for i in range(0, int(max_movieId/5)):
    predict = svd.predict(6, i)
    # Tuple is (movieId, est)
    predicts.append((predict[1], predict[3]))

predicts.sort(key=lambda tup: tup[1], reverse=True)
print(predicts[:25])

