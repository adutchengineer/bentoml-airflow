from __future__ import annotations

import json
import os

import bentoml
import numpy as np
import pandas as pd
import pydantic
from bentoml.io import JSON
from pathlib import Path

curr_dir = Path(__file__).parent
movielens_dir = os.path.join(curr_dir, 'ml-latest-small')
model_dir = os.path.join(curr_dir,'model')
ratings_file = os.path.join(movielens_dir, 'ratings.csv')
df = pd.read_csv(ratings_file)
movie_df = pd.read_csv(os.path.join(movielens_dir, 'movies.csv'))
user_ids = df['userId'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df['movieId'].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df['user'] = df['userId'].map(user2user_encoded)
df['movie'] = df['movieId'].map(movie2movie_encoded)

# `load` the model back in memory:
try:
    model = bentoml.models.import_model(
       model_dir  
    )
except:
    model = bentoml.keras.load_model('addition_model:latest')

runner = bentoml.keras.get('addition_model:latest').to_runner()

svc = bentoml.Service('movie_recommender', runners=[runner])


class KFServingInputSchema(pydantic.BaseModel):
    user_id: int


kfserving_input = JSON(
    pydantic_model=KFServingInputSchema,
    validate_json=True,
)


@svc.api(
    input=kfserving_input,
    output=JSON(),
    route='',
)
def classify(kf_input: KFServingInputSchema) -> json:
    user_id = kf_input.user_id
    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[
        ~movie_df['movieId'].isin(movies_watched_by_user.movieId.values)
    ]['movieId']
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys())),
    )
    movies_not_watched = [
        [movie2movie_encoded.get(x)] for x in movies_not_watched
    ]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched),
    )
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    print(f'Showing recommendations for user: {user_id}')
    print('====' * 9)
    print('Top 10 movie recommendations')
    print('----' * 8)
    recommended_movies = movie_df[
        movie_df['movieId'].isin(
            recommended_movie_ids,
        )
    ]
    for row in recommended_movies.itertuples():
        print(row.title, ':', row.genres)
    return {'Recommendations': recommended_movies}
