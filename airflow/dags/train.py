from __future__ import annotations

import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

import bentoml
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

EMBEDDING_SIZE = 50


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    @tf.function
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)

def train_model(**kwargs):
    curr_dir = Path(__file__).parent 
    movielens_dir = os.path.join(curr_dir, 'ml-latest-small') 
    print(curr_dir)
    if not os.path.exists(movielens_dir):
        movielens_data_file_url = (
        'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        )
        req = requests.get(movielens_data_file_url)
        print('Downloading Completed')

        file = zipfile.ZipFile(BytesIO(req.content))
        file.extractall(curr_dir)

    ratings_file = os.path.join(movielens_dir, 'ratings.csv')
    df = pd.read_csv(ratings_file)

    user_ids = df['userId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df['movieId'].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    df['user'] = df['userId'].map(user2user_encoded)
    df['movie'] = df['movieId'].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    df['rating'] = df['rating'].values.astype(np.float32)
    # min and max ratings will be used to normalize the ratings later
    min_rating = min(df['rating'])
    max_rating = max(df['rating'])

    print(
        'Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}'.format(
            num_users, num_movies, min_rating, max_rating,
        ),
    )
    df = df.sample(frac=1, random_state=42)
    x = df[['user', 'movie']].values
    # Normalize the targets between 0 and 1. Makes it easy to train.
    y = df['rating'].apply(
        lambda x: (x - min_rating) /
        (max_rating - min_rating),
    ).values

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33)

    model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001),
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=5,
        verbose=1,
        validation_data=(x_val, y_val),
    )
    bentoml.tensorflow.save_model('addition_model', model, signatures={"__call__": {"batchable": True}})
