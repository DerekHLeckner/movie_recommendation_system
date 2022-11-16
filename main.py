import json
import random
import pandas as pd
import random
import warnings
import os

from sklearn.cluster import KMeans
from numpy.linalg import norm
from scipy.sparse import lil_matrix

warnings.filterwarnings('ignore')

known_movies = set()

user_ratings = {} # List of all our movie ratings for specific users
movie_ids = []

with open("data/user_ratings.json", "r") as in_file:
    for line in in_file:
        
        this_rating = json.loads(line)
        
        known_movies.add(this_rating["title_id"])
        
        if this_rating["title_id"] not in movie_ids:
            movie_ids.append(this_rating["title_id"])
        
        this_users_ratings = user_ratings.get(this_rating["userId"], [])
        this_users_ratings.append((this_rating["title_id"], this_rating["rating"]))
        
        user_ratings[this_rating["userId"]] = this_users_ratings
        
movie_id_to_index = {m:i for i,m in enumerate(movie_ids)}

actor_id_to_name_map = {}     # Map Actor IDs to actor names
actor_id_to_index_map = {}    # Map actor IDs to a unique index of known actors
index_to_actor_ids = []       # Array mapping unique index back to actor ID (invert of actor_id_to_index_map)

index_counter = 0    # Unique actor index; increment for each new actor
known_actors = set()

movie_actor_map = {} # List of all our movies and their actors

test_count = 0
with open("data/imdb_recent_movies.json", "r") as in_file:
    for line in in_file:
        
        this_movie = json.loads(line)
        
        # Restrict to known movies
        if this_movie["title_id"] not in known_movies:
            continue
            
        # Keep track of all the actors in this movie
        for actor_id,actor_name in zip(this_movie['actor_ids'],this_movie['actor_names']):
            
            # Keep names and IDs
            actor_id_to_name_map[actor_id] = actor_name
            
            # If we've seen this actor before, skip...
            if actor_id in known_actors:
                continue
                
            # ... Otherwise, add to known actor set and create new index for them
            known_actors.add(actor_id)
            actor_id_to_index_map[actor_id] = index_counter
            index_to_actor_ids.append(actor_id)
            index_counter += 1
            
        # Finished with this film
        movie_actor_map[this_movie["title_id"]] = ({
            "movie": this_movie["title_name"],
            "actors": set(this_movie['actor_ids']),
            "genres": this_movie["title_genre"]
        })

## Generate DataFrame using Sparse Matrics

Convert our Movie Ratings data into a DataFrame that we can use for analysis.

# With sparse matrix, initialize to size of Users x Movies of 0s
matrix_sparse = lil_matrix((len(user_ratings), len(known_movies)), dtype=float)

# Update the matrix, user by user, setting non-zero values for the appropriate actors
for row,this_user in enumerate(user_ratings): 
    this_user_ratings = user_ratings[this_user]
    
    for movie_id,rating in this_user_ratings:
        this_movie_index = movie_id_to_index[movie_id]
        matrix_sparse[row,this_movie_index] = rating

df = pd.DataFrame.sparse.from_spmatrix(
    matrix_sparse, 
    index=[u for u in user_ratings],
    columns=movie_ids
).T

#  Reccomendation System Below

def get_user(df : pd.DataFrame) -> str:
    user_list = list(df.index)
    selected_user = random.choice(user_list)
    return selected_user

def get_user_cluster(df : pd.DataFrame, selected_user : str) -> int:
    cluster = df.loc[selected_user][-1]
    return cluster

def run_kmeans(df : pd.DataFrame, clusters : int) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=clusters, random_state=42).fit(df.T)
    new_df = df.T
    new_df['predicted_cluster'] = list(kmeans.labels_)
    return new_df

def get_not_watched_series(df : pd.DataFrame, selected_user : str) -> list:
    user_series = df.loc[selected_user][:-1]
    not_watched_series = list(user_series[user_series == 0].index)
    return not_watched_series

def get_watched_series(df : pd.DataFrame, selected_user : str) -> list:
    user_series = df.loc[selected_user][:-1]
    watched_series = list(user_series[user_series > 0].index)
    return watched_series

def from_group_user_not_seen_movies(new_df : pd.DataFrame, seen_movies : list, user_cluster : int, selected_user : str) -> pd.DataFrame:
    group_1 = new_df[new_df['predicted_cluster'] == user_cluster]# Select group2, the cluster 
    group_1.drop([selected_user], inplace = True) # Droop user from group
    group_1.drop(columns = ['predicted_cluster'], inplace = True) # Get rid of predicted cluster, not needed anymore
    group_1.drop(columns = seen_movies, inplace = True) # Drop movies the target user has already seen from list of what other users have seen, these are columns
    final_recommendation_listings = group_1[group_1 > 0].sum(axis = 0).sort_values(ascending = False)
    return final_recommendation_listings

def find_movies(recommendation_listing_ids : list) ->list:
    movie_name_list = [movie_actor_map[movie_id]['movie'] for movie_id in recommendation_listing_ids]
    return movie_name_list

def get_recommendations_for_user(df : pd.DataFrame, clusters : int) -> pd.DataFrame:
    df = run_kmeans(df, clusters)
    selected_user = get_user(df)
    print(f"For user {selected_user}: ")
    user_cluster = get_user_cluster(df, selected_user)
    not_watched_series = (df, selected_user)
    seen_movies = get_watched_series(df, selected_user)
    recommendation_listing_ids = list(from_group_user_not_seen_movies(df, seen_movies, user_cluster, selected_user).index)
    movie_lists = find_movies(recommendation_listing_ids)
    seen_movie_titles = [movie_actor_map[code]['movie'] for code in seen_movies]
    print(f"User seen movies:")
    print('---------------------------------------------')
    print(', '.join(seen_movie_titles))
    print('')
    print('')
    print(f"Recommended movies based upon movies seen:")
    print('---------------------------------------------')
    print(', '.join(movie_lists[:5]))    

get_recommendations_for_user(df, 3)