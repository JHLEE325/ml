import warnings
import pandas as pd
import numpy as np
import math
import ast as ast
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from ast import literal_eval

warnings.filterwarnings(action='ignore')

## Cosine similarity (CF item-based)
def collabo_item(title):
    # Read two datasets
    movies_ori = pd.read_csv("movies_metadata.csv", encoding='UTF-8')
    ratings_ori = pd.read_csv("ratings_small (1).csv")
    ## Preprocessing (Clean the datasets)
    # Missing value
    # remove records of movies without title.
    title_mask = movies_ori['title'].isna()
    df_movies = movies_ori.loc[title_mask == False]
    # remove records of movies with wrong 'id'
    movieId_mask = df_movies.index[[df_movies['id'].str.contains('-')]].tolist()
    df_movies.drop(movieId_mask, inplace=True)
    # fill 'NaN' value of all records of columns 'overview' and 'tagline' with ''
    df_movies['overview'] = df_movies['overview'].fillna('')
    df_movies['tagline'] = df_movies['tagline'].fillna('')
    # convert values of 'id' datatype str to int
    df_movies['id'] = pd.to_numeric(df_movies['id'])
    # convert values of 'genres' datatype dict to str, and abstract values
    df_movies['genres'] = df_movies['genres'].apply(ast.literal_eval)
    df_movies['genres'] = df_movies['genres'].apply(lambda x:[d['name'] for d in x]).apply(lambda x:' '.join(x))
    # change column name 'id' to 'movieId' & 'tagline' to 'keywords'
    df_movies.rename(columns={'id':'movieId'}, inplace=True) 
    df_movies.rename(columns={'tagline':'keywords'}, inplace=True)
    # Features selection
    df_movies = df_movies[['movieId', 'title', 'overview', 'genres', 'keywords', 'vote_average', 'vote_count']]
    df_ratings = ratings_ori.iloc[:, :-1]
    # Merge two datasets 'df_movies' & 'df_ratings'
    df = pd.merge(df_ratings, df_movies[['movieId', 'title']], on='movieId')
    # dataset for cosine similarity (CF item-based)
    data_collabo_item = df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    # cosine similarity
    sim_rate = cosine_similarity(data_collabo_item, data_collabo_item)
    sim_rate_df = pd.DataFrame(data=sim_rate, index=data_collabo_item.index, columns=data_collabo_item.index)
    sim_rate_df = pd.DataFrame(sim_rate_df[title].sort_values(ascending=False)[1:11]).reset_index()
    sim_rate_df.columns = ['Title', 'Cosine Similarity']
    sim_rate_df = sim_rate_df[['Cosine Similarity', 'Title']]
    print("- Collaborative Filtering")
    print("- Cosine simiarity (item-based)\n")
    print(sim_rate_df)
    

## Matrix Factorization
# *parameters mf(data, userId)
def matrix_faxtorization(user_id):
    # Read two datasets
    movies_ori = pd.read_csv("movies_metadata.csv", encoding='UTF-8')
    ratings_ori = pd.read_csv("ratings_small (1).csv")
    ## Preprocessing (Clean the datasets)
    # Missing value
    # remove records of movies without title.
    title_mask = movies_ori['title'].isna()
    df_movies = movies_ori.loc[title_mask == False]
    # remove records of movies with wrong 'id'
    movieId_mask = df_movies.index[[df_movies['id'].str.contains('-')]].tolist()
    df_movies.drop(movieId_mask, inplace=True)
    # fill 'NaN' value of all records of columns 'overview' and 'tagline' with ''
    df_movies['overview'] = df_movies['overview'].fillna('')
    df_movies['tagline'] = df_movies['tagline'].fillna('')
    # convert values of 'id' datatype str to int
    df_movies['id'] = pd.to_numeric(df_movies['id'])
    # convert values of 'genres' datatype dict to str, and abstract values
    df_movies['genres'] = df_movies['genres'].apply(ast.literal_eval)
    df_movies['genres'] = df_movies['genres'].apply(lambda x:[d['name'] for d in x]).apply(lambda x:' '.join(x))
    # change column name 'id' to 'movieId' & 'tagline' to 'keywords'
    df_movies.rename(columns={'id':'movieId'}, inplace=True) 
    df_movies.rename(columns={'tagline':'keywords'}, inplace=True)
    # Features selection
    df_movies = df_movies[['movieId', 'title', 'overview', 'genres', 'keywords', 'vote_average', 'vote_count']]
    df_ratings = ratings_ori.iloc[:, :-1]
    # Merge two datasets 'df_movies' & 'df_ratings'
    df = pd.merge(df_ratings, df_movies[['movieId', 'title']], on='movieId')
    # dataset for matrix factorization
    df_mat = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    # convert pivot_table dataset to numpy matrix
    matrix = df_mat.to_numpy()
    rating_mean = np.mean(matrix, axis=1)  # user's mean rating
    matrix_mean = matrix - rating_mean.reshape(-1, 1)
    # get U matrix, sigma matrix, Vt transposed matrix from 'svds' meaning 'Truncated SVD'
    U, sigma, Vt = svds(matrix_mean, k = 12)
    sigma = np.diag(sigma)
    # recover original matrix
    # dot(U, sigma, Vt) + user's mean rating
    svd_ratings = np.dot(np.dot(U, sigma), Vt) + rating_mean.reshape(-1, 1)
    df_svd = pd.DataFrame(svd_ratings, columns = df_mat.columns)
    print(df_svd)

    def recommendation(data, userId, ori_movie, ori_rating):
        # now index so do -1
        user_row_num = userId - 1
        sorted_pre = data.iloc[user_row_num].sort_values(ascending=False)
        # abstract datas with same 'userId's from original ratings dataset
        user_data = ori_rating[ori_rating.userId == userId]
        user_history = user_data.merge(ori_movie, on='movieId').sort_values(['rating'], ascending=False)
        user_history = user_history[['userId', 'movieId', 'rating']]
        # abstract datas without movie datas users have seen already from original movies dataset
        recommendations = ori_movie[~ori_movie['movieId'].isin(user_history['movieId'])]
        recommendations = recommendations.merge(pd.DataFrame(sorted_pre).reset_index(), on='movieId')
        recommendations = recommendations.rename(columns = {user_row_num: 'Predictions'}).sort_values('Predictions', ascending=False)
        recommendations = recommendations[['movieId', 'title', 'Predictions']]

        return user_history, recommendations

    already_rated, predictions = recommendation(df_svd, user_id, df_movies, df_ratings)
    print("User's history")
    print(already_rated.head(10))
    print("- Collaborative Filtering")
    print("- Matrix Factorization (SVD)\n")
    predictions = predictions[['Predictions', 'title']]
    predictions.columns = ['Predictions rate', 'Title']
    print(predictions[:10].reset_index(drop=True))
    

### Contant Based Filtering
## Cosine similarity (CBF)
def content_based(title):
    # Read dataset
    data_content = pd.read_csv("movies_metadata.csv", encoding='UTF-8')
    data_content =  data_content.loc[data_content['original_language'] == 'en', :]
    data_content = data_content[['id', 'title', 'original_language', 'genres']]
    data_content.id = data_content.id.astype(int)

    # Convert shape of genres column
    data_content['genres'] = data_content['genres'].apply(literal_eval)
    data_content['genres'] = data_content['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

    # Vectorize genre
    tfidf_vector = TfidfVectorizer()
    tfidf_matrix = tfidf_vector.fit_transform(data_content['genres']).toarray()
    tfidf_matrix_feature = tfidf_vector.get_feature_names()

    tfidf_matrix = pd.DataFrame(tfidf_matrix, columns=tfidf_matrix_feature, index = data_content.title)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix)

    cosine_sim_df = pd.DataFrame(cosine_sim, index = data_content.title, columns = data_content.title)

    # Sort by score and find top k movies
    def genre_recommendations(target_title, matrix, items, k=10):
        recom_idx = matrix.loc[:, target_title].values.reshape(1, -1).argsort()[:, ::-1].flatten()[1:k+1]
        recom_title = items.iloc[recom_idx, :].title.values
        recom_genre = items.iloc[recom_idx, :].genres.values
        target_title_list = np.full(len(range(k)), target_title)
        target_genre_list = np.full(len(range(k)), items[items.title == target_title].genres.values)
        d = {
            'target_title':target_title_list,
            'target_genre':target_genre_list,
            'recom_title' : recom_title,
            'recom_genre' : recom_genre
        }
        return pd.DataFrame(d)
    print("- Content Based Filtering")
    print("- Cosine Similarity (Genre)\n")
    print(genre_recommendations(title, cosine_sim_df, data_content))

### Association Rule Mining
## Apriori
def apriori(title, min_sup=0.08):
    # Read two datasets
    movies_ori = pd.read_csv("movies_metadata.csv", encoding='UTF-8')
    ratings_ori = pd.read_csv("ratings_small (1).csv")
    ## Preprocessing (Clean the datasets)
    # Missing value
    # remove records of movies without title.
    title_mask = movies_ori['title'].isna()
    df_movies = movies_ori.loc[title_mask == False]
    # remove records of movies with wrong 'id'
    movieId_mask = df_movies.index[[df_movies['id'].str.contains('-')]].tolist()
    df_movies.drop(movieId_mask, inplace=True)
    # fill 'NaN' value of all records of columns 'overview' and 'tagline' with ''
    df_movies['overview'] = df_movies['overview'].fillna('')
    df_movies['tagline'] = df_movies['tagline'].fillna('')
    # convert values of 'id' datatype str to int
    df_movies['id'] = pd.to_numeric(df_movies['id'])
    # convert values of 'genres' datatype dict to str, and abstract values
    df_movies['genres'] = df_movies['genres'].apply(ast.literal_eval)
    df_movies['genres'] = df_movies['genres'].apply(lambda x:[d['name'] for d in x]).apply(lambda x:' '.join(x))
    # change column name 'id' to 'movieId' & 'tagline' to 'keywords'
    df_movies.rename(columns={'id':'movieId'}, inplace=True) 
    df_movies.rename(columns={'tagline':'keywords'}, inplace=True)
    # Features selection
    df_movies = df_movies[['movieId', 'title', 'overview', 'genres', 'keywords', 'vote_average', 'vote_count']]
    df_ratings = ratings_ori.iloc[:, :-1]
    # Merge two datasets 'df_movies' & 'df_ratings'
    df = pd.merge(df_ratings, df_movies[['movieId', 'title']], on='movieId')
    # dataset for apriori
    def encode_ratings(x):  
        if x <= 0: 
            return 0
        return 1
    df_apriori = df.drop_duplicates(['userId', 'title']) # drop duplicated values in 'userId' & 'title'
    df_apriori = df_apriori.pivot_table(index='userId', columns='title', values='rating').fillna(0).astype('int64')
    data_apriori = df_apriori.applymap(encode_ratings)  # encoding values to 0 or 1
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.frequent_patterns import apriori
    frequent_itemset = apriori(data_apriori, min_support=min_sup, use_colnames=True)

    rules = association_rules(frequent_itemset, metric='lift', min_threshold=1)
    rules.sort_values(by=['lift'], ascending=False, inplace=True)
    
    df_res = rules[rules['antecedents'].apply(lambda x: len(x) == 1 and next(iter(x)) == title)]
    df_res = df_res[df_res['lift'] > 2]
    movies = df_res['consequents'].values

    movieList = []
    for movie in movies:
        lift = df_res.loc[df_res['consequents']==movie, ['lift']].values
        for title in movie:
            if title not in movieList:
                movieList.append((round(lift[0][0], 4), title))
    print("- Association Rule Mining")
    print("- Apriori\n")
    print(pd.DataFrame(movieList, columns=['Lift rate', 'Title']).drop_duplicates(['Title']).reset_index(drop=True)[:10])


# apriori('The Hours')

# collabo_item('The Hours')

# content_based('Iron Man 2')

matrix_faxtorization(673)

