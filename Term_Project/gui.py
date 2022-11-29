import tkinter as tk
import pandas as pd
import numpy as np
import math
import ast as ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from ast import literal_eval
import warnings
warnings.filterwarnings(action='ignore')

window=tk.Tk()

window.title("Movie recommendation system")
window.resizable(False, False)

### Collaborative Filtering
## Euclidean distance (item-based)
def euclidean(data, title):
    # movie title로 movieId 가져오기
    index = df_cos[df_cos['title'] == title].index
    movie_id= list(df_cos.loc[index, 'movieId'])[0]
    # simiality function
    def sim_distance(data, n1, n2):  
        sum = 0
        # i 값은 df_euclidean 데이터셋의 선택한 row(movieId)중 평점값이 >=0 값들의 index(userId)
        for i in data.loc[n1, data.loc[n1, :]>=0].index:
            # n2 = input movieId와 값이 다른 movieId
            if data.loc[n2, i]>=0:
                sum += math.pow(data.loc[n1, i] - data.loc[n2, i], 2)
        return math.sqrt(1/(sum+1)) # return similarity value
    
    def top_match(data, movieId, rank):
        simList = []
        # i값은 df_euclidean 데이터셋 절반 중에서의 index(movieId)
        # data값이 많아 시간을 줄이기 위해 전체 중 절반만 사용
        for i in data.index[-len(data):]:
            # input movieId 와 값이 다른 movieId의 similarity값을 simList에 append
            if movieId != i:
                simList.append((sim_distance(data, movieId, i), i))
        simList.sort(reverse=True)
        return simList[:rank] # (similarity, movieId) 리스트를 return
    
    def recommendation(data, movie_id):
        res = top_match(data, movie_id, len(data))
        score_dic = {}
        sim_dic = {}
        myList = []
        for sim, mv in res:
            # similarity >= 0을 때만 실행
            if sim < 0:
                continue
            for movie in data.loc[movie_id, data.loc[movie_id, :] < 0].index:
                simSum = 0
                if data.loc[mv, movie] >= 0:
                    simSum += sim * data.loc[mv, movie]
                    score_dic.setdefault(movie, 0)
                    score_dic[movie] += simSum
                    sim_dic.setdefault(movie, 0)
                    sim_dic[movie] += sim
        for key in score_dic:
            myList.append((score_dic[key] / sim_dic[key], key))
        myList.sort(reverse=True)
        return myList
    # 추천 점수가 가장 높은 순으로 예상평점과 영화제목을 추천 (10개까지)
    movieList = []
    for rate, m_id in recommendation(data, movie_id):
        if list(df_movies.loc[df_movies['movieId']==m_id, 'title']) == []:
            continue
        movieList.append((rate, df_movies.loc[df_movies['movieId']==m_id, 'title'].values[0]))
    return "- Collaborative Filtering\n- Euclidean distance similarity\n"+ pd.DataFrame(movieList[:10], columns=['Rating', 'Title']).to_string()


## Cosine similarity (CF item-based)
def cos_item(data, title):
    # cosine similarity
    sim_rate = cosine_similarity(data, data)
    sim_rate_df = pd.DataFrame(data=sim_rate, index=data.index, columns=data.index)
    sim_rate_df = pd.DataFrame(sim_rate_df[title].sort_values(ascending=False)[1:11]).reset_index()
    sim_rate_df.columns = ['Title', 'Cosine Similarity']
    sim_rate_df = sim_rate_df[['Cosine Similarity', 'Title']]
    return "- Collaborative Filtering\n- Cosine simiarity (item-based)\n" + sim_rate_df.to_string()
    

## Matrix Factorization
# *parameters mf(data, userId)
def mf(df_mat, user_id):
    # convert pivot_table dataset to numpy matrix
    matrix = df_mat.to_numpy()
    rating_mean = np.mean(matrix, axis=1)  # user's mean rating
    matrix_mean = matrix - rating_mean.reshape(-1, 1)  # 사용자-영화에 대해 사용자평균 뺀 값
    # get U matrix, sigma matrix, Vt transposed matrix from 'svds' meaning 'Truncated SVD'
    U, sigma, Vt = svds(matrix_mean, k = 12)
    sigma = np.diag(sigma)
    # recover original matrix
    # dot(U, sigma, Vt) + user's mean rating
    svd_ratings = np.dot(np.dot(U, sigma), Vt) + rating_mean.reshape(-1, 1)
    df_svd = pd.DataFrame(svd_ratings, columns = df_mat.columns)

    def recommendation(data, userId, ori_movie, ori_rating):
        # 현재는 index로 적용되어 있어 userId-1
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
#     print("User's history")
#     print(already_rated.head(10))
    predictions = predictions[['Predictions', 'title']]
    predictions.columns = ['Predictions rate', 'Title']
    return "- Collaborative Filtering\n- Matrix Factorization (SVD)\n" + predictions[:10].reset_index(drop=True).to_string()
    

### Contant Based Filtering
## Cosine similarity (CBF)
def content_based(movie_data, target_name):
    movie_data =  movie_data.loc[movie_data['original_language'] == 'en', :]
    movie_data = movie_data[['id', 'title', 'original_language', 'genres']]

    movie_keyword = pd.read_csv('keywords.csv')

    movie_data.id = movie_data.id.astype(int)
    movie_keyword.id = movie_keyword.id.astype(int)
    movie_data = pd.merge(movie_data, movie_keyword, on='id')

    movie_data['genres'] = movie_data['genres'].apply(literal_eval)
    movie_data['genres'] = movie_data['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

    movie_data['keywords'] = movie_data['keywords'].apply(literal_eval)
    movie_data['keywords'] = movie_data['keywords'].apply(lambda x : [d['name'] for d in x]).apply(lambda x : " ".join(x))

    tfidf_vector = TfidfVectorizer()
    #tfidf_matrix = tfidf_vector.fit_transform(movie_data['genres'] + " " + movie_data['keywords']).toarray()
    tfidf_matrix = tfidf_vector.fit_transform(movie_data['genres']).toarray()
    tfidf_matrix_feature = tfidf_vector.get_feature_names()

    tfidf_matrix = pd.DataFrame(tfidf_matrix, columns=tfidf_matrix_feature, index = movie_data.title)

    cosine_sim = cosine_similarity(tfidf_matrix)

    cosine_sim_df = pd.DataFrame(cosine_sim, index = movie_data.title, columns = movie_data.title)

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
    return "- Content Based Filtering\n- Cosine Similarity (Genre)\n" + genre_recommendations(target_name, cosine_sim_df, movie_data).to_string()

### Association Rule Mining
## Apriori
def apriori(data, name, min_sup=0.08):    
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.frequent_patterns import apriori
    frequent_itemset = apriori(data, min_support=min_sup, use_colnames=True)

    rules = association_rules(frequent_itemset, metric='lift', min_threshold=1)
    rules.sort_values(by=['lift'], ascending=False, inplace=True)
    
    df_res = rules[rules['antecedents'].apply(lambda x: len(x) == 1 and next(iter(x)) == name)]
    df_res = df_res[df_res['lift'] > 2]
    movies = df_res['consequents'].values

    movieList = []
    for movie in movies:
        lift = df_res.loc[df_res['consequents']==movie, ['lift']].values
        for title in movie:
            if title not in movieList:
                movieList.append((round(lift[0][0], 4), title))
    return "- Association Rule Mining\n- Apriori\n" + pd.DataFrame(movieList, columns=['Lift rate', 'Title']).drop_duplicates(['Title']).reset_index(drop=True)[:10].to_string()

def prepare():
    # Read two datasets
    global movies_ori
    movies_ori = pd.read_csv("movies_metadata.csv", encoding='UTF-8')
    ratings_ori = pd.read_csv("ratings_small.csv")

    ## Preprocessing (Clean the datasets)
    # Missing value
    # remove records of movies without title.
    title_mask = movies_ori['title'].isna()
    global df_movies
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
    global df_ratings
    df_ratings = ratings_ori.iloc[:, :-1]
    # Merge two datasets 'df_movies' & 'df_ratings'
    df = pd.merge(df_ratings, df_movies[['movieId', 'title']], on='movieId')


    ## Datasets for each model and preprocessing
    # dataset for euclidean distance
    global df_euclidean
    df_euclidean = df.pivot_table(index='movieId', columns='userId', values='rating').fillna(-1)
    # dataset for cosine similarity (CBF)
    global df_cos
    df_cos = df_movies[['movieId', 'overview', 'title']].head(30000)
    df_cos = df_cos.reset_index(drop=True)
    # dataset for cosine similarity (CF item-based)
    global df_cos_item
    df_cos_item = df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    # dataset for matrix factorization
    global df_mf
    df_mf = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    # dataset for apriori
    def encode_ratings(x):  
        if x <= 0: 
            return 0
        return 1
    global df_apriori
    df_apriori = df.drop_duplicates(['userId', 'title']) # drop duplicated values in 'userId' & 'title'
    df_apriori = df_apriori.pivot_table(index='userId', columns='title', values='rating').fillna(0).astype('int64')
    df_apriori = df_apriori.applymap(encode_ratings)  # encoding values to 0 or 1
    # dataset for content based filtering
    df_cbf = df_movies

    resultListBox.delete(0, tk.END)
    processButton["state"] = tk.NORMAL

def process():
    resultListBox.delete(0, tk.END)
    inputString = entryMovieName.get()
    outputString = []

    if inputString == "":
        # Print error
        resultListBox.insert(0, "[Error] Invalid input")
        processButton["state"] = tk.NORMAL
        return

    outputString = apriori(df_apriori, inputString).split("\n") + content_based(movies_ori, inputString).split("\n") + euclidean(df_euclidean, inputString).split("\n") + mf(df_mf, 86).split("\n")

    # Print result
    for (i, result) in enumerate(outputString):
        resultListBox.insert(i, result)

entryMovieName = tk.Entry(window, width=150)
entryMovieName.pack()

processButton = tk.Button(window, text="Process", overrelief="solid", width=15, command=process, repeatdelay=1000, repeatinterval=100)
processButton["state"] = tk.DISABLED
processButton.pack()

frame = tk.Frame()
frame.pack()

resultListBox = tk.Listbox(frame, selectmode='extended', width=150, height=30, activestyle='none')
resultListBox.insert(0, "Please wait...")
resultListBox.pack(side="left")
scrollbar = tk.Scrollbar(frame, orient="vertical")
scrollbar.config(command=resultListBox.yview)
scrollbar.pack(side="right", fill="y")
resultListBox.config(yscrollcommand=scrollbar.set)

window.after(2000, prepare)
window.mainloop()