# import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from pyclustering.cluster.clarans import clarans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# read dataset
df = pd.read_csv('housing.csv')

# Show dataset information
print(df.info(), end="\n\n")

# Handling missing value
df = df.dropna()

# define scaler list to use
# Scaler list use 3 kinds of scaler that we learn at DataScience class
scalers = [StandardScaler(),MinMaxScaler(), RobustScaler()]

# define encoder list to use
# Encoder list use 2 kinds of encoder
encoders = [LabelEncoder(), OrdinalEncoder()]

# define algorithm list to use
# Algorithm list use 5 kinds of clustering algorithm
models = ['KMeans','EM(GMM)','DBSCAN','Spectral','CLARANS']

'''
Below 2 function are function that calculate score
'''

# Function to compute purity_score
def calculate_purity(labels, true, k):
    # Calculate purity score with 'median_house_value
    # Combine clustering results with 'median_house_value'
    labels = pd.DataFrame(labels, columns=['labels'])
    combined = pd.concat([true, labels], axis=1)

    # Sort combined datasets in ascending order by 'median_house_value'
    combined = combined.sort_values(ascending=True, by='median_house_value')

    # Divide combined datasets equally by the number of clusters
    combined['median_house_value'] = pd.cut(combined['median_house_value'], k, labels=range(0, k))

    # output of purity compared to equalized results and actual cluster results
    purity = len(combined.loc[combined['median_house_value'] == combined['labels']]) / len(combined['labels']) * 100

    return purity

# Total score data
def scores(dataset, labels, real_value, k):
    # Compute Purity
    purity = np.round(calculate_purity(labels, real_value, k), 2)

    # If the number of clusters is not 1
    if k > 2:
        # Compute silhouette score euclidean, manhattan
        silhouette_eucl = np.round(silhouette_score(dataset, labels, metric="euclidean"), 4)
        silhouette_man = np.round(silhouette_score(dataset, labels, metric="manhattan"), 4)

        # Store calculated information as a string to print
        data = "Purity Score: " + str(purity) + " %\nEuclidian Silhouette Score: " + str(silhouette_eucl) + "\nManhattan Silhouette Score: " + str(silhouette_man)
    # If the number of cluster is 1
    else:
        # Not compute silhouette
        data = "Purity Score: " + str(purity)+ " %"

    # Return score data
    return data

'''
Below 5 function are function of clustering algorithm
In each function it run several time for each combination of parameters and plot at once
In each function
- setting parameter for each clustering algorithm
- use pca to visualize cluster
- set plot window size
- run several times for each combination of parameters
- return plt
'''

# kmeans algorithm function
def algo_kmeans(feature, realvalue, num_cluster=None):
    # Parameters setting
    if num_cluster is None:
        num_cluster = [2,4,6,8,10] # number of cluster
    n_init = [10,20,30] # n_init list
    algorithm = ['auto', 'full'] # algorithm list

    # for plotting use pca algorithm
    pca = PCA(n_components=2)
    dataset = pd.DataFrame(pca.fit_transform(feature))

    # Set plot window size
    row = len(n_init)*len(algorithm)
    col = len(num_cluster)
    fig, axes = plt.subplots(row, col, constrained_layout=True, figsize=(15,30))
    fig.suptitle("K-Means\n") # name of cluster algorithm

    # Run algorithm by each parameters
    for num_k, k in enumerate(num_cluster): # number of cluster
        for num_init, init in enumerate(n_init): # number of n_init
            for num_algo, algo in enumerate(algorithm): # kind of algorithm
                kmeans = KMeans(n_clusters=k, n_init=init, algorithm=algo, random_state=42).fit(dataset)
                labels = kmeans.predict(dataset)

                # Setting the row and column of plot
                x_row = num_init * len(algorithm) + num_algo
                y_col = num_k

                # Set the combination of parameters to the title of the subplot
                # In each clustering algorithm plot all number of cases in once
                axes[x_row, y_col].set_title("[cluster = " + str(k) + ", n_init = " + str(init) + ", algorithm = " + str(algo) + "]", fontdict={'fontsize': 9})
                axes[x_row, y_col].scatter(x=dataset.iloc[:, 0], y=dataset.iloc[:, 1], c=labels, s=1.5)
                # Calculation of possible score values. (purity, Silhouette)
                scores_data = scores(dataset, labels, realvalue, k)
                # Output the obtained score value to the xlabel of the subplot
                axes[x_row, y_col].set_xlabel(scores_data, loc='left')

    # Return plot result of all parameters
    return plt

# EM(GMM) algorithm function
def algo_gmm(feature, realvalue, num_cluster=None):
    # Parameters setting
    if num_cluster is None:
        num_cluster = [2, 4, 6, 8, 10]  # num of cluster
    covariance = ['full', 'tied']  # covariance_type list
    init_params = ['kmeans', 'random']  # init_params list

    # for plotting use pca algorithm
    pca = PCA(n_components=2)
    feature = pd.DataFrame(pca.fit_transform(feature))

    # Set plot window size
    row = len(covariance) * len(init_params)
    col = len(num_cluster)
    fig, axes = plt.subplots(row, col, constrained_layout=True, figsize=(15, 30))
    fig.suptitle("EM (GMM)\n") # name of cluster algorithm

    # Run algorithm by each parameters
    for num_k, k in enumerate(num_cluster): # number of cluster
        for num_type, types in enumerate(covariance): # kind of covariance type
            for num_par, par in enumerate(init_params): # kind of init_params
                gmm = GaussianMixture(n_components=k, init_params=par, covariance_type=types)
                labels = gmm.fit_predict(feature)

                # Setting the row and column of plot
                x_row = num_type * len(init_params) + num_par
                y_col = num_k

                # Set the combination of parameters to the title of the subplot
                # In each clustering algorithm plot all number of cases in once
                axes[x_row, y_col].set_title("[cluster = " + str(k) + ", params = " + str(par) + ", type = " + str(types) + "]", fontdict={'fontsize': 9})
                axes[x_row, y_col].scatter(x=feature.iloc[:, 0], y=feature.iloc[:, 1], c=labels, s=1.5)

                # Calculation of possible score values. (purity, Silhouette)
                scores_data = scores(feature, labels, realvalue, k)
                # Output the obtained score value to the xlabel of the subplot
                axes[x_row, y_col].set_xlabel(scores_data, loc='left')

    # Return plot result of all parameters
    return plt

# DBSCAN algorithm function
def algo_dbscan(feature, realvalue):
    # Parameters setting
    eps_list = [0.3, 0.5, 0.7]  # eps(radius)
    min_samples = [5, 10, 15]  # min samples count list

    # for plotting use pca algorithm
    pca = PCA(n_components=2)
    feature = pd.DataFrame(pca.fit_transform(feature))

    # Set plot window size
    row = len(eps_list)
    col = len(min_samples)
    fig, axes = plt.subplots(row, col, constrained_layout=True, figsize=(15, 20))
    fig.suptitle("DBSCAN\n") # name of cluster algorithm

    # Run algorithm by each parameters
    for number_eps, eps in enumerate(eps_list):
        for number_samples, samples in enumerate(min_samples):
            dbscan = DBSCAN(min_samples=samples, eps=eps)
            labels = dbscan.fit_predict(feature)

            # Setting the row and column of plot
            x_row = number_eps
            y_col = number_samples

            # Set the combination of parameters to the title of the subplot
            # In each clustering algorithm plot all number of cases in once
            axes[x_row, y_col].set_title("[eps = " + str(eps) + ", min_samples = " + str(samples) + "]", fontdict={'fontsize': 13})
            axes[x_row, y_col].scatter(x=feature.iloc[:, 0], y=feature.iloc[:, 1], c=labels, s=1.5)

            # Calculation of possible score values. (purity, Silhouette)
            scores_data = scores(feature, labels, realvalue, len(set(labels.tolist())))
            # Output the obtained score value to the xlabel of the subplot
            axes[x_row, y_col].set_xlabel(scores_data, loc='left')

    # Return plot result of all parameters
    return plt

# Spectralclustering algorithm function
def algo_spectral(feature, realvalue, num_cluster=None):
    # Parameters
    if num_cluster is None:
        num_cluster = [2, 4, 6, 8, 10]  # num of cluster
    neighbors = [5, 10, 15] # num of neighbors to use when constructing the affinity matrix

    # for plotting use pca algorithm
    pca = PCA(n_components=2)
    feature = pd.DataFrame(pca.fit_transform(feature))

    # Set plot window size
    row = len(neighbors)
    col = len(num_cluster)
    fig, axes = plt.subplots(row, col, constrained_layout=True, figsize=(15, 20))
    fig.suptitle("Spectral\n") # name of cluster algorithm

    # Run algorithm by each parameters
    for num_k, k in enumerate(num_cluster):
        for num_neighbor, neighbor in enumerate(neighbors):
            spec = SpectralClustering(n_clusters=k, n_neighbors=neighbor)
            labels = spec.fit_predict(feature)

            # Setting the row and column of plot
            x_row = num_neighbor
            y_col = num_k

            # Set the combination of parameters to the title of the subplot
            # In each clustering algorithm plot all number of cases in once
            axes[x_row, y_col].set_title("[cluster = " + str(k) + ", type = " + str(neighbor) + "]", fontdict={'fontsize': 9})
            axes[x_row, y_col].scatter(x=feature.iloc[:, 0], y=feature.iloc[:, 1], c=labels, s=1.5)

            # Calculation of possible score values. (purity, Silhouette)
            scores_data = scores(feature, labels, realvalue, k)
            # Output the obtained score value to the xlabel of the subplot
            axes[x_row, y_col].set_xlabel(scores_data, loc='left')

    # Return plot result of all parameters
    return plt

# CLARANS algorithm function
def algo_clarans(feature, realvalue, num_cluster=None):
    # CLARANS' computing time is too huge so we have to reduce dataset
    # return value of clarans is different with other clustering algorithm
    # So we have to reindex dataset and mapping CLARANS label to dataset
    data_copy = feature.copy()
    realvalue_copy = realvalue.copy()
    data_copy = pd.DataFrame(data_copy)
    realvalue_copy = pd.DataFrame(realvalue_copy)
    data_copy.reset_index(inplace=True)
    realvalue_copy.reset_index(inplace=True)
    dataset_concat = pd.concat([data_copy, realvalue_copy], axis=1)
    dataset_concat = dataset_concat.sample(n=200, random_state=42)
    dataset_concat.reset_index(inplace=True)
    data_x = dataset_concat.drop(columns=["median_house_value"])
    data_y = pd.DataFrame(dataset_concat.loc[:, "median_house_value"])

    # Parameters setting
    if num_cluster is None:
        num_cluster = [2, 4, 6, 8, 10]  # number of cluster
    local_list = [2, 4]  # number of local
    neighbor_list = [3, 4]  # number of max neighbor

    # for plotting use pca algorithm
    pca = PCA(n_components=2)
    data_x = pd.DataFrame(pca.fit_transform(data_x))

    # Set plot window size
    row = len(local_list) * len(neighbor_list)
    col = len(num_cluster)
    fig, axes = plt.subplots(row, col, constrained_layout=True, figsize=(15, 18))
    fig.suptitle("CLARANS\n") # name of cluster algorithm

    # Run algorithm by each parameters
    tuple_data = data_x.values.tolist()
    for number_k, k in enumerate(num_cluster):
        for number_local, local in enumerate(local_list):
            for number_neigh, neigh in enumerate(neighbor_list):
                clarans_result = clarans(tuple_data, number_clusters=k, numlocal=local, maxneighbor=neigh)
                clarans_result.process()
                labels = clarans_result.get_clusters()  # Label
                med = clarans_result.get_medoids()  # Medoid value of the cluster

                # Setting the row and column of plot
                x_row = number_local * len(neighbor_list) + number_neigh
                y_col = number_k

                # Add label information to data by each index
                for i in range(len(labels)):
                    for j in range(len(labels[i])):
                        tmp_index = labels[i][j]
                        data_x.loc[tmp_index, 'labels'] = i

                # Set the combination of parameters to the title of the subplot
                # In each clustering algorithm plot all number of cases in once
                axes[x_row, y_col].set_title("[cluster = " + str(k) + ", numlocal = " + str(local) + ", maxneighbor = " + str(neigh) + "]", fontdict={'fontsize': 9})
                axes[x_row, y_col].scatter(x=data_x.iloc[:, 0], y=data_x.iloc[:, 1], c=data_x.loc[:, 'labels'], s=1.5)

                # Calculation of possible score values. (purity, Silhouette)
                scores_data = scores(data_x, data_x.loc[:, 'labels'], data_y, k)
                # Output the obtained score value to the xlabel of the subplot
                axes[x_row, y_col].set_xlabel(scores_data, loc='left')

                # Display medoid values separately by 'X'
                for num_med in med:
                    axes[x_row, y_col].scatter(x=data_x.iloc[num_med, 0], y=data_x.iloc[num_med, 1], c='Red', marker='x')

    # Return plot result of all parameters
    return plt

def AutoML(scalers, models, encoders, data, scaleruse=3, algouse=5, encoderuse=2, n_clusters=None):
    # Select scaler to use
    if scaleruse==3:
        print("Use Standard, MinMax, Robust scalers")
    elif scaleruse==2:
        scalers.pop()
        print("Use Standard, MinMax scaler")
    elif scaleruse==1:
        scalers.pop()
        scalers.pop()
        print("Use Standard scaler")
    elif scaleruse==0:
        print("Dataset need to Scaling\nUse Standard, MinMax, Robust scalers")
    else:
        print("Only defined 3 scalers\nUse Standard, MinMax, Robust scalers")

    # Select encoder to use
    if encoderuse==2:
        print("Use Label, Ordinal encoders")
    elif encoderuse==1:
        encoders.pop()
        print("Use Label encoder")
    elif encoderuse==0:
        print("Dataset need to encoding\nUse Label, Ordinal encoders")
    else:
        print("Only defined 2 encoders\nUse Label, Ordinal encoders")

    # Select algorithm to use
    if algouse==5:
        print("Use Kmeans, EM(GMM), DBSCAN, Spectral, CLARANS algorithms")
    elif algouse==4:
        models.pop()
        print("Use Kmeans, EM(GMM), DBSCAN, Spectral algorithms")
    elif algouse==3:
        models.pop()
        models.pop()
        print("Use Kmeans, EM(GMM), DBSCAN algorithms")
    elif algouse==2:
        models.pop()
        models.pop()
        models.pop()
        print("Use Kmeans, EM(GMM) algorithms")
    elif algouse==1:
        models.pop()
        models.pop()
        models.pop()
        models.pop()
        print("Use Kmeans algorithm")
    else:
        print("Only defined 5 algorithms\nUse Kmeans, EM(GMM), DBSCAN, Spectral, CLARANS algorithms")

    for i in encoders:
        data_copy = data.copy()
        # Reduce dataset size because of huge computing time
        data_copy = pd.DataFrame(data_copy)
        data_copy = data_copy.sample(n=3000, random_state = 42)
        # drop column median house value
        feature_data = data_copy.drop(columns="median_house_value")
        median_value = pd.DataFrame(data_copy.loc[:, "median_house_value"])
        feature_data["ocean_proximity"] = i.fit_transform(data_copy[['ocean_proximity']])
        for j in scalers:
            feature_data = j.fit_transform(feature_data)
            for a in models:
                print("\nUsed Scaler : " + str(j) + "\nUsed encoder : " + str(i) + "\nUsed algorithm : " + a)
                if a == 'KMeans':
                    result = algo_kmeans(feature_data,median_value,n_clusters)
                    result.show()
                elif a == 'EM(GMM)':
                    result = algo_gmm(feature_data, median_value, n_clusters)
                    result.show()
                elif a == 'DBSCAN':
                    result = algo_dbscan(feature_data, median_value)
                    result.show()
                elif a == 'Spectral':
                    result = algo_spectral(feature_data, median_value, n_clusters)
                    result.show()
                elif a == 'CLARANS':
                    result = algo_clarans(feature_data, median_value, n_clusters)
                    result.show()

# Run AutoML function with all features
AutoML(scalers, models, encoders, df)

# Select features to run AutoML function
df_select = df.copy()
df_select = df_select.drop(columns=["longitude","latitude","housing_median_age","total_bedrooms"])

# Run AutoML function with selected features
AutoML(scalers, models, encoders, df_select)