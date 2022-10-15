import numpy as np
import pandas as pd
import warnings
from sklearn import feature_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
warnings.filterwarnings(action='ignore')

# Read data from file
df = pd.read_csv('breast-cancer-wisconsin.data')

# Add columns to dataset
df.columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
              'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# Show dataset information
print(df.info(), end="\n\n")

# Handling missing value
df = df.apply(pd.to_numeric, errors='coerce').fillna(np.nan).dropna()

# Apply SelectKBest class to extract all features ranking
X_independent=df.iloc[:,1:10]
y_target=df.iloc[:,-1]
bf = feature_selection.SelectKBest(score_func=feature_selection.chi2,k=9)
fit = bf.fit(X_independent,y_target)
dfcolumns = pd.DataFrame(X_independent.columns)
dfscores = pd.DataFrame(fit.scores_)

# Concatenate two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns= ['Specs','Score']
print(featureScores.nlargest(9,'Score'))
print()

# Apply extratreesclassifier class to extract important features in order
X_independent=df.iloc[:,1:10]
y_target=df.iloc[:,-1]
model=ExtraTreesClassifier()
model.fit(X_independent, y_target)

# Plot graph of feature importances for better visualization
plt.clf()
feat_importances=pd.Series(model.feature_importances_,index=X_independent.columns)
feat_importances.nlargest(9).plot(kind='barh')
plt.show()

# Select feature
# As we run two feature selection library (SelectKBest, extratreesclassifier)
# We have confuse that those results are different
# But columns "Mistoses" have poor score in both
# So we are going to use all features without "ID" and "Mistoses"
data = df.iloc[:, 1:9]
data = data.astype(np.int32)

# Change Class value 2,4 to 0,1 for convenience
target = df.iloc[:, [-1]]
encoder = LabelEncoder()
target.loc[:,'Class'] = encoder.fit_transform(target.loc[:, 'Class'])

# define scaler and model list to use
# Scaler list use 3 kinds of scaler that we learn at DataScience class
scalers = [None, StandardScaler(),MinMaxScaler(), RobustScaler()]

# Model list use 4 kinds of models, each model define parameter
models = {DecisionTreeClassifier() : {
        'criterion' : ['entropy'],
        'max_depth': [None, 6, 9, 12],
        'min_samples_split': [0.01, 0.05, 0.1,1],
        'splitter': ['best', 'random'],
        'max_features': [ 'sqrt', 'log2']
        },
        DecisionTreeClassifier() : {
        'criterion' : ['gini'],
        'max_depth': [None, 6, 9, 12],
        'min_samples_split': [0.01, 0.05, 0.1,1],
        'splitter': ['best', 'random'],
        'max_features': ['sqrt', 'log2']
        },
        LogisticRegression() : {
        'penalty' : ['l1', 'l2', 'elasticnet',None],
        'solver' : ['liblinear', 'newton-cg', 'lbfgs'],
        'C' : [0.01, 0.1, 1 ],
        },
        SVC() : {
        'degree' : [1,2,3,4,5],
        'kernel' : ['linear','rbf','poly','sigmoid'],
        'C' : [0.01, 0.1, 1],
        'gamma':[0.0001, 0.001, 0.01, 0.1, 1]
        }}

# Make model function with scaling
'''
Parameters
scalers : list of scaler to use
models : list of model to use
x_train, x_test, y_train, y_test : train, test dataset to use
k : value about kfold k
'''
def Model_function(scalers,models,x_train, x_test, y_train, y_test, k):
    df_columns = ['Scaler', 'Model', 'k', 'Score', 'Parameter']
    result_list = []
    for i in scalers:
        if(i!=None):
            x_test=i.fit_transform(x_test)
            x_train=i.fit_transform(x_train)           
        for j in models:
            cv = KFold(k,shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=j, param_grid=dict(models[j].items()),  cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(x_train,y_train.values.ravel())

            # summarize results
            print("%s %s Best: %f using %s" % (i, j, grid_result.best_score_, grid_result.best_params_))
            score = grid_result.best_score_
            result_list.append([i,j,k,score, grid_result.best_params_])

    result = pd.DataFrame(result_list, columns=df_columns)
    return result

# Run
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=42, shuffle=True)

df_columns = ['Scaler', 'Model', 'k', 'Score', 'Parameter']
result_final = pd.DataFrame(columns=df_columns)

for i in [3, 5, 7]:
    print("\n\nResult with KFold k = ",i)
    result_retrun = Model_function(scalers,models,x_train, x_test, y_train, y_test, i)
    result_final = pd.concat([result_final,result_retrun],axis=0)

result_final = result_final.astype({"Score":"float64"})

# Adjusting for Data Frame Output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# print top 10 sore
print("\n\n")
print(result_final.nlargest(10,'Score'))
