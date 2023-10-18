#Are the best and brightest GA students staying in GA?
# import required packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodel.formula.api as smf
import statsmodels as sm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn_pandas import DataFrameMapper

#import data for modeling & key variables
positions = ['DB','DL','LB','OL','QB','RB','TE','WR']
df_main = pd.read_csv('NFL_num.csv')
df_QB = pd.read_csv('NFL_QB.csv')
full_model = {} #Dictionary with keys as positions and values as 1) MSE and 2) MAE for that position
lasso_model = {} #Dictionary with Lasso stats (not currently used)
impact_model = {} #Dictionary with the OLS with only lasso variables
lasso_coef = {} #results from lasso
svr_rbf = {} #svr w/ Lasso
svr_linear = {} #linear w/ Lasso
svr_poly = {} #poly w/ Lasso

def pcaClassifier(df, position):
    df = df.drop(['Years_Played', 'General_Position'], axis= 1)
    scaler = StandardScaler()
    scaler.fit(df)
    #scaler.set_output(transform= "pandas")
    df_scaled = scaler.transform(df)
    pca = PCA(n_components = 2, svd_solver = 'full')
    pca.fit(df_scaled)
    pca_result = pca.transform(df_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    plt.figure(figsize=(10,10))
    #df_col = list(pca_result.columns)
    plt.scatter(pca_result[:,0],pca_result[:,1], cmap='plasma')
    #plt.xlabel(df_col[0])
    #plt.ylabel(df_col[1])
    plt.title('PCA with 2 Components.png')
    plt.figtext(0.99, 0.01, f'Explained Variance: {explained_variance_ratio}, Singular Values: {singular_values}', horizontalalignment='right')
   # plt.savefig('PCA with 2 Components.png')
    return pca_result

teso1 = pcaClassifier(df_QB, 'QB')

