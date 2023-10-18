# import required packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodel.formula.api as smf
import statsmodels as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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

###################### Function to Scale Data but Maintain the DataFrame Structure ##############################

def scale_df(df, position):
    if position == 'QB':
        column_mapping = [ ('General_Position',None),
                            ('Round',StandardScaler()),
                            ('Pick',StandardScaler()),
                            ('Games_Played',StandardScaler()),
                            ('Years_Played',LabelEncoder()),
                            ('Age',StandardScaler()),
                            ('Height',StandardScaler()),
                            ('Weight',StandardScaler()),
                            ('BMI',StandardScaler()),
                            ('Arm_Length',StandardScaler()),
                            ('Hand_Size',StandardScaler()),
                            ('Wingspan',StandardScaler()),
                            ('x40_Yard_Dash',StandardScaler()),
                            ('x20_Yard_Split',StandardScaler()),
                            ('x10_Yard_Split',StandardScaler()),
                            ('Vertical_Leap',StandardScaler()),
                            ('Broad_Jump',StandardScaler()),
                            ('x20_Yd_Shuttle',StandardScaler()),
                            ('Three_Cone',StandardScaler()),
                            ('wAV',StandardScaler()),
                            ('DrAV',StandardScaler()),
                            ('HOF',StandardScaler())]
    else:
        column_mapping = [  ('General_Position',None),
                            ('Round',StandardScaler()),
                            ('Pick',StandardScaler()),
                            ('Games_Played',StandardScaler()),
                            ('Years_Played',LabelEncoder()),
                            ('Age',StandardScaler()),
                            ('Height',StandardScaler()),
                            ('Weight',StandardScaler()),
                            ('BMI',StandardScaler()),
                            ('Arm_Length',StandardScaler()),
                            ('Hand_Size',StandardScaler()),
                            ('Wingspan',StandardScaler()),
                            ('x40_Yard_Dash',StandardScaler()),
                            ('x20_Yard_Split',StandardScaler()),
                            ('x10_Yard_Split',StandardScaler()),
                            ('Bench_Press',StandardScaler()),
                            ('Vertical_Leap',StandardScaler()),
                            ('Broad_Jump',StandardScaler()),
                            ('x20_Yd_Shuttle',StandardScaler()),
                            ('Three_Cone',StandardScaler()),
                            ('Shuttle_Split',StandardScaler()),
                            ('Four_Square',StandardScaler()),
                            ('wAV',StandardScaler()),
                            ('DrAV',StandardScaler()),
                            ('HOF',StandardScaler())]
    mapper = DataFrameMapper(column_mapping, df_out = True)
    processed = mapper.fit_transform(df)
    return processed
    
################ Implementing full feature linear regression ##############################

def full_features(data, position, partition = True): #returns a pd with the estimated coefficients
    if partition:
        df = data[data['General_Position'] == position]
    else:
        df = data
    df = df.dropna(axis = 1, how = 'all')
    df = df.dropna(how = 'any')
    df = scale_df(df, position)
    y = df['Years_Played']
    X = df.drop(columns = ['Years_Played', 'General_Position'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7641)
    model = LinearRegression() #model created
    model.fit(X_train,y_train) #model trained
    yhat = model.predict(X_test) #create yhat
    MSE = mean_squared_error(y_test, yhat) #mean squared error of full model
    MAE = mean_absolute_error(y_test, yhat) #mean absolute error of full model
    R2 = r2_score(y_test, yhat)
    full_model[position] = {}
    full_model[position]['All D MSE'] = MSE
    full_model[position]['All D MAE'] = MAE
    full_model[position]['All D R2'] = R2
    full_coefficients = pd.DataFrame()
    full_coefficients["Columns"] = X_train.columns
    full_coefficients["Coefficient Estimate"] = pd.Series(model.coef_)
    return full_coefficients
    

################ Lasso Regression for feature selection ##############################

def get_alpha(X, y, folds):
    alpha = LassoCV(cv = folds, random_state = 0).fit(X, y)
    return alpha.score(X, y)

def lasso(data, position, partition = True):
    if partition:
        df = data[data['General_Position'] == position]
    else:
        df = data
    df = df.dropna(axis = 1, how = 'all')
    df = df.dropna(how = 'any')
    df = scale_df(df, position)
    X = df.drop(['Years_Played', 'General_Position'], axis= 1)
    y = df['Years_Played']
    alpha = get_alpha(X, y, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7641)
    lasso = Lasso(alpha = alpha, normalize = False)
    lasso.fit(X_train, y_train)
    yhat = lasso.predict(X_test)
    MSE = mean_squared_error(y_test, yhat)
    MAE = mean_absolute_error(y_test, yhat)
    lasso_model[position] = {}
    lasso_model[position]['Lasso MSE'] = MSE
    lasso_model[position]['Lasso MAE'] = MAE
    Lasso_coefficients = pd.DataFrame()
    Lasso_coefficients["Columns"] = X_train.columns
    Lasso_coefficients["Coefficient Estimate"] = pd.Series(lasso.coef_)
    return Lasso_coefficients

def impact_features(df, position):
    feature_dict = df.set_index('Columns')['Coefficient Estimate'].to_dict()
    lasso_coef[position] = feature_dict
    features = []
    for key, value in feature_dict.items():
        if abs(float(value)) > 0:
            features.append(key)
        else:
            continue
    color =['tab:gray', 'tab:blue', 'tab:orange',
        'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
        'tab:orange', 'tab:green', 'tab:blue', 'tab:olive']
    fig, ax = plt.subplots(figsize =(20, 10))
    ax.bar(feature_dict["Columns"],
    feature_dict['Coefficient Estimate'],
    color = color)
    ax.spines['bottom'].set_position('zero')
    plt.style.use('ggplot')
    plt.title("Coefficients After Lasso Regression")
    plt.savefig('Lasso Coef Graph.png')
    return features

################################## Model using results from Lasso ############################################################

def impact_modeling(data, position, features, partition = True): #returns a pd with the estimated coefficients
    if partition:
        df = data[data['General_Position'] == position]
    else:
        df = data
    df = df.dropna(axis = 1, how = 'all')
    df = df.dropna(how = 'any')
    df = scale_df(df, position)
    X = df.drop(['Years_Played', 'General_Position'], axis= 1)
    X = df[features]
    y = df['Years_Played']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7641)
    model = LinearRegression() #model created
    model.fit(X_train,y_train) #model trained
    yhat = model.predict(X_test) #create yhat
    MSE = mean_squared_error(y_test, yhat) #mean squared error of full model
    MAE = mean_absolute_error(y_test, yhat) #mean absolute error of full model
    R2 = r2_score(y_test, yhat)
    impact_model[position] = {}
    impact_model[position]['Select D MSE'] = MSE
    impact_model[position]['Select D MAE'] = MAE
    impact_model[position]['Select D R2'] = R2
    impact_coefficients = pd.DataFrame()
    impact_coefficients["Columns"] = X_train.columns
    impact_coefficients["Coefficient Estimate"] = pd.Series(model.coef_)
    return impact_coefficients

############################# Support Vector Regression #########################################################

def svr_model(data, position, features, svr_kernel, partition = True): #returns a pd with the estimated coefficients
    if partition:
        df = data[data['General_Position'] == position]
    else:
        df = data
    df = df.dropna(axis = 1, how = 'all')
    df = df.dropna(how = 'any')
    df = scale_df(df, position)
    X = df.drop(['Years_Played', 'General_Position'], axis= 1)
    X = df[features]
    y = df['Years_Played']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7641)
    model = SVR(kernel= svr_kernel) #model created
    model.fit(X_train,y_train) #model trained
    yhat = model.predict(X_test) #create yhat
    MSE = mean_squared_error(y_test, yhat) #mean squared error of full model
    MAE = mean_absolute_error(y_test, yhat) #mean absolute error of full model
    R2 = r2_score(y_test, yhat)
    plt.scatter(X, y, color='#B3A369',
    label='Y Actual')
    plt.plot(X, yhat, color='#003057',
    label='Y Predicted')
    plt.title(f"SVR with {svr_kernel} Kernel")
    plt.legend()
    plt.savefig(f'{svr_kernel}.png')
    if svr_kernel == 'rbf':
        dictionary = svr_rbf
    if svr_kernel == 'linear':
        dictionary = svr_linear
    if svr_kernel == 'poly':
        dictionary = svr_poly
    dictionary[position] = {}
    dictionary[position]['Select D MSE'] = MSE
    dictionary[position]['Select D MAE'] = MAE
    dictionary[position]['Select D R2'] = R2

################# Launch Codes ################################################
for i in range(len(positions)):
    target = positions[i]
    partition = True
    if target == 'QB':
        data = df_QB
    else:
        data = df_main
    full = full_features(data, target, partition)
    lasso_co = lasso(data, target, partition)
    lasso_features = impact_features(lasso_co, target)
    lasso_optimal = impact_modeling(data, target, lasso_features, partition)
    pca_features = 'pending' ### Add something here ####
    svr_linear = svr_model(data, target, lasso_features, 'linear', partition)
    svr_poly = svr_model(data, target, lasso_features, 'poly', partition)
    svr_rbf = svr_model(data, target, lasso_features, 'rbf', partition)

################# PCA 2  ################################################
def pcaClassifier(df, position):
    df = scale_df(df, position)
    df = df.drop(['Years_Played', 'General_Position'], axis= 1)
    pca = PCA(n_components = 2, svd_solver = 'full')
    pca.fit(df)
    pca_result = pca.transform(df)
    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    plt.figure(figsize=(10,10))
    df_col = list(pca_result.columns)
    plt.scatter(x[:,0],x[:,1], c=df['Years_Played'], cmap='plasma')
    #plt.xlabel(df_col[0])
    #plt.ylabel(df_col[1])
    plt.title('PCA with 2 Components.png')
    plt.figtext(0.99, 0.01, f'Explained Variance: {explained_variance_ratio}, Singular Values: {singular_values}', horizontalalignment='right')
    plt.savefig('PCA with 2 Components.png')
    return pca_result


################## For tables creation ################################
full_df = pd.DataFrame.from_dict(full_model)
full_df = full_df.transpose()
select_df = pd.DataFrame.from_dict(impact_model)
select_df = select_df.transpose()
models_scoring = pd.concat([full_df, select_df], axis = 1)
lasso_df = pd.DataFrame.from_dict(lasso_coef)

############## Tables to CSV ############################
#lasso_df.to_csv('lasso_coef.csv')
#models_scoring.to_csv('model_scoring.csv')

'''
################# PCA 2  ################################################
def pcaClassifier(x):
    scalar = StandardScaler()
    scaled_data = pd.DataFrame(scalar.fit_transform(x))
    pca = PCA(n_components = 2, svd_solver = 'full')
    pca.fit(scaled_data)
    x = pca.transform(scaled_data)
    return pca, x
'''