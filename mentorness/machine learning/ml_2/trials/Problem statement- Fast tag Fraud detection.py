#%% md
# ## Fast Tag Fraud Detection
#%% md
# #### Problem Statement:
# 
# This internship project focuses on leveraging machine learning classification techniques to develop an effective fraud detection system for Fastag transactions. The dataset comprises key features such as transaction details, vehicle information, geographical location, and transaction amounts. The goal is to create a robust model that can accurately identify instances of fraudulent activity, ensuring the integrity and security of Fastag transactions.
#%% md
# ### Dataset Description:
# 
# 1. Transaction_ID: Unique identifier for each transaction.
# 2. Timestamp: Date and time of the transaction.
# 3. Vehicle_Type: Type of vehicle involved in the transaction.
# 4. FastagID: Unique identifier for Fastag.
# 5. TollBoothID: Identifier for the toll booth.
# 6. Lane_Type: Type of lane used for the transaction.
# 7. Vehicle_Dimensions: Dimensions of the vehicle.
# 8. Transaction_Amount: Amount associated with the transaction.
# 9. Amount_paid: Amount paid for the transaction.
# 10. Geographical_Location: Location details of the transaction.
# 11. Vehicle_Speed: Speed of the vehicle during the transaction.
# 12. Vehicle_Plate_Number: License plate number of the vehicle.
# 13. Fraud_indicator: Binary indicator of fraudulent activity (target variable).
#%%

#%%
import warnings
warnings.filterwarnings('ignore')
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
df_fraud = pd.read_csv("D:\\Internship\Mentorness MIP-ML-06\\Project 1- Fact tag Fraud\\dataset.csv")
#%%
df_fraud.head()
#%%
print('Share before drop duplicate value: ', df_fraud.shape)
df_fraud = df_fraud.drop_duplicates()
print('Share after drop duplicate value: ', df_fraud.shape)
#%%
df_fraud.describe()
#%%
df_fraud.info()
#%%
df_fraud.isnull().sum()
#%%
df_fraud.nunique()
#%%
df_fraud.columns
#%% md
# ### Data pre-processing
#%%
df_fraud['Timestamp'] = pd.to_datetime(df_fraud['Timestamp'])
#%%
df_fraud['Month Name'] = df_fraud['Timestamp'].dt.month_name()
#%%
df_fraud['Year'] = df_fraud['Timestamp'].dt.year
#%%
df_fraud['Day'] = df_fraud['Timestamp'].dt.day
#%%
df_fraud['Week in Year'] = df_fraud['Timestamp'].dt.isocalendar().week
#%%
df_fraud['Name of Day'] = df_fraud['Timestamp'].dt.day_name()
#%%
df_fraud['Hour'] = df_fraud['Timestamp'].dt.hour
#%%
df_fraud['Minutes'] = df_fraud['Timestamp'].dt.minute
#%%

#%%

#%%
df_fraud.info()
#%%
df_fraud.nunique()
#%%
df_fraud['Seasons'] = np.where((df_fraud['Month Name'] == 'January')|(df_fraud['Month Name'] == 'February')|(df_fraud['Month Name'] == 'December'),"Winter", 
np.where((df_fraud['Month Name'] == 'March')|(df_fraud['Month Name'] == 'April')|(df_fraud['Month Name'] == 'May'),"Spring", 
np.where((df_fraud['Month Name'] == 'June')|(df_fraud['Month Name'] == 'July')|(df_fraud['Month Name'] == 'August'),"Summer", 
np.where((df_fraud['Month Name'] == 'September')|(df_fraud['Month Name'] == 'October')|(df_fraud['Month Name'] == 'November'),"Autumn",0))))

#%%
df_fraud.head()
#%%
df_fraud.drop(columns = {'Transaction_ID','FastagID'},inplace = True)
#%%
df_fraud.columns
#%%
df_fraud.info()
#%%
df_fraud.nunique()
#%%
df_fraud.drop(columns = 'Timestamp',inplace = True)
#%%

#%%

#%%
num_col = df_fraud.select_dtypes(include = ['int','float','uint']).columns
num_col
#%%
cat_col = df_fraud.select_dtypes(include = ['object']).columns
cat_col
#%%
df_fraud.nunique()
#%%
df_fraud.describe()
#%%
df_fraud[num_col].nunique()
#%%
cat_col_new = ['Year','Day','Week in Year']
#%%
num_col = num_col.to_list()
#%%
num_col.remove('Year')
#%%
num_col.remove('Day')
#%%
num_col.remove('Week in Year')
#%%
num_col
#%%
cat_col = cat_col+cat_col_new
#%%
cat_col
#%%

#%% md
# ### Univeriate Analysis:-
#%% md
# #### Continuous Column
#%% md
# #### Dist Plot
#%%
for i in num_col:
    plt.figure(figsize = (5,5))
    sns.distplot(df_fraud[i])
#%% md
# #### KDE
#%%
for i in num_col:
    plt.figure(i)
    sns.kdeplot(df_fraud[i])
    plt.title("Density of {}".format(i))
#%% md
# ### BOX Plot
#%%
for i in num_col:
    plt.figure(i)
    sns.boxplot(df_fraud[i])
    plt.title("Density of {}".format(i))
#%%

#%%

#%% md
# #### Checking for Skewness
#%%
for i in num_col:
    print("The Skewness value of {} is {}".format(i, round(df_fraud[i].skew(),2)))
#%%

#%%

#%% md
# ### Vehicle_Plate_Number is Qualitative data
#%%
df_fraud.drop(columns = ['Vehicle_Plate_Number'], inplace  = True)
#%%
df_fraud.drop(columns = ['Year'], inplace  = True) #### In year column has the single value So it will not made any difference in model building 
#%%
cat_col.remove('Vehicle_Plate_Number')
#%%
cat_col.remove('Year')
#%% md
# #### Categorical Columns
#%%
for i in cat_col:
    plt.figure(i)
    ax = sns.countplot(data = df_fraud,x = i)
    
    for j in ax.containers:
        ax.bar_label(j)
        
        plt.xticks(rotation = 'vertical')
        
        plt.title('Density of {}'.format(i))
#%%

#%%

#%%
cat_col.remove('Fraud_indicator') ### Target Column
#%%

#%%
num_col
#%%
cat_col
#%%
target = ['Fraud_indicator']
#%%
cat_col
#%%
num_col
#%% md
# ### Biveriate Analysis
#%%

#%% md
# #### Continuous Column VS Categorical Column
# 
# * Box Plot
# * KDE Plot
#%%
for i in num_col:
    plt.figure()
    sns.boxplot(y = i,x = "Fraud_indicator", data = df_fraud)
#%%

#%%

#%%
for i in num_col:
    plt.figure()
    df_fraud[df_fraud['Fraud_indicator']== 'Fraud'][i].plot(kind = 'kde', label = 'Fraud')
    df_fraud[df_fraud['Fraud_indicator']== 'Not Fraud'][i].plot(kind = 'kde', label = 'Not Fraud')
    plt.legend()
    plt.title('Relationship between Fraud_indicator and {}'.format(i))
#%%

#%% md
# #### Categorical VS Categorical
#%%

for i in cat_col:
    ax = pd.crosstab(index = df_fraud[i], columns = df_fraud['Fraud_indicator'], normalize = 'index').plot.bar(color = ['red','green'])
    
    for j in ax.containers:
        ax.bar_label(j)
#%% md
# ### Creating Categorical Columns
#%%
df_fraud['Speed_Type'] = np.where(df_fraud['Vehicle_Speed']<=40,'Normal Speed',
np.where((df_fraud['Vehicle_Speed']>40)&(df_fraud['Vehicle_Speed']<=80),'High Speed',
np.where(df_fraud['Vehicle_Speed']>80,'Very High Speed',0)))

#%%
cat_col.append('Speed_Type')
#%% md
# ### Feature Selection
#%% md
# #### Statistical Feature Selection (Categorical Vs Continuous) using ANOVA tests
# 
# Analysis of variance(ANOVA) is performed to check if there is any relationship between the given continuous and categorical variable
# 
# * Assumption(H0): There is NO relation between the given variables (i.e. The average(mean) values of the numeric Target variable is same for all the groups in the categorical Predictor variable)
# * ANOVA Test result: Probability of H0 being true
#%%
def Anovatest(inpdata, targetvariable, predictor):
    
    from scipy.stats import f_oneway
    Selected_Col = []
    
    for i in predictor:
        groupdata = df.groupby(targetvariable)[i].apply(list)
        anova_result = f_oneway(*groupdata)
        
        
        if anova_result[1]<0.05:
            Selected_Col.append(i)
            print(i, 'is correlated with',targetvariable,'| P value is',anova_result[1])
        else:
            print(i, 'is not correlated with',targetvariable,'| P value is',anova_result[1])
    return(Selected_Col)
#%%
Anovatest(inpdata = df, targetvariable = target, predictor = num_col)
#%%

#%% md
# #### Statistical Feature Selection (Categorical Vs Categorical) using Chi-Square Test
# 
# Chi-Square tests is conducted to check the correlation between two categorical variables
# 
# * Assumption(H0): The two columns are NOT related to each other
# * Result of Chi-Sq Test: The Probability of H0 being True
#%%
def ChiSquare(inpdata,targetvariable,predictor):
    
    from scipy.stats import chi2_contingency
    
    Selected_Col = []
    
    for i in predictor:
        crosstabular = pd.crosstab(index = df[i], columns = df[targetvariable])
        chi_result = chi2_contingency(crosstabular)
        
        if chi_result[1]<0.05:
            Selected_Col.append(i)
            print(i, 'is correlated with',targetvariable,'| P value is',chi_result[1])
        else:
            print(i, 'is not correlated with',targetvariable,'| P value is',chi_result[1])
    return(Selected_Col)    
        
#%%
ChiSquare(inpdata = df, targetvariable = 'Fraud_indicator', predictor = cat_col)
#%%

#%% md
# ### Final Selected Columns are below
#%%
Final_Selected_Col = ['Transaction_Amount', 'Amount_paid','Vehicle_Type','TollBoothID','Lane_Type','Vehicle_Dimensions','Geographical_Location',
 'Month Name','Seasons','Week in Year']
#%%
Final_Selected_Col
#%%
df_fraud[Final_Selected_Col].head()
#%%

#%%
df_fraud[Final_Selected_Col].columns
#%%
DataforML = df[Final_Selected_Col]
#%%
DataforML['Fraud_indicator'] = df_fraud['Fraud_indicator']
#%%
DataforML.head()
#%%
DataforML.to_pickle('Fast_Tag.pkl')
#%%
DataforML = pd.read_pickle('Fast_Tag.pkl')
#%%
DataforML.head()
#%%

#%%
from sklearn.compose import ColumnTransformer
#%%
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
#%%
tarns = ColumnTransformer(transformers = [
    ('trans1',OneHotEncoder(sparse = False,drop = 'first'),['Vehicle_Type','TollBoothID','Lane_Type','Geographical_Location','Seasons']),
    ('trans2',OrdinalEncoder(categories = [['Large','Medium','Small'],['January','February','March','April','May','June','July','August','September','October','November','December']]),['Vehicle_Dimensions','Month Name'])
], remainder = 'passthrough')
#%%

#%%
X = DataforML.drop(columns = 'Fraud_indicator')
#%%
y = DataforML['Fraud_indicator']
#%%
X_transform = tarns.fit_transform(X)
#%%
X_transform.shape
#%%
le = LabelEncoder()
#%%
y_transform = le.fit_transform(y)
#%%

#%%
from sklearn.model_selection import train_test_split
#%%
X_train,X_test,y_train,y_test = train_test_split(X_transform,y_transform,test_size = 0.3, random_state = 45)
#%%

#%% md
# ## Model Parameter selection using GridSearchCV
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
#%%
from sklearn.model_selection import GridSearchCV
#%% md
# ### Logistic Regression
#%%
LR = LogisticRegression()
param_log = {
    'C':[1,3,5,7],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag','saga'],
    'penalty':['l1','l2']
}

GridsearCV_LR = GridSearchCV(LR,param_log,cv = 5,n_jobs = 1)
GridsearResult_LR = GridsearCV_LR.fit(X_train,y_train)
print('Best Parameter: ',GridsearResult_LR.best_params_)
print('Best Score: ',GridsearResult_LR.best_score_)
#%%

#%% md
# ### Decision Tree
#%%
DT = DecisionTreeClassifier()
param_DT = {
    'max_depth':[1,3,5,7],
    'criterion': ['gini', 'entropy']
}

GridsearCV_DT = GridSearchCV(DT,param_DT,cv = 5,n_jobs = 1)
GridsearResult_DT = GridsearCV_DT.fit(X_train,y_train)
print('Best Parameter: ',GridsearResult_DT.best_params_)
print('Best Score: ',GridsearResult_DT.best_score_)
#%%

#%% md
# ### AdaBoost
#%%
AB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 2))
param_AB = {
    'n_estimators':[100,200,300],
    'learning_rate': [0.1,0.01,0.001]
}

GridsearCV_AB = GridSearchCV(AB,param_AB,cv = 5,n_jobs = 1)
GridsearResult_AB = GridsearCV_AB.fit(X_train,y_train)
print('Best Parameter: ',GridsearResult_AB.best_params_)
print('Best Score: ',GridsearResult_AB.best_score_)
#%%

#%% md
# ### Random Forest
#%%
RF = RandomForestClassifier()
param_RF = {
    'max_depth':[2,3,5,7],
    'criterion':['gini','entropy'],
    'n_estimators': [100,300,200]
}

GridsearCV_RF = GridSearchCV(RF,param_RF,cv = 5,n_jobs = 1)
GridsearResult_RF = GridsearCV_RF.fit(X_train,y_train)
print('Best Parameter: ',GridsearResult_RF.best_params_)
print('Best Score: ',GridsearResult_RF.best_score_)
#%%

#%% md
# ### XGBoost Classifier
#%%
XGB = XGBClassifier()
param_XGB = {
    'max_depth':[2,3,5,7],
    'objective':['reg:linear'],
    'n_estimators': [100,300,200],
    'booster':['gbtree','gblinear'],
    'learning_rate':[0.01,0.001,0.1]
}

GridsearCV_XGB = GridSearchCV(XGB,param_XGB,cv = 5,n_jobs = 1)
GridsearResult_XGB = GridsearCV_XGB.fit(X_train,y_train)
print('Best Parameter: ',GridsearResult_XGB.best_params_)
print('Best Score: ',GridsearResult_XGB.best_score_)
#%%

#%% md
# ### KNN
#%%
KNN = KNeighborsClassifier()
param_KNN = {
    'n_neighbors':[2,3,5,7]
}

GridsearCV_KNN = GridSearchCV(KNN,param_KNN,cv = 5,n_jobs = 1)
GridsearResult_KNN = GridsearCV_KNN.fit(X_train,y_train)
print('Best Parameter: ',GridsearResult_KNN.best_params_)
print('Best Score: ',GridsearResult_KNN.best_score_)
#%%

#%% md
# ### SVM
#%%
SVM = svm.SVC(gamma = 'auto')
param_svm = {
    'C':[5,10,20,30],
    'kernel': ['rbf', 'linear','poly', 'sigmoid']
}

GridsearCV_svm = GridSearchCV(SVM,param_svm,cv = 5,n_jobs = 1)
GridsearResult_svm = GridsearCV_svm.fit(X_train,y_train)
print('Best Parameter: ',GridsearResult_svm.best_params_)
print('Best Score: ',GridsearResult_svm.best_score_)
#%%

#%% md
# ### Final Model Selection Process
#%%
model = {
    'LogisticRegression':LogisticRegression(C= 5, penalty= 'l2',solver= 'sag'),
    'DecisionTreeClassifier':DecisionTreeClassifier(criterion= 'entropy', max_depth= 7),
    'AdaBoost':AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 2),learning_rate= 0.1, n_estimators= 200),
    'RandomForest': RandomForestClassifier(criterion =  'gini', max_depth= 7, n_estimators= 300),
    'XGBoost': XGBClassifier(booster =  'gbtree', learning_rate= 0.1, max_depth= 7, n_estimators= 100, objective= 'reg:linear'),
    'KNN': KNeighborsClassifier(n_neighbors =  2),
    'SVM': svm.SVC(gamma = 'auto',C= 5, kernel= 'poly')
}
#%%

#%%
for i in model:
    print(i)
    clf = model[i].fit(X_train,y_train)
    
    prediction = clf.predict(X_test)
    TestCase = pd.DataFrame()
    TestCase['Actual'] = y_test
    TestCase['Prediction'] = prediction
    
    print(TestCase.head(10))
    
    from sklearn import metrics
    
    print(metrics.classification_report(y_test,prediction))
    print(metrics.confusion_matrix(y_test,prediction))
    
    
    F1_score = metrics.f1_score(y_test,prediction, average = 'weighted')
    print('Test Score:- ',F1_score)
    
    from sklearn.model_selection import cross_val_score
    FinalAccuracyScore = cross_val_score(clf, X_transform, y_transform, cv = 5, scoring = 'f1_weighted')
    
    print('\nAccuracy values for 5-fold Cross Validation:\n',FinalAccuracyScore)
    print('\nFinal Average Accuracy of the model:', round(FinalAccuracyScore.mean(),2))
    print()
#%% md
# * After checking the accuracy score selected model is SVM. 
#%% md
# #### Machine Learning Pipeline creation
#%%
from sklearn.pipeline import make_pipeline
#%%
column_trans = ColumnTransformer(transformers = [
    ('trans1',OneHotEncoder(sparse = False,drop = 'first'),['Vehicle_Type','TollBoothID','Lane_Type','Geographical_Location','Seasons']),
    ('trans2',OrdinalEncoder(categories = [['Large','Medium','Small'],['January','February','March','April','May','June','July','August','September','October','November','December']]),['Vehicle_Dimensions','Month Name'])
], remainder = 'passthrough')
#%%
SVM = svm.SVC(gamma = 'auto',C= 5, kernel= 'poly')
#%%

#%%
pipe = make_pipeline(column_trans,SVM)
#%%
pipe.fit(X,y)
#%%
y_pred = pipe.predict(X)
#%%
y_pred
#%%

#%%
import pickle

pickle.dump(pipe,open('SVM_Fast.pkl','wb'))
#%%

#%%
pipe.predict(X)
#%% md
# ### Checking the predictive analysis
#%%
pipe.predict(pd.DataFrame([[150,100,'Car','B-102','Regular','Small','13.059816123454882, 77.77068662374292','January','Winter',1]],columns = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'TollBoothID',
       'Lane_Type', 'Vehicle_Dimensions', 'Geographical_Location',
       'Month Name', 'Seasons', 'Week in Year']))
#%%

#%%

#%% md
# ### Model Deployment
#%%
def ModelDeployment(InputData):
    import pandas as pd
    predictor = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'TollBoothID',
       'Lane_Type', 'Vehicle_Dimensions', 'Geographical_Location',
       'Month Name', 'Seasons', 'Week in Year']
    
    X = InputData[predictor]
    
    import pickle
    
    with open('SVM_Fast.pkl','rb') as readFile:
        predictive_model = pickle.load(readFile)
        readFile.close()
        
    prediction=  predictive_model.predict(X)
    predictionresult = pd.DataFrame(data = prediction, columns = ['Status'])
    
    return(predictionresult)
#%%

#%%
ModelDeployment(InputData = DataforML)
#%%

#%% md
# ### Conclusion:- 
#%% md
# Here I have used SVM as it has given 100% accuracy.
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
