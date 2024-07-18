#%%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import pickle

#%%
# Load the dataset
df = pd.read_csv("FastagFraudDetection.csv")

#%% md
# # 1.EXPLORATORY DATA ANALYSIS(EDA)
#%%
df.nunique()
#%%
df.head()
#%%
df.info()
#%%
df.describe()
#%%
df.dtypes
#%%
# Check for missing values
print(df.isnull().sum())
#%%
# Drop duplicates and missing values
df = df.drop_duplicates().dropna()
#%%
# Analyze feature distributions (histograms)
df.hist(figsize=(12, 8))
plt.show()
#%% md
# **Investigate correlations with the target variable (fraud indicator)**
#%%
# Convert date/time columns to numeric representation before calculating correlations
df['Timestamp'] = pd.to_datetime(df['Timestamp']) # Convert to datetime
df['Date_Column_Numeric'] = df['Timestamp'].astype(int)  # Convert to numeric timestamp

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr(method="spearman")  # Now calculate correlations

print(correlation_matrix)
#%% md
# **Visualize categorical feature relationships (boxplots)**
#%%
df.boxplot(by="Vehicle_Type", column="Transaction_Amount")
plt.show()
#%%
numerical_features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed']
df[numerical_features].hist(bins=30, figsize=(10, 7))
plt.show()
#%% md
# **Box plots for numerical features**
#%%
plt.figure(figsize=(10, 7))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=feature, data=df)
plt.show()
#%% md
# **Bar plots for categorical features**
#%%
categorical_features = ['Vehicle_Type', 'Lane_Type', 'TollBoothID']
plt.figure(figsize=(15, 7))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(1, 3, i)
    df[feature].value_counts().plot(kind='bar')
    plt.title(feature)
plt.show()
#%% md
# **Count of Fraud and Non_Fraud Indicators bold text**
#%%
sns.countplot(x='Fraud_indicator', data=df, palette=['red', 'green'])
plt.xlabel('Fraud Indicator')
plt.ylabel('Count')
plt.title('Count of Fraud and Non-Fraud Indicators')
plt.show()
#%%
# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
#%% md
# **Distribution of Transaction Amount by Fraud Indicator**
#%%

sns.boxplot(
    x = "Fraud_indicator",
    y = "Transaction_Amount",
    showmeans=True,
    data=df,
    palette=["red", "green"]
)

plt.xlabel("Fraud Indicator")
plt.ylabel("Transaction Amount")
plt.title("Distribution of Transaction Amount by Fraud Indicator")
plt.xticks(rotation=45)
plt.show()
#%% md
# **Uniqueness**
#%%
print("\nUnique values in categorical columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()}")
sns.pairplot(df)
plt.show()
#%% md
# # 2.Feature Engineering:
#%%
# Extract time-based features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Month'] = df['Timestamp'].dt.month
df['Week'] = df['Timestamp'].dt.isocalendar().week
#%%
# Drop unnecessary columns
df = df.drop(columns=['Transaction_ID', 'FastagID', 'Vehicle_Plate_Number', 'Timestamp'])

#%%
# Define feature columns
features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'TollBoothID',
            'Lane_Type', 'Vehicle_Dimensions', 'Geographical_Location',
            'Month', 'Week']
#%%
# Encode target variable
le = LabelEncoder()
df['Fraud_indicator'] = le.fit_transform(df['Fraud_indicator'])

#%%
# Define X and y
X = df[features]
y = df['Fraud_indicator']

#%%
# Define ColumnTransformer
column_trans = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(drop='first'), ['Vehicle_Type', 'TollBoothID', 'Lane_Type', 'Geographical_Location']),
    ('ordinal', OrdinalEncoder(), ['Vehicle_Dimensions'])
], remainder='passthrough')

#%%
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

#%%
df.columns
#%% md
# # 3.Model Development
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, \
    precision_score, recall_score
from sklearn.model_selection import GridSearchCV
#%%
# Transform the features
X_train_transformed = column_trans.fit_transform(X_train)
X_test_transformed = column_trans.transform(X_test)
#%% md
# **Define models and their hyperparameters**
#%%
models = {
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'params': {'max_depth': [1, 3, 5, 7], 'criterion': ['gini', 'entropy']}
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {'max_depth': [2, 3, 5, 7], 'criterion': ['gini', 'entropy'], 'n_estimators': [100, 200, 300]}
    },
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier(),
        'params': {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.01, 0.001]}
    },
    'XGBClassifier': {
        'model': XGBClassifier(),
        'params': {'max_depth': [2, 3, 5, 7], 'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.001]}
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors': [2, 3, 5, 7]}
    },
    'SVM': {
        'model': svm.SVC(probability=True),
        'params': {'C': [5, 10, 20, 30], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
    }
}
#%%
# Evaluate each model using GridSearchCV
best_models = {}
for model_name in models:
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(models[model_name]['model'], models[model_name]['params'], cv=5, n_jobs=-1, scoring='f1_weighted')
    grid_search.fit(X_train_transformed, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best Cross-validation score for {model_name}: {grid_search.best_score_}")

#%% md
# **Model Evaluation**
#%%

for model_name, model in best_models.items():
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test_transformed)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_transformed)[:, 1])
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')
    print(f'ROC AUC Score: {roc_auc}')
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_transformed)[:, 1])
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')

#%%
import pandas as pd

# Initialize an empty list to store the results
results = []

# Iterate over the best_models dictionary
for model_name, model in best_models.items():
    # Get the best cross-validation score
    best_score = cross_val_score(model, X_train_transformed, y_train, cv=5).mean()
    # Get the best parameters
    best_params = model.get_params()
    # Append the results to the list
    results.append({
        'Model': model_name,
        'Best Parameters': best_params,
        'Best Score': best_score
    })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)
#%%
import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(10, 5))

# Create a bar plot of the best scores
plt.bar(results_df['Model'], results_df['Best Score'])

# Add labels and title
plt.xlabel('Model')
plt.ylabel('Best Cross-validation Score')
plt.title('Model Performance')
# Tilt the x-axis labels
plt.xticks(rotation=45)

# Display the plot
plt.show()
#%%
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
#%%
# Visualization: Predicted vs Actual
best_model = best_models['SVM']
y_pred_final = best_model.predict(X_test_transformed)

plt.figure(figsize=(10, 6))
sns.countplot(x=y_test, label='Actual', color='blue', alpha=0.6)
sns.countplot(x=y_pred_final, label='Predicted', color='red', alpha=0.4)
plt.legend()
plt.title('Actual vs Predicted Fraud Indicators')
plt.show()
#%%
# Iterate over each model in the best_models dictionary
for model_name, model in best_models.items():
    # Generate predictions
    y_pred_final = model.predict(X_test_transformed)

    # Create a new figure for each model
    plt.figure(figsize=(10, 6))

    # Create countplots for actual and predicted values
    sns.countplot(x=y_test, label='Actual', color='blue', alpha=0.6)
    sns.countplot(x=y_pred_final, label='Predicted', color='red', alpha=0.4)

    # Add legend and title
    plt.legend()
    plt.title(f'Actual vs Predicted Fraud Indicators for {model_name}')

    # Display the plot
    plt.show()
#%%
# Define the model
model = svm.SVC(gamma='auto', C=5, kernel='poly')

#%%
# Create pipeline
pipe = make_pipeline(column_trans, model)
#%%
# Fit the model
pipe.fit(X_train, y_train)
#%%
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Predict the test set results
y_pred = pipe.predict(X_test)

# Create a DataFrame from the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index = ['Actual Negative', 'Actual Positive'], 
                     columns = ['Predicted Negative', 'Predicted Positive'])

# Print the confusion matrix in tabular format
print("Confusion Matrix:")
print(cm_df)

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
#%%
# Save the model
with open('SVM_Fast.pkl', 'wb') as f:
    pickle.dump(pipe, f)
#%%
# Load the model and make a prediction
def predict_fraud(input_data):
    with open('SVM_Fast.pkl', 'rb') as f:
        model = pickle.load(f)
    return model.predict(input_data)
#%%
# Example prediction
example_data = pd.DataFrame([[150, 100, 'Car', 'B-102', 'Regular', 'Small', '13.059816123454882, 77.77068662374292', 1, 1]],
                            columns=features)
#%%
print(predict_fraud(example_data))
#%%
# Visualize the data
sns.pairplot(df, hue='Fraud_indicator')
plt.show()

#%%
import numpy as np

# Select only numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Correlation heatmap
sns.heatmap(df_numeric.corr(), annot=True)
plt.show()
#%%
# Cross-validation
scores = cross_val_score(pipe, X, y, cv=5)
print(scores)

#%%
# Hyperparameter tuning
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'svc__gamma': ['scale', 'auto']
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X, y)
print(grid.best_params_)
print(grid.best_score_)

#%%
# Save the best model
with open('SVM_Fast_best.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
#%% md
# # 4.Model Deployment
#%%
# Load the best model and make a prediction
def predict_fraud_best(input_data):
    with open('SVM_Fast_best.pkl', 'rb') as f:
        model = pickle.load(f)
    return model.predict(input_data)
#%%
import numpy as np

# Transform the test data
X_test_transformed = column_trans.transform(X_test)

# Make predictions
y_pred_prob = model.predict(X_test_transformed)

# Convert probabilities to binary predictions
y_pred = np.round(y_pred_prob)

# Print accuracy metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print accuracy metrics
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
#%%
#Accuracy metrics
import matplotlib.pyplot as plt


metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Score')
plt.title('Model Metrics')
plt.ylim(0, 1)  
plt.show()
#%%
# Create a sample input where Fraud_indicator is 1
sample_data = pd.DataFrame([[200, 150, 'Truck', 'C-103', 'Regular', 'Large', '13.059816123454882, 77.77068662374292', 9, 1]],
                            columns=features)
#%%
print(sample_data)
#%%
# Predict the sample data
print(predict_fraud(sample_data))

#%%
# Load the dataset
testing = pd.read_csv("FastagFraudDetection.csv")

#%%

# Preprocess the data in the same way as it was done while training
testing = testing.drop_duplicates().dropna()
testing['Timestamp'] = pd.to_datetime(testing['Timestamp'])
testing['Month'] = testing['Timestamp'].dt.month
testing['Week'] = testing['Timestamp'].dt.isocalendar().week
testing = testing.drop(columns=['Transaction_ID', 'FastagID', 'Vehicle_Plate_Number', 'Timestamp'])

#%%

# Define the feature columns
features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'TollBoothID',
            'Lane_Type', 'Vehicle_Dimensions', 'Geographical_Location',
            'Month', 'Week']

# Define X
X = testing[features]


#%%
# Load the model
with open('SVM_Fast_best.pkl', 'rb') as f:
    model = pickle.load(f)


#%%
# Make predictions
predictions = model.predict(X)

#%%

# Print whether each transaction is a fraud or not
for i, prediction in enumerate(predictions):
    print(f'Transaction {i+1} is {"a fraud" if prediction == 1 else "not a fraud"}')
#%%
# Save the predictions to a CSV file
testing['Fraud_indicator'] = predictions

testing.to_csv('predictions.csv', index=False)
#%%
# Load the original dataset
original_df = pd.read_csv("FastagFraudDetection.csv")

#%%
# Preprocess the original dataset in the same way as it was done while training
original_df = original_df.drop_duplicates().dropna()
original_df['Timestamp'] = pd.to_datetime(original_df['Timestamp'])
original_df['Month'] = original_df['Timestamp'].dt.month
original_df['Week'] = original_df['Timestamp'].dt.isocalendar().week
#%%
original_df = original_df.drop(columns=['Transaction_ID', 'FastagID', 'Vehicle_Plate_Number', 'Timestamp'])

#%%
# Load the predicted results
predicted_df = pd.read_csv('predictions.csv')
#%%
# Compare the 'Fraud_indicator' column in the original dataset with the 'Fraud_indicator' column in the predicted results
comparison_df = pd.DataFrame({
    'Actual': original_df['Fraud_indicator'],
    'Predicted': predicted_df['Fraud_indicator']
})
#%%
# Display the comparison DataFrame
print(comparison_df.head(50))
#%%
# Calculate the accuracy of the model
accuracy = (comparison_df['Actual'] == comparison_df['Predicted']).mean()
print(f'Accuracy: {accuracy:.2f}')

#%%
from sklearn.metrics import confusion_matrix as cm

# Map 'Fraud' to 1 and 'Not Fraud' to 0 in 'Actual' column
comparison_df['Actual'] = comparison_df['Actual'].map({'Fraud': 1, 'Not Fraud': 0})

# Calculate the confusion matrix
confusion_mat = cm(comparison_df['Actual'], comparison_df['Predicted'])
cm_df = pd.DataFrame(confusion_mat, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print(cm_df)
#%%
print(original_df['Vehicle_Type'].unique())
#%% md
# # 5. Explanatory Analysis
#%% md
# **Feature Importance Using SHAP Values**
#%%
import shap
import pickle

# Load the best model
with open('SVM_Fast_best.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare the data
X_transformed = model.named_steps['columntransformer'].transform(X)

# Use a sample of the data for the SHAP KernelExplainer initialization
# Typically, 100 samples are sufficient; adjust this based on your data size
sample_size = min(100, X_transformed.shape[0])
X_sample = shap.sample(X_transformed, sample_size, random_state=42)

# Initialize the SHAP KernelExplainer
explainer = shap.KernelExplainer(model.named_steps['svc'].predict, X_sample)

# Calculate SHAP values
shap_values = explainer.shap_values(X_transformed)

# Plot feature importance
shap.summary_plot(shap_values, X_transformed, feature_names=features)

#%%
print("Length of SHAP values: ", len(shap_values[0]))
print("Number of features in transformed data: ", X_transformed.shape[1])
print("Number of feature names: ", len(features))
#%%
# Get feature names after transformation
feature_names_after_transform = column_trans.get_feature_names_out()

# Plot feature importance
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names_after_transform)
#%% md
# **Correlation Analysis**
#%%
# Get feature names after transformation
feature_names_after_transform = column_trans.get_feature_names_out()
# Convert the transformed data back to a DataFrame if necessary
import pandas as pd

# Convert the transformed data back to a DataFrame
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names_after_transform)

# Calculate the correlation matrix
correlation_matrix = X_transformed_df.corr()

# Plot the heatmap`
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()
#%% md
# **Pairplot Analysis**
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("FastagFraudDetection.csv")

# Preprocess the data
df = df.drop_duplicates().dropna()
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Month'] = df['Timestamp'].dt.month
df['Week'] = df['Timestamp'].dt.isocalendar().week
df = df.drop(columns=['Transaction_ID', 'FastagID', 'Vehicle_Plate_Number', 'Timestamp'])

# Pairplot analysis
sns.pairplot(df, hue='Fraud_indicator')
plt.show()

#%%
