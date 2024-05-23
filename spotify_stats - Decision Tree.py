import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('spotify_songs.csv')

# Replace special characters
df['track_name'] = df['track_name'].replace(r'\$', '\\$', regex=True)
df['track_name'] = df['track_name'].str.replace(r'\u202C|\u202D|\u202E|\u200E|\u200F|\u202A|\u202B', '', regex=True)

# Create categorical variables
def create_categories(row, col):
    if row[col] < 0.33:
        return 'Low'
    elif row[col] < 0.66:
        return 'Medium'
    else:
        return 'High'

for col in ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo']:
    df[col + '_category'] = df.apply(lambda row: create_categories(row, col), axis=1)

# Define target variable and make sure it has the required categories
def categorize_popularity(popularity):
    if popularity < df['track_popularity'].quantile(0.33):
        return 'Low'
    elif popularity < df['track_popularity'].quantile(0.66):
        return 'Medium'
    else:
        return 'High'

df['track_popularity_category'] = df['track_popularity'].apply(categorize_popularity)

# Make sure the target variable has the correct categories
assert df['track_popularity_category'].isin(['Low', 'Medium', 'High']).all()

# Define predictors and response variable
X = df[['danceability_category', 'energy_category', 'loudness_category', 'speechiness_category', 
        'acousticness_category', 'instrumentalness_category', 'liveness_category', 
        'valence_category', 'tempo_category']]
y = df['track_popularity_category']

# Convert categorical features to dummy variables
X = pd.get_dummies(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the decision tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['Low', 'Medium', 'High'], filled=True, rounded=True)
plt.show()