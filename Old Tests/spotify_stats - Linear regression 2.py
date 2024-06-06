import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('spotify_songs.csv')

df['track_name'] = df['track_name'].replace(r'\$', '\\$', regex=True)
#Adds a backslash before dollar signs to treat as regular character because LaTeX was trying to interpret 
df['track_name'] = df['track_name'].str.replace(r'\u202C|\u202D|\u202E|\u200E|\u200F|\u202A|\u202B', '', regex=True)
#Remove special characters from data set that were affecting matplotlib

font_path = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
#default font was causing issues with matlib with rendering certain unicode characters, so font was replaced

X = df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo']] # Predictor variables
y = df['track_popularity']  # Response variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Split Data into training and testing sets

model = LinearRegression()
model.fit(X_train, y_train)
#initialize and fit th emodel

predictions = model.predict(X_test)
#make predictions

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
#evaluate model

plt.figure(figsize=(10, 6))

#Plot actual vs. predicted
# Scatter plot for actual values
plt.scatter(X_test['energy'], y_test, color='blue', label='Actual values')

# Line plot for predicted values
sorted_idx = np.argsort(X_test['energy'])
plt.plot(X_test['energy'].values[sorted_idx], predictions[sorted_idx], color='red', label='Predicted values')

plt.title('Actual vs Predicted Popularity by Energy')
plt.xlabel('Energy')
plt.ylabel('Popularity')
plt.legend()
plt.grid(True)
plt.show()


#Linear regression is used to model the relationship between a dependent variable (target) and one or more independent variables (features)

