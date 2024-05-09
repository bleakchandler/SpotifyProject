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

# Assuming df is your DataFrame and you've already prepared your data
X = df[['danceability', 'energy', 'loudness']]  # Predictor variables
y = df['track_popularity']  # Response variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model. Creates an instance of the Linear Regression model. Will attempt to find a linear relationship between the features (x) and target (y).
model = LinearRegression()

# Fit model. Calculates the coefficients for the features in X_train that result in the best prediction of y_train.
model.fit(X_train, y_train)

# Predict on the test data. Predict the values of y_test using X_test.
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
#  The MSE is a common measure to evaluate the accuracy of regression models, representing the average squared difference between actual and predicted values.


# A MSE of 593.54 suggests that, on average, the predictions of the model are about 593.54 points off from the actual popularity scores when squared. Helps understand how well the model is predicting track popularity based on features like danceability, energy, and loudness.
# A lower MSE indicates that the model's predictions are very close to the actual results, meaning the model has high predictive accuracy.
# A higher MSE suggests that the predictions are often far off from the actual values, indicating lower accuracy.
