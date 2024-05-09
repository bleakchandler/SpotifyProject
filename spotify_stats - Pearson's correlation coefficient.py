import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

df = pd.read_csv('spotify_songs.csv')

df['track_name'] = df['track_name'].replace(r'\$', '\\$', regex=True)
#Adds a backslash before dollar signs to treat as regular character because LaTeX was trying to interpret 
df['track_name'] = df['track_name'].str.replace(r'\u202C|\u202D|\u202E|\u200E|\u200F|\u202A|\u202B', '', regex=True)
#Remove special characters from data set that were affecting matplotlib

font_path = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
#default font was causing issues with matlib with rendering certain unicode characters, so font was replaced

corr_matrix = df[['track_popularity', 'danceability', 'energy', 'loudness']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
# annot - controls whether to annotate each cell of the heatmap with the numeric value of the data.
# fmt - format the annotations to two decimal places
plt.title('Correlation between Features and Track Popularity')
plt.show()