import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

df = pd.read_csv('spotify_songs.csv')

df['track_name'] = df['track_name'].replace(r'\$', '\\$', regex=True)
#Adds a backslash before dollar signs to treat as regular character because LaTeX was trying to interpret 
df['track_name'] = df['track_name'].str.replace(r'\u202C|\u202D|\u202E|\u200E|\u200F|\u202A|\u202B', '', regex=True)
#Remove special characters from data set that were affecting matplotlib

font_path = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font_prop = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
#default font was causing issues with matlib with rendering certain unicode characters, so font was replaced

yearly_popularity = df.groupby('playlist_genre')['track_popularity'].mean().reset_index()

numerical_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
summary_stats = df[numerical_features].describe()
#describe generates descriptive statistics
print("Summary Statistics:\n", summary_stats)

plt.figure(figsize=(10, 6))
sns.barplot(data=yearly_popularity, x='playlist_genre', y='track_popularity')
plt.title('Average Track Popularity by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Popularity')
plt.grid(True)
plt.show()