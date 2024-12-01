from nba_api.stats.endpoints import playerestimatedmetrics, leaguedashplayerclutch, leaguedashplayerstats,leaguedashplayerbiostats, leaguehustlestatsplayer, playerdashptshots, playerdashboardbyshootingsplits
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time
from requests.exceptions import Timeout, ConnectionError, RequestException
from urllib3.exceptions import ProtocolError
from requests.exceptions import ReadTimeout


from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt


df = pd.read_csv('defensive_profile_with_name_filtered.csv')
# Drop non-numeric columns before scaling and clustering
numeric_df = df.select_dtypes(include=['number'])
numeric_df = numeric_df.drop(columns=['PLAYER_ID'], errors='ignore')
# Impute missing values with column mean
imputer = SimpleImputer(strategy='mean')
numeric_df_imputed = imputer.fit_transform(numeric_df)

# Scale the numeric data
scaler = StandardScaler()
stats_scaled = scaler.fit_transform(numeric_df_imputed)


# Generate the linkage matrix
Z = linkage(stats_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Players')
plt.ylabel('Distance')
plt.show()

# Set the maximum distance or the number of clusters to get cluster labels
max_distance = 14 # Adjust this based on the dendrogram visualization
clusters = fcluster(Z, max_distance, criterion='distance')
df['Cluster'] = clusters


# Print player names in each cluster
unique_clusters = df['Cluster'].unique()
unique_clusters.sort()
df.to_csv('defensive_clustered_players.csv', index=False)

for cluster in unique_clusters:
    players_in_cluster = df[df['Cluster'] == cluster]['PLAYER_NAME']
    print(f"Cluster {cluster}:")
    print(players_in_cluster.tolist())
    print("\n" + "-" * 50 + "\n")

# df = pd.read_csv('nba1.csv')
# # Drop non-numeric columns before scaling and clustering
# numeric_df = df.select_dtypes(include=['number'])
# numeric_df = numeric_df.drop(columns=['PLAYER_ID'], errors='ignore')
# # Impute missing values with column mean
# imputer = SimpleImputer(strategy='mean')
# numeric_df_imputed = imputer.fit_transform(numeric_df)

# # Scale the numeric data
# scaler = StandardScaler()
# stats_scaled = scaler.fit_transform(numeric_df_imputed)


# # Generate the linkage matrix
# Z = linkage(stats_scaled, method='ward')

# # Plot the dendrogram
# plt.figure(figsize=(10, 7))
# dendrogram(Z)
# plt.title('Dendrogram for Hierarchical Clustering')
# plt.xlabel('Players')
# plt.ylabel('Distance')
# plt.show()

# # Set the maximum distance or the number of clusters to get cluster labels
# max_distance = 13  # Adjust this based on the dendrogram visualization
# clusters = fcluster(Z, max_distance, criterion='distance')
# df['Cluster'] = clusters


# # Print player names in each cluster
# unique_clusters = df['Cluster'].unique()
# unique_clusters.sort()
# df.to_csv('clustered_players.csv', index=False)

# for cluster in unique_clusters:
#     players_in_cluster = df[df['Cluster'] == cluster]['PLAYER_NAME']
#     print(f"Cluster {cluster}:")
#     print(players_in_cluster.tolist())
#     print("\n" + "-" * 50 + "\n")
