from nba_api.stats.endpoints import leaguedashplayerbiostats, leaguedashplayerptshot, leaguedashplayerstats
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data from the API
table1 = leaguedashplayerbiostats.LeagueDashPlayerBioStats(season='2023-24').get_data_frames()[0]
table2 = leaguedashplayerptshot.LeagueDashPlayerPtShot(season='2023-24').get_data_frames()[0]
table3 = leaguedashplayerstats.LeagueDashPlayerStats(season='2023-24').get_data_frames()[0]

# Merge the tables on 'PLAYER_ID'
merged_df = pd.merge(table1, table2, on='PLAYER_ID', how='inner')
df = pd.merge(merged_df, table3, on='PLAYER_ID', how='inner')

# Save the merged DataFrame to a CSV file
# df.to_csv('nba_player_data_2023_24.csv', index=False)


# Drop non-numeric columns before scaling and clustering
numeric_df = df.select_dtypes(include=['number'])

# Impute missing values with column mean
imputer = SimpleImputer(strategy='mean')
numeric_df_imputed = imputer.fit_transform(numeric_df)

# Scale the numeric data
scaler = StandardScaler()
stats_scaled = scaler.fit_transform(numeric_df_imputed)

# Determine optimal number of clusters (set manually as 4)
optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(stats_scaled)

# Print player names in each cluster
for cluster in range(optimal_k):
    players_in_cluster = df[df['Cluster'] == cluster]['PLAYER_NAME']
    print(f"Cluster {cluster}:")
    print(players_in_cluster.tolist())
    print("\n" + "-" * 50 + "\n")
