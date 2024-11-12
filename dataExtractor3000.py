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

seasonData = '2023-24'
# Load data from the API
seasonData = '2023-24'
max_retries = 5
retry_delay = 5  # seconds

# Function to fetch data with retry and exponential backoff
def fetch_data_with_retries(func, *args, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            # Call the API function and get data
            return func(*args, **kwargs)
        except ReadTimeout:
            retries += 1
            print(f"ReadTimeout error encountered. Retrying {retries}/{max_retries}...")
            time.sleep(retry_delay * (2 ** retries))  # Exponential backoff
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    return None  # Return None if all retries fail

# Fetching data for each table with retry mechanism
# table1 = fetch_data_with_retries(playerestimatedmetrics.PlayerEstimatedMetrics, season=seasonData)
# if table1 is not None:
#     table1 = table1.get_data_frames()[0]

# table2 = fetch_data_with_retries(leaguedashplayerclutch.LeagueDashPlayerClutch, season=seasonData)
# if table2 is not None:
#     table2 = table2.get_data_frames()[0]

# table3 = fetch_data_with_retries(leaguedashplayerstats.LeagueDashPlayerStats, season=seasonData)
# if table3 is not None:
#     table3 = table3.get_data_frames()[0]

# table4 = fetch_data_with_retries(leaguedashplayerbiostats.LeagueDashPlayerBioStats, season=seasonData)
# if table4 is not None:
#     table4 = table4.get_data_frames()[0]

# table5 = fetch_data_with_retries(leaguehustlestatsplayer.LeagueHustleStatsPlayer, season=seasonData)
# if table5 is not None:
#     table5 = table5.get_data_frames()[0]
# print('got tables')

# table1 = table1[table1['MIN'] > 8]
# fromTable1 = table1[['PLAYER_ID', 'PLAYER_NAME','MIN', 'E_PACE', 'E_TOV_PCT']]
# print('table1')

# table2['CFGA/CMIN']= table2['FGA'] / table2['MIN'].replace(0, 1e-9)
# table2['CFTA/CMIN'] = table2['FTA'] / table2['MIN'].replace(0, 1e-9)
# fromTable2 = table2[['PLAYER_ID', 'CFGA/CMIN', 'CFTA/CMIN']]
# print('table2')

# table3 = table3[table3['GP'] > 41]
# table3['FGA/MIN'] = table3['FGA'] / table3['MIN'].replace(0, 1e-9)
# table3['FTA/MIN'] = table3['FTA'] / table3['MIN'].replace(0, 1e-9)
# fromTable3 = table3[['PLAYER_ID', 'FGA/MIN', 'FTA/MIN']]
# print('table3')

# fromTable4 = table4[['PLAYER_ID', 'OREB_PCT', 'USG_PCT', 'AST_PCT']]
# print('table4')

# table5['SCREEN_AST/MIN'] = table5['SCREEN_ASSISTS'] / table5['MIN'].replace(0, 1e-9)
# fromTable5 = table5[['PLAYER_ID', 'SCREEN_AST/MIN']]
# print('table5')

# playerMin = table3[['PLAYER_ID','TEAM_ID','PLAYER_NAME']]

# shot_data = [] 
# index =0 
# max_retries = 10
# retry_delay = 5  
# index =0

# for _, row in playerMin.iterrows():
#     curr_player_id = row['PLAYER_ID']
#     curr_team_id = row['TEAM_ID']
#     player_name = row['PLAYER_NAME']
#     success = False
#     retries = 0
#     index+=1

#     while not success and retries < max_retries:
#         try:
#             response_shots = playerdashptshots.PlayerDashPtShots(player_id=curr_player_id, team_id=curr_team_id, season=seasonData, timeout=60)
#             shotType = response_shots.get_data_frames()[1]
#             dribbles = response_shots.get_data_frames()[3]
#             contest = response_shots.get_data_frames()[4]

#             response_split = playerdashboardbyshootingsplits.PlayerDashboardByShootingSplits(player_id=curr_player_id, season=seasonData, timeout=60)
#             fa = response_split.get_data_frames()[0]
#             fga = fa['FGA'][0]

#             shotDistance = response_split.get_data_frames()[3]
#             shot_data.append({
#                 'PLAYER_ID': curr_player_id,
#                 'C&S_FREQ': shotType['FGA_FREQUENCY'].iloc[0],
#                 'PULL_UP_FREQ': shotType['FGA_FREQUENCY'].iloc[1],
#                 'LOW_DRIBBLE_FREQ': dribbles['FGA_FREQUENCY'].iloc[0] + dribbles['FGA_FREQUENCY'].iloc[1] + dribbles['FGA_FREQUENCY'].iloc[2],
#                 'CONTESTED_FREQ': contest['FGA_FREQUENCY'].iloc[0] + contest['FGA_FREQUENCY'].iloc[1],
#                 'RA_FREQ': shotDistance['FGA'][0] / fga,
#                 'PAINT_FREQ': shotDistance['FGA'][1] / fga,
#                 'MIDRANGE_FREQ': shotDistance['FGA'][2] / fga,
#                 'CORNER_FREQ': (shotDistance['FGA'][3] + shotDistance['FGA'][4]) / fga,
#                 'ATB_FREQ': shotDistance['FGA'][5] / fga,
#             })
#             success = True  
#         except (Timeout, ConnectionError, RequestException, ProtocolError):
#             retries += 1
#             print(f"Timeout for PLAYER_ID {curr_player_id}, retrying {retries}/{max_retries} after {retry_delay} seconds...")
#             time.sleep(retry_delay*retries)
#         except Exception as e:
#             print(f"Failed to retrieve data for PLAYER_ID {curr_player_id} with TEAM_ID {curr_team_id}: {e}")
#             break
#     print("%s/%s: %s -> %s" %(index,playerMin.shape[0],player_name,success))
      
# print('done table6')
# fromTable6 = pd.DataFrame(shot_data)
# print('table6')
# # List of dataframes to merge
# dataframes = [fromTable1, fromTable2, fromTable3, fromTable4, fromTable5, fromTable6]

# # Perform successive inner joins on 'PLAYER_ID' to keep only common players
# df = dataframes[0]
# for curr in dataframes[1:]:
#     df = pd.merge(df, curr, on='PLAYER_ID', how='inner')
# print('joined')
# df.to_csv('nba.csv', index=False)

df = pd.read_csv('nba.csv')
# Drop non-numeric columns before scaling and clustering
numeric_df = df.select_dtypes(include=['number'])
numeric_df = numeric_df.drop(columns=['PLAYER_ID'], errors='ignore')
# Impute missing values with column mean
imputer = SimpleImputer(strategy='mean')
numeric_df_imputed = imputer.fit_transform(numeric_df)

# Scale the numeric data
scaler = StandardScaler()
stats_scaled = scaler.fit_transform(numeric_df_imputed)

# wcss = []
# k_values = range(1, 15)  # Try up to 15 clusters, for example
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(stats_scaled)
#     wcss.append(kmeans.inertia_)  # inertia_ is the WCSS

# plt.figure(figsize=(10, 6))
# plt.plot(k_values, wcss, marker='o')
# plt.title('Elbow Method For Optimal k')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
# plt.show()

# Use the optimal number of clusters based on the elbow graph
optimal_k = 8  # Replace with the value you observe from the elbow graph
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(stats_scaled)

# Print player names in each cluster
for cluster in range(optimal_k):
    players_in_cluster = df[df['Cluster'] == cluster]['PLAYER_NAME']
    print(f"Cluster {cluster}:")
    print(players_in_cluster.tolist())
    print("\n" + "-" * 50 + "\n")

