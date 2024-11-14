from nba_api.stats.endpoints import playerestimatedmetrics, leaguedashplayerclutch, leaguedashplayerstats,leaguedashplayerbiostats, leaguehustlestatsplayer, playerdashptshots, playerdashboardbyshootingsplits, leagueseasonmatchups
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

# Determine optimal k using elbow method for offensive clustering
# wcss = []
# k_values = range(1, 15)
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(stats_scaled)
#     wcss.append(kmeans.inertia_)
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, wcss, marker='o')
# plt.title('Elbow Method For Optimal k - Offensive Roles')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
# plt.show()

# Set the optimal number of clusters for offensive roles
optimal_k = 8  # Adjust based on the elbow graph
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(stats_scaled)

# Load defensive matchups data
matchups_df = fetch_data_with_retries(leagueseasonmatchups.LeagueSeasonMatchups, season=seasonData)
if matchups_df is not None:
    matchups_df = matchups_df.get_data_frames()[0]

# Merge offensive clusters with matchups data
matchups_with_clusters = matchups_df.merge(df[['PLAYER_ID', 'Cluster']], left_on='OFF_PLAYER_ID', right_on='PLAYER_ID')
matchups_with_clusters = matchups_with_clusters.rename(columns={'Cluster': 'Offensive_Cluster'})

matchups_with_clusters['MATCHUP_TIME_MIN'] = matchups_with_clusters['MATCHUP_TIME_SEC'] / 60

# Aggregate total matchup minutes each defender spends on each offensive cluster
defensive_profile_df = (
    matchups_with_clusters
    .groupby(['DEF_PLAYER_ID', 'Offensive_Cluster'])
    .agg(total_matchup_min=('MATCHUP_TIME_MIN', 'sum'))
    .reset_index()
)

# Calculate total minutes each defender has played in matchups
total_minutes_df = defensive_profile_df.groupby('DEF_PLAYER_ID')['total_matchup_min'].sum().reset_index()
total_minutes_df = total_minutes_df.rename(columns={'total_matchup_min': 'total_minutes'})

# Ensure columns are numeric for division
defensive_profile_df['total_matchup_min'] = pd.to_numeric(defensive_profile_df['total_matchup_min'], errors='coerce')
total_minutes_df['total_minutes'] = pd.to_numeric(total_minutes_df['total_minutes'], errors='coerce')

# Fill any NaN values from conversion, if necessary
defensive_profile_df['total_matchup_min'].fillna(0, inplace=True)
total_minutes_df['total_minutes'].fillna(0, inplace=True)

# Merge to calculate the percentage of time each defender spent guarding each offensive cluster
defensive_profile_df = defensive_profile_df.merge(total_minutes_df, on='DEF_PLAYER_ID')
defensive_profile_df['percentage_time'] = defensive_profile_df['total_matchup_min'] / defensive_profile_df['total_minutes']

# Pivot to have each offensive cluster as a separate column for each defender
defense_profile_pivot = defensive_profile_df.pivot(index='DEF_PLAYER_ID', columns='Offensive_Cluster', values='percentage_time').fillna(0)

defense_profile_pivot.to_csv('test.csv', index=False)
# Scale the defensive profile data
scaler = StandardScaler()
defense_profile_scaled = scaler.fit_transform(defense_profile_pivot)

# Determine optimal k using elbow method for defensive clustering, if needed
# wcss_defensive = []
# k_values_defensive = range(1, 15)
# for k in k_values_defensive:
#     kmeans_defensive = KMeans(n_clusters=k, random_state=42)
#     kmeans_defensive.fit(defense_profile_scaled)
#     wcss_defensive.append(kmeans_defensive.inertia_)
# plt.figure(figsize=(10, 6))
# plt.plot(k_values_defensive, wcss_defensive, marker='o')
# plt.title('Elbow Method For Optimal k - Defensive Roles')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
# plt.show()

# Set the optimal number of clusters for defensive roles
optimal_k_defensive = 5  # Adjust based on the elbow graph
kmeans_defensive = KMeans(n_clusters=optimal_k_defensive, random_state=42)
defensive_clusters = kmeans_defensive.fit_predict(defense_profile_scaled)

# Add defensive clusters back to defense_profile_pivot
defense_profile_pivot['Defensive_Cluster'] = defensive_clusters

# Merge defensive clusters with the main DataFrame
defensive_roles = defense_profile_pivot[['Defensive_Cluster']].reset_index()
df = df.merge(defensive_roles, left_on='PLAYER_ID', right_on='DEF_PLAYER_ID', how='left')

# Print players in each defensive cluster
for cluster in range(optimal_k_defensive):
    players_in_cluster = df[df['Defensive_Cluster'] == cluster]['PLAYER_NAME']
    print(f"Defensive Cluster {cluster}:")
    print(players_in_cluster.tolist())
    print("\n" + "-" * 50 + "\n")


# Print player names in each cluster
for cluster in range(optimal_k):
    players_in_cluster = df[df['Cluster'] == cluster]['PLAYER_NAME']
    print(f"Cluster {cluster}:")
    print(players_in_cluster.tolist())
    print("\n" + "-" * 50 + "\n")

