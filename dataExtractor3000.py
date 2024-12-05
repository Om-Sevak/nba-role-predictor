from nba_api.stats.endpoints import playerestimatedmetrics, leaguedashplayerclutch, leaguedashplayerstats,leaguedashplayerbiostats, leaguehustlestatsplayer, playerdashptshots, playerdashboardbyshootingsplits, leagueseasonmatchups
from utils import fetch_data_with_retries, seasonData
import pandas as pd
import time
from requests.exceptions import Timeout, ConnectionError, RequestException
from urllib3.exceptions import ProtocolError
# Fetching data for each table with retry mechanism

def extractOffensive():
    table1 = fetch_data_with_retries(playerestimatedmetrics.PlayerEstimatedMetrics, season=seasonData)
    if table1 is not None:
        table1 = table1.get_data_frames()[0]

    table2 = fetch_data_with_retries(leaguedashplayerbiostats.LeagueDashPlayerBioStats, season=seasonData)
    if table2 is not None:
        table2 = table2.get_data_frames()[0]

    table1 = table1[table1['MIN'] > 8]
    table1 = table1[table1['GP'] > 41]
    fromTable1 = table1[['PLAYER_ID', 'PLAYER_NAME','MIN']]
    print('table1')

    fromTable2 = table2[['PLAYER_ID', 'USG_PCT', 'AST_PCT']]
    print('table2')

    playerInfo = pd.merge(table1[['PLAYER_ID']], table2[['PLAYER_ID','TEAM_ID','PLAYER_NAME']], on='PLAYER_ID', how='inner')
    #playerInfo = merged[['PLAYER_ID','TEAM_ID','PLAYER_NAME']]

    shot_data = [] 
    index =0 
    max_retries = 10
    retry_delay = 5  
    index =0

    for _, row in playerInfo.iterrows():
        curr_player_id = row['PLAYER_ID']
        curr_team_id = row['TEAM_ID']
        player_name = row['PLAYER_NAME']
        success = False
        retries = 0
        index+=1

        while not success and retries < max_retries:
            try:
                response_shots = playerdashptshots.PlayerDashPtShots(player_id=curr_player_id, team_id=curr_team_id, season=seasonData, timeout=60)
                shotType = response_shots.get_data_frames()[1]
                dribbles = response_shots.get_data_frames()[3]
                contest = response_shots.get_data_frames()[4]

                response_split = playerdashboardbyshootingsplits.PlayerDashboardByShootingSplits(player_id=curr_player_id, season=seasonData, timeout=60)
                fa = response_split.get_data_frames()[0]
                fga = fa['FGA'][0]
                
                shotTech = response_split.get_data_frames()[5]

                shotDistance = response_split.get_data_frames()[3]
                shot_data.append({
                    'PLAYER_ID': curr_player_id,
                    'C&S_FREQ': shotType['FGA_FREQUENCY'].iloc[0],
                    'LOW_DRIBBLE_FREQ': dribbles['FGA_FREQUENCY'].iloc[0] + dribbles['FGA_FREQUENCY'].iloc[1] + dribbles['FGA_FREQUENCY'].iloc[2],
                    'CONTESTED_FREQ': contest['FGA_FREQUENCY'].iloc[0] + contest['FGA_FREQUENCY'].iloc[1],
                    'PAINT_FREQ': shotDistance['FGA'][1] / fga,
                    'MIDRANGE_FREQ': shotDistance['FGA'][2] / fga,
                    'CORNER_FREQ': (shotDistance['FGA'][3] + shotDistance['FGA'][4]) / fga,
                    'ATB_FREQ': shotDistance['FGA'][5] / fga,
                    'DUNK_FREQ': shotTech['FGA'][2] / fga,
                    'LAYUP_FREQ': shotTech['FGA'][7] / fga,
                    'HOOK_FREQ': shotTech['FGA'][5] / fga,               
                })
                success = True  
            except (Timeout, ConnectionError, RequestException, ProtocolError):
                retries += 1
                print(f"Timeout for PLAYER_ID {curr_player_id}, retrying {retries}/{max_retries} after {retry_delay} seconds...")
                time.sleep(retry_delay*retries)
            except Exception as e:
                print(f"Failed to retrieve data for PLAYER_ID {curr_player_id} with TEAM_ID {curr_team_id}: {e}")
                break
        print("%s/%s: %s -> %s" %(index,playerInfo.shape[0],player_name,success))
        
    print('done table3')
    fromTable3 = pd.DataFrame(shot_data)
    print('table3')
    # List of dataframes to merge
    dataframes = [fromTable1, fromTable2, fromTable3]

    # Perform successive inner joins on 'PLAYER_ID' to keep only common players
    df = dataframes[0]
    for curr in dataframes[1:]:
        df = pd.merge(df, curr, on='PLAYER_ID', how='inner')
    print('joined')
    df.to_csv('offensive-data-%s.csv' %(seasonData), index=False)

def extractDefensive(offClustered):
    matchups_df = fetch_data_with_retries(leagueseasonmatchups.LeagueSeasonMatchups, season=seasonData)
    if matchups_df is not None:
        matchups_df = matchups_df.get_data_frames()[0]

    # Merge offensive clusters with matchups data
    matchups_with_clusters = matchups_df.merge(offClustered[['PLAYER_ID', 'Cluster']], left_on='OFF_PLAYER_ID', right_on='PLAYER_ID')
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

    defense_profile_pivot_with_id = defense_profile_pivot.reset_index()

    # Perform an inner join to only include players present in the original df
    defense_profile_with_name = defense_profile_pivot_with_id.merge(
        offClustered[['PLAYER_ID', 'PLAYER_NAME']], 
        left_on='DEF_PLAYER_ID', 
        right_on='PLAYER_ID', 
        how='inner'  # Ensures only matching records are included
    )

    # Drop the duplicate PLAYER_ID column (if any)
    defense_profile_with_name = defense_profile_with_name.drop(columns=['DEF_PLAYER_ID'], errors='ignore')

    # Save the defensive profile with player names to a CSV file
    # Convert all column names to strings
    defense_profile_with_name.columns = defense_profile_with_name.columns.map(str)

    defense_profile_with_name.to_csv('defensive-data-%s.csv' %(seasonData), index=False)
    return defense_profile_with_name

