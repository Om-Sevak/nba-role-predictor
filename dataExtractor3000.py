from nba_api.stats.endpoints import playerestimatedmetrics, leaguedashplayerclutch, leaguedashplayerstats, leaguedashplayerbiostats, leaguehustlestatsplayer, playerdashptshots, playerdashboardbyshootingsplits, leagueseasonmatchups
from utils import fetchDataWithRetries, seasonData
import pandas as pd
import time
from requests.exceptions import Timeout, ConnectionError, RequestException
from urllib3.exceptions import ProtocolError

# get all the needed offensive stats from the API
def extractOffensive():
    table1 = fetchDataWithRetries(playerestimatedmetrics.PlayerEstimatedMetrics, season=seasonData)
    if table1 is not None:
        table1 = table1.get_data_frames()[0]

    table2 = fetchDataWithRetries(leaguedashplayerbiostats.LeagueDashPlayerBioStats, season=seasonData)
    if table2 is not None:
        table2 = table2.get_data_frames()[0]

    table1 = table1[table1['MIN'] > 8]
    table1 = table1[table1['GP'] > 41]
    fromTable1 = table1[['PLAYER_ID', 'PLAYER_NAME', 'MIN']]
    print('got table1')

    fromTable2 = table2[['PLAYER_ID', 'USG_PCT', 'AST_PCT']]
    print('got table2')

    playerInfo = pd.merge(table1[['PLAYER_ID']], table2[['PLAYER_ID', 'TEAM_ID', 'PLAYER_NAME']], on='PLAYER_ID', how='inner')

    shotData = [] 
    index = 0 
    maxRetries = 10
    retryDelay = 5  

    for _, row in playerInfo.iterrows():
        currPlayerId = row['PLAYER_ID']
        currTeamId = row['TEAM_ID']
        playerName = row['PLAYER_NAME']
        success = False
        retries = 0
        index += 1

        while not success and retries < maxRetries:
            try:
                responseShots = playerdashptshots.PlayerDashPtShots(player_id=currPlayerId, team_id=currTeamId, season=seasonData, timeout=60)
                shotType = responseShots.get_data_frames()[1]
                dribbles = responseShots.get_data_frames()[3]
                contest = responseShots.get_data_frames()[4]

                responseSplit = playerdashboardbyshootingsplits.PlayerDashboardByShootingSplits(player_id=currPlayerId, season=seasonData, timeout=60)
                fa = responseSplit.get_data_frames()[0]
                fga = fa['FGA'][0]
                
                shotTech = responseSplit.get_data_frames()[5]

                shotDistance = responseSplit.get_data_frames()[3]
                shotData.append({
                    'PLAYER_ID': currPlayerId,
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
                print(f"Timeout for PLAYER_ID {currPlayerId}, retrying {retries}/{maxRetries} after {retryDelay} seconds...")
                time.sleep(retryDelay * retries)
            except Exception as e:
                print(f"Failed to retrieve data for PLAYER_ID {currPlayerId} with TEAM_ID {currTeamId}: {e}")
                break
        print(f"{index}/{playerInfo.shape[0]}: {playerName} -> Success: {success}")

    fromTable3 = pd.DataFrame(shotData)
    print('got table3')

    dataframes = [fromTable1, fromTable2, fromTable3]
    df = dataframes[0]
    for curr in dataframes[1:]:
        df = pd.merge(df, curr, on='PLAYER_ID', how='inner')
    print('joined all the tables. got offensive data')
    
    df.to_csv(f'offensive-data-{seasonData}.csv', index=False)

# get all the needed defensive mutchups from the API and create the matchup frequencies
def extractDefensive(offClustered):
    matchupsDf = fetchDataWithRetries(leagueseasonmatchups.LeagueSeasonMatchups, season=seasonData)
    if matchupsDf is not None:
        matchupsDf = matchupsDf.get_data_frames()[0]

    # merge offensive clusters with matchups data
    matchupsWithClusters = matchupsDf.merge(offClustered[['PLAYER_ID', 'Cluster']], left_on='OFF_PLAYER_ID', right_on='PLAYER_ID')
    matchupsWithClusters = matchupsWithClusters.rename(columns={'Cluster': 'Offensive_Cluster'})

    matchupsWithClusters['MATCHUP_TIME_MIN'] = matchupsWithClusters['MATCHUP_TIME_SEC'] / 60

    # aggregate total matchup minutes each defender spends on each offensive cluster
    defensiveProfileDf = (
        matchupsWithClusters
        .groupby(['DEF_PLAYER_ID', 'Offensive_Cluster'])
        .agg(totalMatchupMin=('MATCHUP_TIME_MIN', 'sum'))
        .reset_index()
    )

    # calculate total minutes each defender has played in matchups
    totalMinutesDf = defensiveProfileDf.groupby('DEF_PLAYER_ID')['totalMatchupMin'].sum().reset_index()
    totalMinutesDf = totalMinutesDf.rename(columns={'totalMatchupMin': 'totalMinutes'})

    # ensure columns are numeric for division
    defensiveProfileDf['totalMatchupMin'] = pd.to_numeric(defensiveProfileDf['totalMatchupMin'], errors='coerce')
    totalMinutesDf['totalMinutes'] = pd.to_numeric(totalMinutesDf['totalMinutes'], errors='coerce')

    # fill any NaN values from conversion, if necessary
    defensiveProfileDf['totalMatchupMin'].fillna(0, inplace=True)
    totalMinutesDf['totalMinutes'].fillna(0, inplace=True)

    # merge to calculate the percentage of time each defender spent guarding each offensive cluster
    defensiveProfileDf = defensiveProfileDf.merge(totalMinutesDf, on='DEF_PLAYER_ID')
    defensiveProfileDf['percentageTime'] = defensiveProfileDf['totalMatchupMin'] / defensiveProfileDf['totalMinutes']

    # have each offensive cluster as a separate column for each defender
    defenseProfilePivot = defensiveProfileDf.pivot(index='DEF_PLAYER_ID', columns='Offensive_Cluster', values='percentageTime').fillna(0)
    defenseProfilePivotWithId = defenseProfilePivot.reset_index()

    # perform an inner join to only include players present in the offensive clusters
    defenceStatsDf = defenseProfilePivotWithId.merge(
        offClustered[['PLAYER_ID', 'PLAYER_NAME']], 
        left_on='DEF_PLAYER_ID', 
        right_on='PLAYER_ID', 
        how='inner'
    )

    defenceStatsDf = defenceStatsDf.drop(columns=['DEF_PLAYER_ID'], errors='ignore')
    defenceStatsDf.columns = defenceStatsDf.columns.map(str)

    defenceStatsDf.to_csv(f'defensive-data-{seasonData}.csv', index=False)
    return defenceStatsDf

