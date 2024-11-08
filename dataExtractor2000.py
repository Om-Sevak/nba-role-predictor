from nba_api.stats.endpoints import leaguedashplayerbiostats, leaguedashplayerptshot, leaguedashplayerstats
import pandas as pd


table1 = leaguedashplayerbiostats.LeagueDashPlayerBioStats( season= '2023-24').get_data_frames()[0]
table2 = leaguedashplayerptshot.LeagueDashPlayerPtShot( season= '2023-24').get_data_frames()[0]
table3 = leaguedashplayerstats.LeagueDashPlayerStats( season= '2023-24').get_data_frames()[0]

merged_df = pd.merge(table1, table2, on='PLAYER_ID', how='inner')
output = pd.merge(merged_df, table3, on='PLAYER_ID', how='inner')

print(output)