from nba_api.stats.endpoints import leaguedashplayerbiostats, leaguedashplayerptshot, leaguedashplayerstats
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


table1 = leaguedashplayerbiostats.LeagueDashPlayerBioStats( season= '2023-24').get_data_frames()[0]
table2 = leaguedashplayerptshot.LeagueDashPlayerPtShot( season= '2023-24').get_data_frames()[0]
table3 = leaguedashplayerstats.LeagueDashPlayerStats( season= '2023-24').get_data_frames()[0]

merged_df = pd.merge(table1, table2, on='PLAYER_ID', how='inner')
output = pd.merge(merged_df, table3, on='PLAYER_ID', how='inner')

df_numeric = output.select_dtypes(include=[np.number])  # Select only numeric columns
df_final = df_numeric.fillna(0)
# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_final)
print(df_scaled.shape)

# Perform PCA
pca = PCA(0.6)  # Set the number of components you need
df_pca = pca.fit_transform(df_scaled)

#df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
print(df_pca.shape)

print("Explained variance ratio:", pca.explained_variance_ratio_)

loadings = pd.DataFrame(
    pca.components_.T,  # Transpose to make each row correspond to a feature
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=df_final.columns  # Assuming `df` has the original feature names as columns
)

for i in range(pca.n_components_):
    component = loadings.iloc[:, i]  # Get loadings for the i-th component
    sorted_component = component.abs().sort_values(ascending=False)  # Sort by absolute value
    print(f"\nTop features for Principal Component {i+1}:")
    print(sorted_component.head(20))  # Display top 5 features for each component (adjust as needed)