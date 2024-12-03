from utils import seasonData, elbowFunction, dataFrameScale
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# how to use: put defensive/offensive in role
def pcaCluster(role, clusters):
    df = pd.read_csv('%s-data-%s.csv' %(role, seasonData))

    stats_scaled, numeric_df = dataFrameScale(df)

    # Perform PCA
    pca = PCA(n_components=4)  # Set the number of components you need
    df_pca = pca.fit_transform(stats_scaled)

    pca_df = pd.DataFrame(
        df_pca,
        columns=[f"PC{i+1}" for i in range(df_pca.shape[1])]
    )
    pca_df['PLAYER_ID'] = df['PLAYER_ID']
    pca_df['PLAYER_NAME'] = df['PLAYER_NAME']

    # Save PCA-transformed data to a CSV
    pca_df.to_csv('pca-components-%s-%s.csv' %(role, seasonData), index=False)
    print("PCA components saved to pca_components-role-season.csv")

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    loadings = pd.DataFrame(
        pca.components_.T,  # Transpose to make each row correspond to a feature
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=numeric_df.columns  # Assuming `df` has the original feature names as columns
    )

    for i in range(pca.n_components_):
        component = loadings.iloc[:, i]  # Get loadings for the i-th component
        # Sort by absolute value but retain the original sign by reindexing
        sorted_component = component.reindex(component.abs().sort_values(ascending=False).index)
        print(f"\nTop features for Principal Component {i+1}:")
        print(sorted_component.head(20))  # Display top 20 features for each component (adjust as needed)

    elbowFunction(df_pca)

    kmeans = KMeans(n_clusters=clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_pca)

    return df