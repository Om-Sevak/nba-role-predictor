from utils import seasonData, dataFrameScale
import pandas as pd
from sklearn.decomposition import PCA

# how to use: put defensive/offensive in role
def pcaCluster(df, role, components):

    stats_scaled, numeric_df = dataFrameScale(df)

    # Perform PCA
    pca = PCA(n_components=components)  # Set the number of components you need
    df_pca = pca.fit_transform(stats_scaled)

    pca_df = pd.DataFrame(
        df_pca,
        columns=[f"PC{i+1}" for i in range(df_pca.shape[1])]
    )
    pca_df['PLAYER_ID'] = df['PLAYER_ID']
    pca_df['PLAYER_NAME'] = df['PLAYER_NAME']

    # Save PCA-transformed data to a CSV
    pca_df.to_csv(f'pca-components-{seasonData}.csv', index=False)
    print("PCA components saved to pca_components-season.csv")

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

    return pca_df