from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt


def hierarchicalCluster(df, distance):

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
    max_distance = distance # Adjust this based on the dendrogram visualization
    clusters = fcluster(Z, max_distance, criterion='distance')
    df['Cluster'] = clusters


    # Print player names in each cluster
    unique_clusters = df['Cluster'].unique()
    unique_clusters.sort()

    return df