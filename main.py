import os
import pandas as pd
from dataExtractor3000 import extractOffensive, extractDefensive
from pca import pcaCluster
from hierarchicalClustering import hierarchicalCluster
from utils import printPlayerCluster, seasonData, kMeansCluster
from classification import classificationAccuracy

# Set parameters directly in the script
offense_method = 'pca_hierarchical'  # Options: 'pca_kmeans', 'pca_hierarchical', 'kmeans', 'hierarchical'
offense_clusters = 8  # Number of clusters for KMeans or PCA + KMeans
offense_distance = 9 # Distance threshold for Hierarchical clustering
offense_pca_components = 4  # Number of PCA components (if using PCA-based methods) 4 usually good

defense_method = 'pca_hierarchical'  # Options: 'pca_kmeans', 'pca_hierarchical', 'kmeans', 'hierarchical'
defense_clusters = 6  # Number of clusters for KMeans or PCA + KMeans
defense_distance = 11  # Distance threshold for Hierarchical clustering
defense_pca_components = 4  # Number of PCA components (if using PCA-based methods)

def check_and_load_data(file_name, extractor_function):
    """
    Check if data exists for the given season. If not, run the extractor function to download data.
    """
    if os.path.exists(file_name):
        print(f"Data file '{file_name}' already exists. Loading from file...")
        return pd.read_csv(file_name)
    else:
        print(f"Data file '{file_name}' not found. Extracting data...")
        extractor_function()
        return pd.read_csv(file_name)

def cluster_and_save(role, method, clusters=None, distance=None, pca_components=None, data=None):
    """
    Perform clustering (PCA, KMeans, or Hierarchical) on the specified role and save the results.
    """
    if method == 'pca_kmeans':
        print(f"Running PCA + KMeans clustering for {role} roles with {pca_components} components...")
        pca_df = pcaCluster(data, role, pca_components)
        clustered_df = kMeansCluster(pca_df,clusters)
        file_name = f'{role}-pca-kmeans-clusters-{seasonData}.csv'
    elif method == 'pca_hierarchical':
        print(f"Running PCA + Hierarchical clustering for {role} roles with {pca_components} components...")
        pca_df = pcaCluster(data, role, pca_components)
        clustered_df = hierarchicalCluster(pca_df, distance)
        file_name = f'{role}-pca-hierarchical-clusters-{seasonData}.csv'
    elif method == 'kmeans':
        print(f"Running KMeans clustering for {role} roles with {clusters} clusters...")
        clustered_df = kMeansCluster(data, clusters)
        file_name = f'{role}-kmeans-clusters-{seasonData}.csv'
    elif method == 'hierarchical':
        print(f"Running Hierarchical clustering for {role} roles with distance {distance}...")
        file_name = f'{role}-hierarchical-clusters-{seasonData}.csv'
        clustered_df = hierarchicalCluster(data, distance)
    else:
        raise ValueError(f"Invalid clustering method: {method}")
    clustered_df.to_csv(file_name, index=False)
    printPlayerCluster(file_name, role)
    return clustered_df

if __name__ == "__main__":
    # File names for offensive and defensive data
    offensive_file = f'offensive-data-{seasonData}.csv'

    # Load or extract offensive data
    offensive_data = check_and_load_data(offensive_file, extractOffensive)

    # Perform offensive clustering
    offensive_clusters = cluster_and_save(
        role='offensive',
        method=offense_method,
        clusters=offense_clusters,
        distance=offense_distance,
        pca_components=offense_pca_components,
        data = offensive_data
    )

    print("\nExtracting defensive data based on updated offensive roles...")
    defensive_data = extractDefensive(offensive_clusters)

    # Perform defensive clustering
    defensive_clusters = cluster_and_save(
        role='defensive',
        method=defense_method,
        clusters=defense_clusters,
        distance=defense_distance,
        pca_components=defense_pca_components,
        data = defensive_data
    )

    print("classification accuracy results for offensive roles:")
    classificationAccuracy(offensive_clusters, 'offensive')
    print("classification accuracy results for defensive roles:")
    classificationAccuracy(defensive_clusters, 'defensive')

    print("Clustering completed for both offensive and defensive roles.")


