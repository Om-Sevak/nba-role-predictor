import os
import pandas as pd
from dataExtractor3000 import extractOffensive, extractDefensive
from pca import pcaCluster
from hierarchicalClustering import hierarchicalCluster
from utils import printPlayerCluster, seasonData

# Set parameters directly in the script
offense_method = 'hierarchical'  # Options: 'pca_kmeans', 'pca_hierarchical', 'kmeans', 'hierarchical'
offense_clusters = 5  # Number of clusters for KMeans or PCA + KMeans
offense_distance = 12  # Distance threshold for Hierarchical clustering
offense_pca_components = None  # Number of PCA components (if using PCA-based methods)

defense_method = 'hierarchical'  # Options: 'pca_kmeans', 'pca_hierarchical', 'kmeans', 'hierarchical'
defense_clusters = 5  # Number of clusters for KMeans or PCA + KMeans
defense_distance = 16  # Distance threshold for Hierarchical clustering
defense_pca_components = None  # Number of PCA components (if using PCA-based methods)

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

def cluster_and_save(role, method, clusters=None, distance=None, pca_components=None):
    """
    Perform clustering (PCA, KMeans, or Hierarchical) on the specified role and save the results.
    """
    if method == 'pca_kmeans':
        print(f"Running PCA + KMeans clustering for {role} roles with {pca_components} components...")
        clustered_df = pcaCluster(role, clusters, pca_components=pca_components)
        file_name = f'{role}-pca-kmeans-clusters-{clusters}-{seasonData}.csv'
    elif method == 'pca_hierarchical':
        print(f"Running PCA + Hierarchical clustering for {role} roles with {pca_components} components...")
        clustered_df = pcaCluster(role, None, pca_components=pca_components)
        hierarchicalCluster(role, distance)
        file_name = f'{role}-pca-hierarchical-clusters-{distance}-{seasonData}.csv'
    elif method == 'kmeans':
        print(f"Running KMeans clustering for {role} roles with {clusters} clusters...")
        clustered_df = pcaCluster(role, clusters)
        file_name = f'{role}-kmeans-clusters-{clusters}-{seasonData}.csv'
    elif method == 'hierarchical':
        print(f"Running Hierarchical clustering for {role} roles with distance {distance}...")
        hierarchicalCluster(role, distance)
        file_name = f'{role}-hierarchical-clusters-{distance}-{seasonData}.csv'
        clustered_df = pd.read_csv(file_name)
    else:
        raise ValueError(f"Invalid clustering method: {method}")

    printPlayerCluster(file_name, role)
    return clustered_df

if __name__ == "__main__":
    # File names for offensive and defensive data
    offensive_file = f'offensive-data-{seasonData}.csv'
    defensive_file = f'defensive-data-{seasonData}.csv'

    # Load or extract offensive data
    offensive_data = check_and_load_data(offensive_file, extractOffensive)

    # Perform offensive clustering
    offensive_clusters = cluster_and_save(
        role='offensive',
        method=offense_method,
        clusters=offense_clusters,
        distance=offense_distance,
        pca_components=offense_pca_components
    )

    # Load or extract defensive data
    defensive_data = check_and_load_data(defensive_file, lambda: extractDefensive(offensive_clusters))

    # Perform defensive clustering
    defensive_clusters = cluster_and_save(
        role='defensive',
        method=defense_method,
        clusters=defense_clusters,
        distance=defense_distance,
        pca_components=defense_pca_components
    )

    print("Clustering completed for both offensive and defensive roles.")


