import os
import pandas as pd
from dataExtractor3000 import extractOffensive, extractDefensive
from pca import pcaCluster
from hierarchicalClustering import hierarchicalCluster
from utils import printPlayerCluster, seasonData, kMeansCluster
from classification import classificationAccuracy

# set parameters to decide which clustering method to use for the roles
offenseMethod = 'pca_hierarchical'  # options: 'pca_kmeans', 'pca_hierarchical', 'kmeans', 'hierarchical'
offenseClusters = 8  # number of clusters for kmeans or pca_kmeans
offenseDistance = 9  # distance threshold for hierarchical
offensePcaComponents = 4  # number of PCA components (if using PCA) 4 usually good

defenseMethod = 'pca_hierarchical'  # options: 'pca_kmeans', 'pca_hierarchical', 'kmeans', 'hierarchical'
defenseClusters = 6  # number of clusters for kmeans or pca_kmeans
defenseDistance = 11  # distance threshold for hierarchical
defensePcaComponents = 4  # number of PCA components (if using PCA)

# check if data exists for the given season. if not, run the extractor function to get data.
def checkAndLoadData(fileName, extractorFunction):
    if os.path.exists(fileName):
        print(f"Data file '{fileName}' already exists. Loading from file...")
        return pd.read_csv(fileName)
    else:
        print(f"Data file '{fileName}' not found. Extracting data...")
        extractorFunction()
        return pd.read_csv(fileName)

# call the clustering function based on the pre determined method
def clusterAndSave(role, method, clusters=None, distance=None, pcaComponents=None, data=None):
    if method == 'pca_kmeans':
        print(f"Running PCA + KMeans clustering for {role} roles with {pcaComponents} components...")
        pcaDf = pcaCluster(data, pcaComponents)
        clusteredDf = kMeansCluster(pcaDf, clusters)
    elif method == 'pca_hierarchical':
        print(f"Running PCA + Hierarchical clustering for {role} roles with {pcaComponents} components...")
        pcaDf = pcaCluster(data, pcaComponents)
        clusteredDf = hierarchicalCluster(pcaDf, distance)
    elif method == 'kmeans':
        print(f"Running KMeans clustering for {role} roles with {clusters} clusters...")
        clusteredDf = kMeansCluster(data, clusters)
    elif method == 'hierarchical':
        print(f"Running Hierarchical clustering for {role} roles with distance {distance}...")
        clusteredDf = hierarchicalCluster(data, distance)
    else:
        raise ValueError(f"Invalid clustering method: {method}")
    printPlayerCluster(clusteredDf, role)
    return clusteredDf

if __name__ == "__main__":

    offensiveFile = f'offensive-data-{seasonData}.csv'
    offensiveData = checkAndLoadData(offensiveFile, extractOffensive)

    # perform offensive clustering
    offensiveClusters = clusterAndSave(
        role='offensive',
        method=offenseMethod,
        clusters=offenseClusters,
        distance=offenseDistance,
        pcaComponents=offensePcaComponents,
        data=offensiveData
    )

    print("\nExtracting defensive data based on updated offensive roles...")
    defensiveData = extractDefensive(offensiveClusters)

    # Perform defensive clustering
    defensiveClusters = clusterAndSave(
        role='defensive',
        method=defenseMethod,
        clusters=defenseClusters,
        distance=defenseDistance,
        pcaComponents=defensePcaComponents,
        data=defensiveData
    )

    print("classification accuracy results for offensive roles:")
    classificationAccuracy(offensiveClusters)
    print("classification accuracy results for defensive roles:")
    classificationAccuracy(defensiveClusters)

    print("Clustering completed for both offensive and defensive roles.")



