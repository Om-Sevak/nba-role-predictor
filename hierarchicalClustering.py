from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# preforms hierarchical clustering based on how much distance you want between clusters
def hierarchicalCluster(df, distance):

    numericDf = df.select_dtypes(include=['number'])
    numericDf = numericDf.drop(columns=['PLAYER_ID'], errors='ignore')

    imputer = SimpleImputer(strategy='mean')
    numericDfImputed = imputer.fit_transform(numericDf)

    scaler = StandardScaler()
    statsScaled = scaler.fit_transform(numericDfImputed)

    Z = linkage(statsScaled, method='ward')

    # plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Players')
    plt.ylabel('Distance')
    plt.show()

    maxDistance = distance  
    clusters = fcluster(Z, maxDistance, criterion='distance')
    df['Cluster'] = clusters

    return df
