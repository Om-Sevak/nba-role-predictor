import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from requests.exceptions import ReadTimeout

# you can change the season here if you want in the same format as '2023-24'
# keep in mind new data extraction takes about an hour (on a good day)
seasonData = '2023-24'
maxRetries = 5
retryDelay = 5  

# function to fetch data. needed to put exponential backoff retries because we had a lot of trouble with timeouts
def fetchDataWithRetries(func, *args, **kwargs):
    retries = 0
    while retries < maxRetries:
        try:
            return func(*args, **kwargs)
        except ReadTimeout:
            retries += 1
            print(f"ReadTimeout error encountered. Retrying {retries}/{maxRetries}...")
            time.sleep(retryDelay * (2 ** retries)) 
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    return None 

# takes a dataframe and prints out all the clusters with the players in the cluster
def printPlayerCluster(df, role):
    clusters = df['Cluster'].unique()
    clusters.sort()

    for cluster in clusters:
        playersInCluster = df[df['Cluster'] == cluster]['PLAYER_NAME']
        print(f"{role} cluster {cluster}:")
        print(playersInCluster.tolist())
        print("\n" + "-" * 50 + "\n")

# prints out an elbow function for kmeans clustering
def elbowFunction(df):
    wcss = []
    kValues = range(1, 15)
    for k in kValues:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(kValues, wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()

# makes the data numeric so the player name and other columns that are string gets thrown out. also scales the data
def dataFrameScale(df):
    numericDf = df.select_dtypes(include=['number'])
    numericDf = numericDf.drop(columns=['PLAYER_ID'], errors='ignore')

    imputer = SimpleImputer(strategy='mean')
    numericDfImputed = imputer.fit_transform(numericDf)

    scaler = StandardScaler()
    statsScaled = scaler.fit_transform(numericDfImputed)

    return statsScaled, numericDf

# preform kmean clustering based on how many clusters you want
def kMeansCluster(df, clusters):
    scaled, num = dataFrameScale(df)
    elbowFunction(scaled)
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)
    return df
