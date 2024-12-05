import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from requests.exceptions import ReadTimeout
seasonData = '2023-24'
# Load data from the API
max_retries = 5
retry_delay = 5  # seconds

# Function to fetch data with retry and exponential backoff
def fetch_data_with_retries(func, *args, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            # Call the API function and get data
            return func(*args, **kwargs)
        except ReadTimeout:
            retries += 1
            print(f"ReadTimeout error encountered. Retrying {retries}/{max_retries}...")
            time.sleep(retry_delay * (2 ** retries))  # Exponential backoff
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    return None 

def printPlayerCluster(fileName, role):
    df = pd.read_csv(fileName)

    clusters = df['Cluster'].unique()
    clusters.sort()

    for cluster in clusters:
        players_in_cluster = df[df['Cluster'] == cluster]['PLAYER_NAME']
        print(f"{role} cluster {cluster}:")
        print(players_in_cluster.tolist())
        print("\n" + "-" * 50 + "\n")

def elbowFunction(df):
    wcss = []
    k_values = range(1, 15)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()

def dataFrameScale(df):
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df = numeric_df.drop(columns=['PLAYER_ID'], errors='ignore')

    imputer = SimpleImputer(strategy='mean')
    numeric_df_imputed = imputer.fit_transform(numeric_df)

    scaler = StandardScaler()
    stats_scaled = scaler.fit_transform(numeric_df_imputed)

    return stats_scaled, numeric_df

def kMeansCluster(df, clusters):
    scaled,num = dataFrameScale(df)
    elbowFunction(scaled)
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)
    return df