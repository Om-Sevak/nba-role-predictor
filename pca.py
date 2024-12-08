from utils import seasonData, dataFrameScale
import pandas as pd
from sklearn.decomposition import PCA

# Function that gives the PCA values when given the data and the amount of components we want
def pcaCluster(df, components):

    statsScaled, numericDf = dataFrameScale(df)
    pca = PCA(n_components=components)
    dfPca = pca.fit_transform(statsScaled)

    pcaDf = pd.DataFrame(
        dfPca,
        columns=[f"PC{i+1}" for i in range(dfPca.shape[1])]
    )
    pcaDf['PLAYER_ID'] = df['PLAYER_ID']
    pcaDf['PLAYER_NAME'] = df['PLAYER_NAME']

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    loadings = pd.DataFrame(
        pca.components_.T,  
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=numericDf.columns 
    )

    for i in range(pca.n_components_):
        component = loadings.iloc[:, i] 
        sortedComponent = component.reindex(component.abs().sort_values(ascending=False).index)
        print(f"\nTop features for Principal Component {i+1}:")
        print(sortedComponent.head(20))

    return pcaDf
