from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# preforms classification accuracy test based on Random Forest Classifier algorithm
def classificationAccuracy(data):
    features = data.drop(columns=['Cluster', 'PLAYER_NAME', 'PLAYER_ID']) 
    target = data['Cluster'] 

    model = RandomForestClassifier(random_state=42)

    cvScores = cross_val_score(model, features, target, cv=10, scoring='accuracy')
    print("Cross-validation scores for each fold:", cvScores)
    print("Average accuracy across all folds:", cvScores.mean())
