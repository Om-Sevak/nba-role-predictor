from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def classificationAccuracy(data, role):
    # Separate features and target variable
    if role == 'offensive':
        features = data.drop(columns=['Cluster', 'PLAYER_NAME', 'PLAYER_ID'])
    else:
        features = data.drop(columns=['Cluster', 'PLAYER_NAME', 'PLAYER_ID']) 
    target = data['Cluster'] 

    model = RandomForestClassifier(random_state=42)

    # Perform 10-fold cross-validation
    cv_scores = cross_val_score(model, features, target, cv=10, scoring='accuracy')

    # Print the cross-validation scores and the average accuracy
    print("Cross-validation scores for each fold:", cv_scores)
    print("Average accuracy across all folds:", cv_scores.mean())