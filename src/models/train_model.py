from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):

    models = {}

    models["logistic"] = LogisticRegression(max_iter=1000)
    models["decision_tree"] = DecisionTreeClassifier()
    models["random_forest"] = RandomForestClassifier()

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models
