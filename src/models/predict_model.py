from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(models, X_test, y_test):

    for name, model in models.items():

        preds = model.predict(X_test)

        print(f"\n{name}")
        print("Accuracy:", accuracy_score(y_test, preds))
        print("Precision:", precision_score(y_test, preds))
        print("Recall:", recall_score(y_test, preds))
        print("F1:", f1_score(y_test, preds))
