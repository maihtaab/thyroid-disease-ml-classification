from preprocessing_data.pre_processing import load_data, preprocess
from models.train_model import train_models
from models.predict_model import evaluate_models
from sklearn.model_selection import train_test_split

def main():
    df = load_data("src/data/raw/allhyper.data")
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()