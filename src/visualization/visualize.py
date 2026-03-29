import matplotlib.pyplot as plt

def plot_model_perfromance(results):
    models = list(results.keys())
    scores = list(results.values())

    plt.bar(models, scores)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()