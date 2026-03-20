import matplotlib.pyplot as plt
import os


def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss Curve")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/loss_curve.png")
    plt.close()


def plot_predictions(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")
    plt.title("Predicted vs Actual")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/predictions_vs_actual.png")
    plt.close()


def plot_feature_vs_target(X, y, feature_index, feature_name):
    plt.figure()
    plt.scatter(X[:, feature_index], y)
    plt.xlabel(feature_name)
    plt.ylabel("Exam Score")
    plt.title(f"{feature_name} vs Exam Score")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{feature_name}_vs_target.png")
    plt.close()