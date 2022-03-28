import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    auc,
    precision_recall_curve,
    roc_curve,
)
import seaborn as sn
import matplotlib.pyplot as plt


def write_results(model_name: str, metrics: dict, figures: dict) -> None:
    """Writes results to the results directory

    Args:
        model_name: Name of the model (this will be the name of the
          subfolder in the results directory)
        metrics: Dictionary of metrics to be saved
        figures: Dictionary of matplotlib figures to be saved
    """

    directory = f"results/{model_name}"
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(f"{directory}/metrics.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(key + ": " + str(value) + "\n")
        f.close()
    for name, fig in figures.items():
        fig.savefig(f"{directory}/{name}", dpi=150)


def get_plot(x: list, y: list, xlabel: str, ylabel: str,
             title: str) -> plt.figure:
    """Generates a matplotlib plot.

    Args:
        x: List of x-values.
        y: List of y-values.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Title of the plot.

    Returns:
        A matplotlib figure with the plotted data.
    """
    fig, ax = plt.subplots(facecolor="white")
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray,
                    y_pred_probas: np.ndarray) -> (dict, dict):
    """Computes metrics and figures for evaluating binary predictions

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        y_pred_probas: Matrix where each row contains the probabilities
          of the sample belonging to each class.

    Returns:
        Relevant metrics and matplotlib figures in dictionary form,
        with keys representing the name of the metric/figure and
        values containing the data.
    """

    f1 = f1_score(y_true, y_pred, average="binary")
    auroc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probas[:, 1])
    prc = get_plot(recall, precision, "Recall", "Precision",
                   "Precision-Recall Curve")
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probas[:, 1])
    roc = get_plot(
        fpr,
        tpr,
        "False Positive Rate",
        "True Positive Rate",
        "Receiver Operating Characteristic",
    )
    metrics = {
        "F1 (binary)": f1,
        "AUROC": auroc,
        "AUPRC": auprc,
    }

    figures = {
        "precision_recall": prc,
        "roc": roc,
    }

    return metrics, figures


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray):
    f1 = f1_score(y_true, y_pred, average="macro")
    metrics = {
        "F1 (macro)": f1,
    }
    figures = {}

    return metrics, figures


def get_heatmap(data):
    fig, ax = plt.subplots(facecolor="white")

    cm_heatmap = sn.heatmap(data, annot=True, fmt="g", cmap="Greens", ax=ax)
    cm_heatmap.set(
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion matrix",
    )

    return fig


def evaluate(model_name: str, y_pred_probas: np.ndarray, y_true: np.ndarray,
             save_results: bool) -> None:
    """Computes evaluation metrics for given predictions.

    Args:
        model_name: Name of the model (this will be the name of the
          subfolder in the results directory)
        y_pred_probas: Matrix where each row contains the probabilities
          of the sample belonging to each class.
        y_true: Array of true labels.
        save_results: Saves results if set to True.
    """

    n_classes = y_pred_probas.shape[1]

    # Converts [0.1, 0.2] -> [[0.9, 0.1], [0.8, 0.2]]
    if n_classes == 1:
        y_pred_probas = np.array([[1 - v, v] for v in y_pred_probas])

    # Probabilities -> class predictions
    y_pred = np.argmax(y_pred_probas, axis=1)

    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    cm_norm = np.around(cm_norm, 3)
    cm = confusion_matrix(y_true, y_pred)

    cm_heatmap = get_heatmap(cm)
    cm_heatmap_norm = get_heatmap(cm_norm)

    acc = accuracy_score(y_true, y_pred)

    if n_classes == 2 or n_classes == 1:
        metrics, figures = evaluate_binary(y_true, y_pred, y_pred_probas)
    else:
        metrics, figures = evaluate_multiclass(y_true, y_pred)

    metrics["Accuracy"] = acc
    figures["confusion_matrix"] = cm_heatmap
    figures["confusion_matrix_norm"] = cm_heatmap_norm

    if save_results:
        write_results(model_name, metrics, figures)

    print(metrics)


# Example use
if __name__ == "__main__":
    from mock.sample_predictions import MULTICLASS_TEST_PREDS, BINARY_TEST_PREDS

    y_pred_probas = MULTICLASS_TEST_PREDS["y_pred_probas"]
    y_true = MULTICLASS_TEST_PREDS["y_true"]

    evaluate("test_multiclass", y_pred_probas, y_true, save_results=True)

    plt.clf()

    y_pred_probas = BINARY_TEST_PREDS["y_pred_probas"]
    y_true = BINARY_TEST_PREDS["y_true"]

    evaluate("test_binary", y_pred_probas, y_true, save_results=True)
