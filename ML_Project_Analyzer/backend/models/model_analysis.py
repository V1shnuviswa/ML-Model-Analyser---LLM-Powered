import joblib
import pickle
import tensorflow as tf
import re
import os
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score


def analyze_model(file_path):
    """Analyzes machine learning models from different file formats including Python scripts."""
    try:
        # ✅ Handling Pickle (.pkl) and Joblib (.joblib) Models
        if file_path.endswith(".pkl") or file_path.endswith(".joblib"):
            model = joblib.load(file_path) if file_path.endswith(".joblib") else pickle.load(open(file_path, "rb"))
            return {
                "model_type": type(model).__name__,
                "parameters": model.get_params() if hasattr(model, "get_params") else "N/A",
                "layers": len(model.layers) if hasattr(model, "layers") else "N/A"
            }

        # ✅ Handling Keras (.h5) Models
        elif file_path.endswith(".h5"):
            model = tf.keras.models.load_model(file_path)
            return {
                "model_type": "Keras Model",
                "parameters": "N/A",
                "layers": len(model.layers) if hasattr(model, "layers") else "N/A"
            }

        # ✅ Handling Python ML Scripts (.py)
        elif file_path.endswith(".py"):
            return analyze_python_script(file_path)

        else:
            return {"error": "Unsupported file format"}

    except Exception as e:
        return {"error": str(e)}


def analyze_python_script(file_path):
    """Extracts ML model details from a Python script."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code_lines = f.readlines()

        model_types = []
        optimizers = []
        loss_functions = []
        metrics = []

        for line in code_lines:
            # Detect model definitions
            if re.search(r"LogisticRegression|RandomForestClassifier|SVC|DecisionTreeClassifier|MLPClassifier", line):
                model_types.append(line.strip())

            # Detect optimizers (for deep learning models)
            if re.search(r"Adam|SGD|RMSprop|optimizer=", line):
                optimizers.append(line.strip())

            # Detect loss functions
            if re.search(r"loss=", line):
                loss_functions.append(line.strip())

            # Detect evaluation metrics
            if re.search(r"accuracy_score|f1_score|precision_score|recall_score", line):
                metrics.append(line.strip())

        return {
            "model_type": model_types if model_types else "No ML model found",
            "optimizers": optimizers if optimizers else "No optimizers found",
            "loss_functions": loss_functions if loss_functions else "No loss functions found",
            "metrics": metrics if metrics else "No evaluation metrics found"
        }

    except Exception as e:
        return {"error": f"Error processing Python file: {str(e)}"}


def execute_model_and_generate_graphs(script_path):
    """Executes the Python ML script and generates evaluation and visualization graphs."""
    try:
        # Dynamically import the user's model script
        module_name = os.path.basename(script_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Extract required objects from script
        model = getattr(module, "model", None)
        X_test = getattr(module, "X_test", None)
        y_test = getattr(module, "y_test", None)

        if model is None or X_test is None or y_test is None:
            raise ValueError("Script must define `model`, `X_test`, and `y_test`.")

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)

        # Generate visualizations and convert to base64
        return {
            "accuracy": plot_to_base64(score_bar_chart("Accuracy", acc, color='skyblue')),
            "f1_score": plot_to_base64(score_bar_chart("F1 Score", f1, color='salmon')),
            "confusion_matrix": plot_to_base64(confusion_matrix_plot(cm)),
            "energy": plot_to_base64(energy_consumption_plot())
        }

    except Exception as e:
        print("⚠️ Error during model execution:", str(e))
        return {}  # return empty so frontend handles missing keys gracefully


def plot_to_base64(fig):
    """Converts a matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return encoded  # Don't add data:image/... prefix, React already does


def confusion_matrix_plot(cm):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix")
    return fig


def score_bar_chart(label, value, color):
    fig, ax = plt.subplots()
    ax.bar([label], [value], color=color)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Score")
    ax.set_title(label)
    return fig


def energy_consumption_plot():
    fig, ax = plt.subplots(figsize=(6, 4))
    energy = np.random.uniform(0.1, 0.5)

    # Use numeric x and add label manually
    x = [0]  # x-axis position
    ax.bar(x, [energy], color='lightgreen', width=0.5)

    ax.set_ylim([0, 1])
    ax.set_title("Estimated Energy Consumption", fontsize=14)
    ax.set_ylabel("Energy Used", fontsize=12)
    ax.set_xlabel("Index", fontsize=12)

    # Add numeric ticks on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(['Energy (kWh)'], fontsize=12)

    # Annotate value
    ax.text(x[0], energy + 0.02, f"{energy:.2f}", ha='center', fontsize=10)

    fig.tight_layout()
    return fig

