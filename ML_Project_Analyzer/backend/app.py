from flask import Flask, request, jsonify, send_file 
from flask_cors import CORS
import os
import logging
import atexit
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from llama_cpp import Llama  # Import LLaMA for local inference
from waitress import serve  # Use waitress for better Windows compatibility
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mlxtend.plotting import plot_decision_regions
import pandas as pd
import io
import uuid
import base64
from utils.file_utils import save_uploaded_file
from models.model_analysis import analyze_model, execute_model_and_generate_graphs
from models.code_analysis import analyze_code
from models.ai_agent import generate_ai_suggestions
from utils.file_utils import save_uploaded_file 
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)

# Load LLaMA model
LLAMA_MODEL_PATH = r"C:\Users\Vishnu\Downloads\ML_Project_Analyzer\ML_Project_Analyzer\backend\models\llama-2-7b-chat.Q2_K.gguf"

try:
    llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=1024)  # Reduced n_ctx for stability
    logging.info("✅ LLaMA Model Loaded Successfully!")
except Exception as e:
    logging.error(f"❌ Failed to load LLaMA model: {str(e)}")
    llm = None

# Cleanup function for proper shutdown
def cleanup():
    global llm
    if llm:
        del llm  # Free model memory
        llm = None
        logging.info("🧹 Cleaned up LLaMA model")

atexit.register(cleanup)  # Ensure cleanup happens on exit

# Store latest results for generating PDFs and searching
latest_results = {"model_analysis": {}, "code_analysis": {}, "ai_suggestions": {}}

@app.route("/", methods=["GET"])
def home():
    """Root endpoint to check if the API is running."""
    return jsonify({"message": "✅ ML Project Analyzer API is running with LLaMA 2!"}), 200

# Sample Dataset
np.random.seed(42)
X_train = pd.DataFrame(np.random.rand(100, 2), columns=["Feature1", "Feature2"])
y_train = np.random.randint(0, 2, size=100)

@app.route("/visualize_model", methods=["GET"])
def visualize_model():
    """Visualizes a model's behavior (Decision Tree, Random Forest, KNN, etc.)."""
    model_name = request.args.get("model_name", "").lower()
    fig, ax = plt.subplots(figsize=(8, 6))

    if model_name == "decision tree":
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)
        plot_tree(model, feature_names=X_train.columns, class_names=["Class 0", "Class 1"], filled=True, ax=ax)
        plt.title("Decision Tree Visualization")
    elif model_name == "random forest":
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sns.barplot(x=importances[indices], y=X_train.columns[indices], ax=ax)
        plt.title("Feature Importance in Random Forest")
    elif model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train.values, y_train)
        plot_decision_regions(X_train.values, y_train, clf=model, legend=2, ax=ax)
        plt.title("KNN Decision Boundaries")
    elif model_name == "svm":
        model = SVC(kernel="linear", probability=True)
        model.fit(X_train.values, y_train)
        plot_decision_regions(X_train.values, y_train, clf=model, legend=2, ax=ax)
        plt.title("SVM Decision Boundaries")
    elif model_name == "pca":
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette="coolwarm", ax=ax)
        plt.title("PCA Visualization")
    elif model_name == "kmeans":
        model = KMeans(n_clusters=3, random_state=42)
        y_kmeans = model.fit_predict(X_train)
        sns.scatterplot(x=X_train.iloc[:, 0], y=X_train.iloc[:, 1], hue=y_kmeans, palette="viridis", ax=ax)
        plt.title("K-Means Clustering Visualization")
    else:
        return jsonify({"error": f"Visualization for {model_name} is not available yet."}), 400

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    plt.close(fig)
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")
    return jsonify({"image": img_base64})


@app.route('/upload-model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({"error": "No file part"}), 400

    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    filepath = os.path.join('uploads', model_file.filename)
    model_file.save(filepath)

    # Assuming the model is a scikit-learn model saved as a .pkl file
    model = joblib.load(filepath)

    # Simulate model prediction on test data (for demonstration)
    # Replace this with actual data
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = model.predict(np.random.rand(5, 10))  # Example test data

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix plot as an image
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(cm)), labels=np.unique(y_true))
    plt.yticks(np.arange(len(cm)), labels=np.unique(y_true))
    plt.savefig("uploads/confusion_matrix.png")
    plt.close()

    # Calculate model metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    # Send data back to frontend
    response = {
        "confusionMatrixUrl": "uploads/confusion_matrix.png",
        "accuracy": accuracy,
        "precision": precision
    }

    return jsonify(response)



@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file uploads and triggers ML and code analysis."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filepath = save_uploaded_file(file, app.config["UPLOAD_FOLDER"])
        logging.info(f"📂 File uploaded: {filepath}")

        model_results = analyze_model(filepath)
        code_results = analyze_code(filepath)
        ai_suggestions = generate_ai_suggestions(model_results, code_results)

        # Store the latest results
        latest_results.update({
            "model_analysis": model_results,
            "code_analysis": code_results,
            "ai_suggestions": ai_suggestions
        })

        return jsonify({
            "model_analysis": model_results,
            "code_analysis": code_results,
            "ai_suggestions": ai_suggestions
        })

    except Exception as e:
        logging.error(f"❌ Error in /upload: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
def generate_graphs(filepath):
    """Generates model performance graphs (e.g., confusion matrix, feature importance)."""
    # Example: Assuming the model is in the filepath and can be loaded using joblib
    model = joblib.load(filepath)  # You can customize this to match how the model is saved

    # Assuming X_test and y_test are the test dataset
    X_test = pd.DataFrame(np.random.rand(20, 2), columns=["Feature1", "Feature2"])  # Example test data
    y_test = np.random.randint(0, 2, size=20)  # Example test labels

    # Confusion matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Save confusion matrix as base64 string
    cm_bytes = io.BytesIO()
    plt.savefig(cm_bytes, format="png")
    cm_bytes.seek(0)
    cm_base64 = base64.b64encode(cm_bytes.read()).decode("utf-8")

    # You can add more graphs like feature importance, performance metrics, etc.
    return {
        "confusion_matrix": cm_base64,
        # Add other graphs if needed
    }

@app.route("/search", methods=["POST"])
def search():
    """Handles user questions related to the uploaded model analysis."""
    if llm is None:
        return jsonify({"error": "LLaMA model failed to load."}), 500

    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Invalid request. Provide a 'question'"}), 400

        question = data["question"].strip()
        logging.info(f"🔎 User Question: {question}")

        # Prepare context from latest analysis
        context = "\n".join([f"{key}: {str(value)[:500]}" for key, value in latest_results.items()])

        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

        # Limit token usage safely
        response = llm(
            prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stop=["\n", "Question:"]
        )

        answer = response["choices"][0].get("text", "No answer generated.").strip() if response else "No answer generated."

        return jsonify({"question": question, "answer": answer})

    except Exception as e:
        logging.error(f"❌ Error in /search: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/generate_pdf", methods=["GET"])
def generate_pdf():
    """Generates a detailed PDF report of the latest ML Project Analysis results."""
    try:
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle("ML Project Analysis Report")

        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(100, 750, "📄 ML Project Analysis Report")

        pdf.setFont("Helvetica", 12)
        y_position = 720

        for section, content in latest_results.items():
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(80, y_position, f"{section.replace('_', ' ').title()}:")
            y_position -= 20
            pdf.setFont("Helvetica", 10)

            text = str(content)  # Convert content to string
            wrapped_text = textwrap.wrap(text, width=100)  # Wrap long lines for readability

            for line in wrapped_text:
                pdf.drawString(100, y_position, line)
                y_position -= 15  # Adjust spacing for readability

                # Add new page if reaching the bottom
                if y_position < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 10)
                    y_position = 750

            y_position -= 20  # Extra space between sections

        pdf.save()
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name="ML_Project_Analysis.pdf", mimetype="application/pdf")

    except Exception as e:
        logging.error(f"❌ Error generating PDF: {str(e)}")
        return jsonify({"error": "Failed to generate PDF", "details": str(e)}), 500

@app.route('/model-evaluation', methods=['POST'])
def evaluate_model():
    try:
        file = request.files['file']
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        analysis = analyze_model(filepath)
        visualization = execute_model_and_generate_graphs(filepath)

        # Check if visualization is valid and not empty
        if not visualization or "error" in visualization:
            return jsonify({"error": "Visualization generation failed."}), 500

        return jsonify({
            "analysis": analysis,
            "visualization": visualization
        })

    except Exception as e:
        print(f"🔥 Server Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.info("🚀 Starting ML Project Analyzer API...")
    serve(app, host="0.0.0.0", port=5000)  # Using waitress for better stability