import os
import uuid
from models.model_analysis import execute_model_and_generate_graphs  # 👈 Ensure model_analysis.py is importable

def save_uploaded_file(file, upload_folder):
    filename = str(uuid.uuid4()) + "_" + file.filename
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    return file_path

# 🔽 Optional Helper: Get visualizations directly from the saved file
def analyze_and_visualize_file(file_path):
    """
    Helper to get analysis and visualization from a saved Python model script (.py).
    Only runs visualization if it's a Python file.
    """
    if file_path.endswith(".py"):
        return execute_model_and_generate_graphs(file_path)
    else:
        return {"message": "Visualization is only supported for .py scripts."}
