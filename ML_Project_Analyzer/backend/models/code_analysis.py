import os
import re

def analyze_code(filepath):
    """Extracts key details from an uploaded Python ML script."""
    if not filepath.endswith(".py"):
        return {"error": "Not a Python file."}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code_content = f.readlines()

        # Extracting ML-related features
        imports = [line.strip() for line in code_content if line.startswith("import") or line.startswith("from")]
        functions = [line.strip() for line in code_content if line.strip().startswith("def ")]
        model_definitions = [line.strip() for line in code_content if "model" in line.lower() or "classifier" in line.lower()]
        training_steps = [line.strip() for line in code_content if "fit(" in line.lower()]
        evaluation_metrics = [line.strip() for line in code_content if "score" in line.lower() or "evaluate(" in line.lower()]
        optimizers = [line.strip() for line in code_content if "optimizer=" in line.lower()]
        hyperparameters = [line.strip() for line in code_content if re.search(r"\b(n_estimators|learning_rate|epochs|batch_size)\b", line.lower())]

        # Return structured analysis
        summary = {
            "imports": imports,
            "functions": functions,
            "model_definitions": model_definitions,
            "training_steps": training_steps,
            "evaluation_metrics": evaluation_metrics,
            "optimizers": optimizers,
            "hyperparameters": hyperparameters
        }

        return summary

    except Exception as e:
        return {"error": f"Error processing code: {str(e)}"}