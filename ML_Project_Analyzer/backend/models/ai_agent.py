import json
import logging
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Model Path (Update if needed)
MODEL_PATH = r"C:\Users\Vishnu\Downloads\ML_Project_Analyzer\ML_Project_Analyzer\backend\models\llama-2-7b-chat.Q2_K.gguf"

# Load the LLaMA Model
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,  # Adjusted context size
        chat_format="llama-2",
        verbose=False
    )
    logging.info("✅ LLaMA model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading LLaMA model: {str(e)}")
    llm = None  # Prevent crashes if the model fails

def rule_based_suggestions(code_analysis):
    """Analyzes ML Python scripts and provides rule-based insights."""
    suggestions = []

    imports = code_analysis.get("imports", [])
    functions = code_analysis.get("functions", [])
    model_definitions = code_analysis.get("model_definitions", [])
    training_steps = code_analysis.get("training_steps", [])
    evaluation_metrics = code_analysis.get("evaluation_metrics", [])
    optimizers = code_analysis.get("optimizers", [])
    hyperparameters = code_analysis.get("hyperparameters", [])

    # Check for missing imports
    if not imports:
        suggestions.append("⚠️ No imports found. Ensure you're using the correct ML libraries.")

    # ML Library Recommendations
    if any("tensorflow" in imp.lower() or "keras" in imp.lower() for imp in imports):
        suggestions.append("🔹 Optimize TensorFlow models using mixed precision or XLA compilation.")
    elif any("sklearn" in imp.lower() for imp in imports):
        suggestions.append("🔹 Improve Scikit-learn performance with StandardScaler or MinMaxScaler.")
    elif any("torch" in imp.lower() for imp in imports):
        suggestions.append("🔹 Speed up PyTorch training with CUDA and AMP.")

    # Model Definitions
    if not model_definitions:
        suggestions.append("⚠️ No ML model detected. Ensure you define and initialize a model before training.")

    # Training Steps
    if not training_steps:
        suggestions.append("⚠️ No training process found. Ensure you are calling `model.fit()` or `model.train()`.")

    # Evaluation Metrics
    if not evaluation_metrics:
        suggestions.append("⚠️ No evaluation metrics found. Consider using Accuracy, F1-score, RMSE, or AUC-ROC.")

    # Optimizers
    if not optimizers:
        suggestions.append("⚠️ No optimizer detected. Try Adam, SGD, or RMSprop for better performance.")

    # Hyperparameter Tuning
    if hyperparameters:
        suggestions.append("🎯 Your model has hyperparameters! Tune them using GridSearchCV, RandomizedSearchCV, or Optuna.")

    # Train-Test Split
    if not any("train_test_split" in line for line in imports):
        suggestions.append("🚀 No `train_test_split` found. Ensure you're splitting data into train and test sets.")

    # Feature Engineering
    if not any("StandardScaler" in line or "MinMaxScaler" in line for line in imports):
        suggestions.append("✨ Consider feature scaling to improve model performance.")

    # Data Imbalance Handling
    if "SMOTE" not in str(imports) and "RandomUnderSampler" not in str(imports):
        suggestions.append("📉 If dataset is imbalanced, use `SMOTE` or `RandomUnderSampler`.")

    # Overfitting Prevention
    if "Dropout" not in str(imports) and "L1" not in str(imports):
        suggestions.append("⚡ Prevent overfitting with L1/L2 regularization or dropout layers.")

    # Model Saving
    if "joblib.dump" not in str(functions) and "pickle.dump" not in str(functions):
        suggestions.append("💾 No model saving found. Save models using `joblib.dump()` or `pickle.dump()`.")

    return suggestions if suggestions else ["✅ No issues detected! Your code looks well-structured."]

def generate_ai_suggestions(model_info, code_info):
    """Generate AI-based analysis and suggestions for ML models and code."""
    
    if llm is None:
        return {"error": "⚠️ LLaMA model failed to load. Check logs for details."}

    logging.info(f"📩 Model Info: {model_info}")
    logging.info(f"📩 Code Info: {code_info}")

    # Rule-based insights
    rule_based = rule_based_suggestions(code_info)
    rule_suggestions_str = "\n".join(rule_based)

    # AI Model Prompt (Structured Format)
    prompt = f"""
    You are an expert AI analyzing an ML project. Provide structured suggestions.

    Improvements for the Model: 
    1. Feature Scaling - Use techniques like StandardScaler or MinMaxScaler to scale features.  
    2. Data Augmentation - Implement data augmentation techniques to generate new training samples.  
    3. Regularization - Use L1/L2 regularization or dropout to prevent overfitting.  

    Code Optimizations:  
    1. Remove Unused Imports - Cleaning unnecessary imports improves readability.  
    2. Simplify Code - Remove unnecessary complexity for better maintainability.  
    3. Use Efficient Libraries - Utilize `joblib` or `pickle` for model persistence.  

    Alternative ML Models:
    1. Decision Tree-Based Models - Quick training, effective for missing data handling.  
    2. Neural Network Models** - Handles complex feature relationships.  
    3. Gradient Boosting Models - Combine weak models for improved performance.  

    Final Suggestions:
    1. Run Additional Experiments- Test different variations of the model.  
    2. Visualize Data - Gain better insights by analyzing feature distributions.  

    Additional Rule-Based Findings:  
    {rule_suggestions_str}
    """

    try:
        logging.info("🔍 Sending Prompt to LLaMA...")

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700,
            temperature=0.7
        )

        if isinstance(response, dict) and "choices" in response and response["choices"]:
            ai_response = response["choices"][0].get("message", {}).get("content", "").strip()
        else:
            ai_response = "⚠️ LLaMA did not return a valid response."

        logging.info("✅ AI Response Received!")

    except Exception as e:
        logging.error(f"❌ LLaMA Error: {str(e)}")
        ai_response = f"⚠️ LLaMA Error: {str(e)}"

    return {
        "rule_based_suggestions": rule_based,
        "ai_suggestions": ai_response.split("\n")  # Return structured response as a list
    }
