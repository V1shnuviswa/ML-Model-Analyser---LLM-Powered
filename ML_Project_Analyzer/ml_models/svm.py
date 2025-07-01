from llama_cpp import Llama

MODEL_PATH = "/Users/apple/Desktop/ML_Project_Analyzer/backend/models/llama-2-7b-chat.Q4_K_M.gguf"

# Load LLaMA Model
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)

# Simple test prompt
prompt = "Explain Logistic Regression in machine learning."

response = llm(prompt, max_tokens=200, temperature=0.7)

print("LLaMA Response:", response)
