from llama_cpp import Llama

MODEL_PATH = r"C:\Users\Vishnu\Downloads\ML_Project_Analyzer\ML_Project_Analyzer\backend\models\llama-2-7b-chat.Q2_K.gguf"

# Load LLaMA model using llama.cpp
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,  # Increase context size for better responses
    chat_format="llama-2",
    verbose=True
)

def fine_tune(dataset_path):
    """Simulated fine-tuning using llama.cpp (actual fine-tuning not supported)."""
    print("⚠️ llama.cpp does NOT support fine-tuning directly!")
    print("Consider converting GGUF to HF format or using LoRA adapters.")

    # Simulating inference (NOT training)
    test_prompt = "Explain overfitting in machine learning."
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": test_prompt}],
        max_tokens=200
    )

    print("🧠 Sample Response:", response["choices"][0]["message"]["content"])

