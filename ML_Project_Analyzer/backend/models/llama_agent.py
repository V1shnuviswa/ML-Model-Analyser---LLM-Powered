from llama_cpp import Llama  

# Load LLaMA 2 GGUF Model
llm = Llama(model_path=r"C:\Users\Vishnu\Downloads\ML_Project_Analyzer\ML_Project_Analyzer\backend\models\llama-2-7b-chat.Q2_K.gguf", n_ctx=1024)

def analyze_ml_project(query):
    """Generates a response using the LLaMA model."""
    max_input_length = 512  # Ensure context doesn't exceed limit
    trimmed_query = query[:max_input_length]

    response = llm.create_completion(
        prompt=trimmed_query, 
        max_tokens=100  
    )
    
    return response
