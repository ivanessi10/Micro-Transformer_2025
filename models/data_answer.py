from llama_cpp import Llama
from config import MODEL_FILE

class GemmaModel:
    def __init__(self):
        self.llm = Llama(
            model_path=MODEL_FILE,
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=-1,
            chat_format="gemma",
            verbose=False
        )

    def generate_response(self, prompt: str) -> str:
        output = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            stop=["<end_of_turn>"],
        )
        
        return output['choices'][0]['message']['content'].strip()
