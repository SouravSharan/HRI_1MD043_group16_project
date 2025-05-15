from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import utils
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load MiniLM for embedding
retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load CodeT5
generator_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-770m")

def generate_response(query):
    
    # Create input prompt with retrieved config
    prompt = f"You are a female british therapist talking to your client. Query: {query}"
    
    inputs = generator_tokenizer(prompt, return_tensors="pt")
    output_ids = generator_model.generate(**inputs, max_length=50)
    response = generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return response





while True:
    query = input("Enter your query: ")
    
    if query == "exit":
        break

    response = generate_response(query)

    print(response)


