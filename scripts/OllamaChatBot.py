from ollama import Client

class LLMInteraction:
    """
    A class to interact with an Ollama language model, allowing for different personalities
    and tracking of conversations.
    """
    def __init__(self, model_name: str, initial_prompt: str):
        """
        Initializes the LLMInteraction object.

        Args:
            model_name: The name of the Ollama model to use (e.g., 'llama2').
            initial_prompt: The initial instruction or query to set the model's persona.
        """
        self.client = Client()
        self.model_name = model_name
        self.conversation_history = [{"role": "system", "content": initial_prompt}]

    def get_response(self, user_input: str) -> str:
        """
        Sends a user input to the Ollama model and retrieves its response,
        while maintaining the conversation history.

        Args:
            user_input: The user's message to the model.

        Returns:
            The model's response as a string.
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self.client.chat(
            model=self.model_name,
            messages=self.conversation_history,
            stream=False  # For a single response, set stream to False
        )
        model_response = response['message']['content']
        self.conversation_history.append({"role": "assistant", "content": model_response})
        return model_response

# Example usage for your therapy demo:
if __name__ == "__main__":
    therapist_prompt = "You are a kind and empathetic British female therapist. Respond to the user with understanding and offer helpful reflections."
    therapist_bot = LLMInteraction(model_name="llama2-uncensored", initial_prompt=therapist_prompt)

    print("Starting a conversation with the British therapist bot:")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["bye", "goodbye", "exit", "quit"]:
            print("Therapist: Take care, and feel free to reach out again if you need to.")
            break
        therapist_response = therapist_bot.get_response(user_query)
        print(f"Therapist: {therapist_response}")

    # You can create other personalities as needed:
    helpful_assistant_prompt = "You are a helpful and concise assistant."
    assistant_bot = LLMInteraction(model_name="mistral", initial_prompt=helpful_assistant_prompt)
    print("\nStarting a conversation with the helpful assistant bot:")
    print(f"Assistant: {assistant_bot.get_response('What is the capital of Sweden?')}")