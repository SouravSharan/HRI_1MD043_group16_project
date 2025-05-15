from furhat_remote_api import FurhatRemoteAPI
import time
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
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=self.conversation_history,
                stream=False
            )
            model_response = response['message']['content']
            self.conversation_history.append({"role": "assistant", "content": model_response})
            return model_response
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            return "I'm having a little trouble processing right now. Could you please repeat that?"

def run_therapy_session():
    """Runs a therapy session with the Furhat robot powered by an LLM."""
    try:
        # Connect to Furhat
        furhat = FurhatRemoteAPI("localhost")  # Replace with robot IP if needed

        # Define the initial prompt for the therapist persona
        therapist_prompt = """You are a kind and empathetic British female therapist.
Respond to the user with understanding and offer helpful reflections.
Keep your responses concise but thoughtful. When the user indicates they want to end the session,
respond appropriately and then the script should exit."""
        therapist_bot = LLMInteraction(model_name="llama2-uncensored", initial_prompt=therapist_prompt)

        # Step 1: Greet
        furhat.say(text="Hello there. Welcome. Please, have a seat, if you like.", blocking=True)
        time.sleep(1)
        furhat.gesture(name="Smile")

        # Start the conversation loop
        while True:
            # Step 2: Listen for user input
            print("Furhat is listening...")
            response = furhat.listen()

            print(f"Furhat listen response: {response}")  # Print the response for inspection

            if response and isinstance(response, dict) and response.get("success") and response.get("message"):
                spoken_text = response["message"]
                print(f"User said: {spoken_text}")

                # Check for exit phrases *before* sending to LLM
                if any(phrase.lower() in spoken_text.lower() for phrase in ["bye", "goodbye", "exit", "quit", "that's all", "end session"]):
                    furhat.say(text="Thank you for our session. Feel free to talk again anytime. Take care.", blocking=True)
                    break

                # Step 3: Get LLM response
                therapist_response = therapist_bot.get_response(spoken_text)
                print(f"Therapist says: {therapist_response}")

                # Step 4: Furhat responds
                furhat.say(text=therapist_response, blocking=True)
                time.sleep(1)

            else:
                furhat.say(text="I didn't quite catch that. Could you please say it again?", blocking=True)
                time.sleep(1)

    except Exception as e:
        print(f"An error occurred during the Furhat session: {e}")
    finally:
        print("Therapy session ended.")

if __name__ == "__main__":
    run_therapy_session()