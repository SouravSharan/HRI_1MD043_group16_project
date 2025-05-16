from furhat_remote_api import FurhatRemoteAPI
import time
from ollama import Client
import speech_recognition as sr

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

def recognize_speech(timeout=None, phrase_time_limit=None):
    """Uses SpeechRecognition library to capture and transcribe audio with optional timeout."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for user input...")
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            spoken_text = r.recognize_google(audio)  # You can change the recognizer
            print(f"You said: {spoken_text}")
            return spoken_text
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout.")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service; {e}")
            return None


def create_gesture(name: str, **kwargs) -> dict:
    """
    Creates a gesture dictionary without parameters.

    Args:
        name (str): The name of the gesture.

    Returns:
        dict: A gesture object with 'name' and empty 'parameters'.
    """
    return {"name": name, "parameters": {}}

def perform_gesture(furhat, gesture: dict):
    """
    Requests Furhat to perform the specified gesture.

    Args:
        furhat: FurhatRemoteAPI instance.
        gesture (dict): Gesture dictionary with 'name' and 'parameters'.

    Raises:
        Prints error message if gesture execution fails.
    """
    try:
        furhat.gesture(name=gesture["name"], **gesture["parameters"])
    except Exception as e:
        print(f"[Error] Gesture '{gesture['name']}' failed: {e}")

def select_gesture(response_text: str) -> list:
    """
    Selects a list of gestures based on keywords in the response text.

    Args:
        response_text (str): Text to analyze for selecting gestures.

    Returns:
        list: A list of gesture dictionaries to perform.
    """
    global last_gesture
    response_text = response_text.lower()
    gestures = []

    def maybe_add(name, probability=0.7):
        """
        Adds a gesture to the list based on probability and last gesture check.

        Args:
            name (str): Gesture name to add.
            probability (float): Chance to add the gesture (default 0.7).
        """
        if name != last_gesture and random.random() < probability:
            gestures.append(create_gesture(name))

    def add_if_contains(keywords, gestures_to_add):
        """
        Adds gestures if any keyword matches in the response text.

        Args:
            keywords (list): Keywords to check in the text.
            gestures_to_add (list): Gestures to add if keywords match.

        Returns:
            bool: True if any keyword matched, else False.
        """
        if any(word in response_text for word in keywords):
            gestures.extend(gestures_to_add)
            return True
        return False

    categories = [
        (["happy", "glad", "pleased", "delighted", "good", "great", "awesome", "smile", "thank you", "thanks", "welcome", "joy", "love", "wonderful", "fantastic"], [
            create_gesture("BigSmile"),
            create_gesture("Nod")
        ]),
        (["understand", "empathize", "support", "comfort", "care", "helpful", "together", "kind", "concern", "sympathy"], [
            create_gesture("Smile"),
            create_gesture("Nod")
        ]),
        (["think", "reflect", "consider", "ponder", "thoughtful", "perhaps", "maybe", "question", "explore", "analyze"], [
            create_gesture("Thoughtful"),
            create_gesture("BrowFrown")
        ]),
        (["surprise", "surprised", "oh", "wow", "really", "unexpected", "astonished", "shocked"], [
            create_gesture("Surprise"),
            create_gesture("OpenEyes")
        ]),
        (["sad", "sorry", "unfortunately", "hard", "pain", "loss", "heartbreaking", "difficult", "grief", "tears"], [
            create_gesture("ExpressSad")
        ]),
        (["angry", "frustrated", "annoyed", "disgust", "hate", "irritated", "upset", "mad", "resentful"], [
            create_gesture("ExpressAnger")
        ]),
        (["scared", "fear", "worried", "anxious", "nervous", "concerned", "afraid", "tense"], [
            create_gesture("ExpressFear"),
            create_gesture("Blink")
        ]),
        (["yes", "agree", "right", "correct", "exactly", "definitely", "sure", "absolutely", "nod", "indeed"], [
            create_gesture("Nod"),
            create_gesture("Smile")
        ]),
        (["not sure", "maybe", "perhaps", "doubt", "question", "uncertain", "confused", "hesitate"], [
            create_gesture("BrowFrown"),
            create_gesture("Shake")
        ])
    ]

    matched = any(add_if_contains(keywords, gestures_to_add) for keywords, gestures_to_add in categories)
    if not matched:
        maybe_add("Nod", probability=0.2)

    return gestures

def dynamic_sleep_for_speech(text: str, avg_time_per_char=0.07, min_sleep=1.0, max_sleep=10.0):
    """
    Calculates and executes a dynamic sleep based on text length to simulate speaking time.

    Args:
        text (str): The text that will be spoken.
        avg_time_per_char (float): Average seconds per character (default 0.07).
        min_sleep (float): Minimum sleep time in seconds (default 1.0).
        max_sleep (float): Maximum sleep time in seconds (default 10.0).
    """
    sleep_time = len(text) * avg_time_per_char
    # 최소/최대 범위로 제한
    sleep_time = max(min_sleep, min(sleep_time, max_sleep))
    time.sleep(sleep_time)

def furhat_response(text: str):
    """
    Makes Furhat say the given text, performs related gestures, and waits dynamically.

    Args:
        text (str): The text for Furhat to speak.
    """
    furhat.say(text=text, blocking=False)
    gestures_to_do = select_gesture(text)
    for gesture in gestures_to_do:
        perform_gesture(furhat, gesture)
    dynamic_sleep_for_speech(text)


def run_therapy_session():
    """Runs a therapy session with the Furhat robot powered by an LLM and local ASR."""
    try:
        # Connect to Furhat
        furhat = FurhatRemoteAPI("localhost")  # Replace with robot IP if needed

        # Define the initial prompt for the therapist persona
        therapist_prompt = """You are a kind and empathetic British female therapist.
Respond to the user with understanding and offer helpful reflections.
Keep your responses concise but thoughtful. When the user indicates they want to end the session,
respond appropriately and then the script should exit. Keep your response limited to maximum 2 lines."""
        therapist_bot = LLMInteraction(model_name="llama2-uncensored", initial_prompt=therapist_prompt)

        # Step 1: Greet
        furhat.say(text="Hello there. Welcome. Please, have a seat, if you like.", blocking=True)
        time.sleep(1)
        furhat.gesture(name="Smile")

        # Start the conversation loop
        while True:
            # Step 2: Listen for user input using local ASR with increased timeout (e.g., 10 seconds)
            spoken_text = recognize_speech(timeout=30)  # Increase the timeout here
            print("spoken text: ",spoken_text)
            if spoken_text:
                # Check for exit phrases *before* sending to LLM
                if any(phrase.lower() in spoken_text.lower() for phrase in ["bye", "goodbye", "exit", "quit", "that's all", "end session"]):
                    furhat.say(text="Thank you for our session. Feel free to talk again anytime. Take care.", blocking=True)
                    break

                # Step 3: Get LLM response
                therapist_response = therapist_bot.get_response(spoken_text)
                print(f"Therapist says: {therapist_response}")

                # Step 4: Furhat responds
                furhat_response(therapist_response)
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
