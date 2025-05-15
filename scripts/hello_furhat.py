from furhat_remote_api import FurhatRemoteAPI
import time

# Connect to Furhat
furhat = FurhatRemoteAPI("localhost")  # Replace with robot IP if needed

# Step 1: Greet
furhat.say(text="Hey! It's so good to see you again.")
time.sleep(1)
furhat.gesture(name="Smile")

# Step 2: Ask about the week
furhat.say(text="So tell me, how's your week been? Anything interesting or stressful?")
response = furhat.listen()

# Step 3: Respond with empathy
furhat.say(text="Hmm, that sounds important.")
time.sleep(1)
furhat.say(text="Want to talk more about it?")
followup = furhat.listen()

# Step 4: React and pivot
furhat.say(text="I always enjoy our chats. You're really good at handling things, you know.")
time.sleep(1)
furhat.gesture(name="Nod")
furhat.say(text="Do you have any plans for the weekend?")
weekend_plans = furhat.listen()

# Step 5: Wrap up warmly
furhat.say(text="That sounds awesome! Whatever you do, make sure to take some time for yourself.")
furhat.gesture(name="BigSmile")
time.sleep(1)
furhat.say(text="Okay, I’ll be right here if you want to chat more later. Take care!")


