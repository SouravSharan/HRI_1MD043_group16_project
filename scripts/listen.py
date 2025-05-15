# In your Furhat skill's main file
from furhat_skill import *

class MySkill(Skill):
    def __init__(self):
        super().__init__()

    @sits
    def idle(self, event):
        return self.goto_state(self.listening)

    @state
    def listening(self, event):
        if event.type == 'UserSpeech':
            print(f"Skill heard: {event.text}")
            return self.goto_state(self.idle)

if __name__ == '__main__':
    MySkill().run()