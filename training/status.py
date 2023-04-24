import os
import json

# Manager for the current status of the optimization.
class TrainingStatus:
    def __init__(self, load=True):
        if load and os.path.exists("training-status.json"):
            with open("training-status.json", "r") as status_file:
                self.status = json.load(status_file)
                status_file.close()
        else:
            self.status = {'iteration': -1, 'losses': {}}

    def save(self):
        with open("training-status.json", "w") as status_file:
            json.dump(self.status, status_file)
            status_file.close()

    def close(self):
        if os.path.exists("training-status.json"):
            os.remove("training-status.json")