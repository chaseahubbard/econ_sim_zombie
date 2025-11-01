import numpy as np

class Job:
    def __init__(self, name, position, reward, required_skills,min_ability = 0, training_job=False):
        self.name = name
        self.position = np.array(position)
        self.reward = reward  # Can be positive (paying job) or negative (training)
        self.required_skills = required_skills  # Dictionary of skill names and minimum levels
        self.training_job = training_job  # Whether this is a training job
        self.experience_reward = {skill: 20 for skill in required_skills}  # Base experience gain
        self.completion_time = 5  # Base time to complete job
        self.min_ability = min_ability
        self.machine_compatible = True  # Whether machines can help with this job
