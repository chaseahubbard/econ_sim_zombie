class Skill:
    def __init__(self, name, level=0, experience=0):
        self.name = name
        self.level = level
        self.experience = experience
        self.next_level_exp = 100  # Base experience for next level

    def gain_experience(self, amount):
        self.experience += amount
        while self.experience >= self.next_level_exp:
            self.level_up()

    def level_up(self):
        self.level += 1
        self.experience -= self.next_level_exp
        self.next_level_exp = int(self.next_level_exp * 1.5)
