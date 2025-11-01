class Market:
    def __init__(self):
        self.agents = []
        self.goods = []
        self.jobs = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_good(self, good):
        self.goods.append(good)

    def add_job(self, jobs):
        self.jobs.append(jobs)

    def __str__(self):
        return "\n".join(str(agent) for agent in self.agents)
