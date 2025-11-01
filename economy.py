import random
import numpy as np

from market import Market
from agent import EconomicAgent
from job import Job
from good import Good
from machine import Machine
from dynamic_ai import DynamicAI

class Economy:
    def __init__(self):
        self.market = Market()
        self.good_types = {
            'good1': 'necessary',
            'good2': 'necessary',
            'good3': 'luxury'
        }
        self.good_types = {
            'good1': 'necessary',
            'good2': 'necessary',
            'good3': 'luxury'
        }
        self.jobs = []
        self.machines = []
        self.training_jobs = []
        print("Market Open")
        self.ai = DynamicAI(self.market, self.good_types)
        
    def add_machine(self, machine, owner_name= None):
        self.machines.append(machine)
        print(f"New machine {machine.name} has been added to the economy!")
        if owner_name:
            agent = next((a for a in self.market.agents if a.name == owner_name), None)
            if agent is not None:
                agent.acquire_machine(machine)
            else:
                print(f"Owner '{owner_name}' not found among agents.")

    def simulate(self, steps):
        for _ in range(steps):
            self.move_agents()
            self.acquire_goods()
            self.buy_machines()
            self.process_jobs()
            self.maintain_machines()
            self.ai.run_market_cycle()
            
    def process_jobs(self):
        for agent in self.market.agents:
            if agent.current_job:
                job_completed = agent.perform_job(agent.current_job)
                if job_completed:
                    agent.current_job = None
            else:
                # Choose new job based on skills and wealth
                available_jobs = self.jobs + self.training_jobs
                suitable_jobs = [j for j in available_jobs 
                               if agent.can_perform_job(j)]
                
                if suitable_jobs:
                    if agent.wealth < 50:  # Low on money, prioritize paying jobs
                        paying_jobs = [j for j in suitable_jobs 
                                     if j.reward > 0]
                        if paying_jobs:
                            agent.current_job = random.choice(paying_jobs)
                    else:  # Can afford training
                        agent.current_job = random.choice(suitable_jobs)
                        
    def maintain_machines(self):
        for agent in self.market.agents:
            agent.maintain_machines()

    def add_agent(self, name, initial_wealth, position, values_1, initial_inventory=None):
        agent = EconomicAgent(name, initial_wealth, position, values_1)
        if initial_inventory:
            agent.inventory.update(initial_inventory)
        self.market.add_agent(agent)

    def add_good(self, name, position, good_type):
        good = Good(name, position)
        self.market.add_good(good)
        self.good_types[name] = good_type  # Assign type (luxury or necessary)

 

    def add_job(self, name, position, reward, required_skills, training_job=False):
        job = Job(name, position, reward, required_skills,training_job)
        self.jobs.append(job)

    def buy_machines(self):
    #Example: let a random agent buy any unowned machine if they can afford it.
        unowned_machines = [m for m in self.machines if m.owner is None]
        if not unowned_machines:
            return  # Nothing to buy

        buyer = random.choice(self.market.agents)
        machine = random.choice(unowned_machines)

        price = 50  # Some price or formula
        if buyer.wealth >= price:
            buyer.spend(price)
            buyer.acquire_machine(machine)  # <-- This calls agent.acquire_machine(...)
            print(f"{buyer.name} bought {machine.name} for {price}.")
        else:
            print(f"{buyer.name} cannot afford {machine.name}.")

    def move_agents(self):
        for agent in self.market.agents:
            agent.choose_target(self.market.agents, self.market.goods, self.jobs)
            if agent.target:
                agent.move_towards(agent.target[0])

    def acquire_goods(self):
        for agent in self.market.agents:
            if agent.target and agent.target[1] != 'agent':
                good = next((g for g in self.market.goods if g.name == agent.target[1] and np.array_equal(g.position, agent.target[0])), None)
                if good:
                    if not agent.collecting:
                        agent.collecting = True
                        agent.collecting_steps = 5  # Reduced steps to collect for quicker simulation
                    else:
                        agent.collecting_steps -= 1
                        if agent.collecting_steps <= 0:
                            agent.pick_up_good(good.name)
                            self.market.goods.remove(good)
                            agent.collecting = False
                            print(f"{agent.name} acquired {good.name}")

    def plot_market(self, ax):
        ax.clear()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        # For demonstration, we'll use simple markers instead of images
        for agent in self.market.agents:
            x, y = agent.position
            size = max(agent.wealth / 10, 5)  # Size proportional to wealth
            if agent.collecting:
                # Shake the agent position a bit
                x += random.uniform(-0.2, 0.2)
                y += random.uniform(-0.2, 0.2)
            ax.plot(x, y, 'bo', markersize=size)
            ax.text(x, y, f'{agent.name}', size=10, zorder=1, color='k')

        for good in self.market.goods:
            x, y = good.position
            ax.plot(x, y, 'rs', markersize=10)
            ax.text(x, y, f'{good.name}', size=10, zorder=1, color='r')

        for jobs in self.market.jobs:
            x, y = jobs.position
            ax.plot(x, y, 'g^', markersize=10)
            ax.text(x, y, f'{jobs.name}', size=10, zorder=1, color='g')

        for machine in self.machines:
            if machine.owner is None:
                # Machine is unowned: plot at machine.position
                x, y = machine.position
                # Let's use a purple diamond
                ax.plot(x, y, marker='D', color='purple', markersize=10)
                ax.text(x, y, f'{machine.name}', size=10, zorder=1, color='purple')
            else:
                # Machine is owned; plot near the agent’s position
                # (You could do exactly the agent’s position or a slight offset)
                x, y = machine.owner.position
                # slight random offset to avoid overlapping the agent marker
                x += random.uniform(-0.3, 0.3)
                y += random.uniform(-0.3, 0.3)
                ax.plot(x, y, marker='D', color='purple', markersize=8)
                ax.text(x, y, f'{machine.name}', size=8, zorder=1, color='purple')
        
        ax.set_title("Economy Simulation")

def update(frame, economy, ax):
    economy.simulate(1)
    economy.plot_market(ax)

