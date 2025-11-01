import numpy as np
import random
from skill import Skill
from machine import Machine

class EconomicAgent:
    def __init__(self, name, initial_wealth, position, values_1, luxury_pref=0.2):
        # Basic agent properties
        self.name = name
        self.wealth = initial_wealth
        self.position = np.array(position)
        self.values_1 = values_1
        self.luxury_pref = luxury_pref
        
        # Initialize inventory and demands
        self.inventory = {'good1': 0, 'good2': 0, 'good3': 0}
        self.desired_inventory = {good: 0 for good in ['good1', 'good2', 'good3']}
        self.internal_demand = {good: 0 for good in ['good1', 'good2', 'good3']}
        
        # Movement and collection properties
        self.target = None
        self.collecting = False
        self.collecting_steps = 0
        
        # Initialize skills
        self.skills = {
            'manufacturing': Skill('manufacturing'),
            'technology': Skill('technology'),
            'research': Skill('research'),
            'maintenance': Skill('maintenance'),
            'operations': Skill('operations')
        }
        
        # Job and machine properties
        self.machines = []  # Machines owned by this agent
        self.current_job = None
        self.job_progress = 0
        self.experience_multiplier = 1.0

    
    def earn(self, amount):
        self.wealth += amount

    def spend(self, amount):
        if amount > self.wealth:
            amount = self.wealth
        self.wealth -= amount

    def update_internal_demand(self, goods_list, good_types, scaling_factor=10):
            for good in goods_list:
                # Adjust desired inventory based on the type of good
                priority = scaling_factor * (1 + self.luxury_pref if good_types[good] == 'luxury' else 1)
                self.desired_inventory[good] = self.values_1.get(good, 0) * priority
                self.internal_demand[good] = self.desired_inventory[good] - self.inventory.get(good, 0)

    def perceived_value(self, good, leader_price):
        # Agents perceive the leader's price with some variation
        perception_error = random.uniform(-0.1, 0.1)  # +/- 10% error in perception
        return leader_price * (1 + perception_error)

    def pick_up_good(self, good, quantity=1):
        self.inventory[good] += quantity

    def trade_good(self, other, good, quantity):
        if self.inventory[good] >= quantity:
            self.inventory[good] -= quantity
            other.inventory[good] += quantity

    def move_towards(self, target_position):
        direction = target_position - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction_x = direction[0] / distance
            direction_y = direction[1] / distance  # Normalize the direction
            self.position[0] += direction_x * 0.1  # Move a fraction towards the target
            self.position[1] += direction_y * 0.1

    def choose_target(self, agents, goods, paying_jobs, training_jobs):
        """
        paying_jobs: list of jobs that reward > 0
        training_jobs: list of jobs that reward < 0
        """
        
        # 1. Determine if the agent "needs" money:
        needs_money = (self.wealth < 20)  # e.g., threshold or a more complex check
        
        # 2. Check if the agent can do any paying job
        affordable_paying_jobs = [
            j for j in paying_jobs
            if self.can_perform_job(j)  # i.e. meets skill requirements
        ]
        
        # 3. Check if there's a relevant training job (the agent might need new skills to do better-paying jobs)
        relevant_training_jobs = [
            j for j in training_jobs
            # if it trains a skill that is needed for some paying job the agent *cannot* do right now
            if self.training_helps_for_future_jobs(j, paying_jobs)
        ]
        
        # 4. Decide if the agent can afford the training job cost
        affordable_training_jobs = [
            j for j in relevant_training_jobs
            if abs(j.reward) <= self.wealth  # negative reward => cost to agent
        ]
        
        # 5. Evaluate collecting goods:
        # For example, see if there's a good that is very profitable or needed for internal demand
        # We'll build a small list of potential "collect" targets
        collecting_targets = self.evaluate_collecting_opportunities(goods)
        
        # --- Now pick which path is best ---
        # Priority logic (example):
        #   a) If needs money & can do paying job => do the nearest paying job
        if needs_money and affordable_paying_jobs:
            chosen_job = self.pick_nearest_job(affordable_paying_jobs)
            print(f"{self.name} chooses to do paying job {chosen_job.name}")
            self.target = (chosen_job.position, chosen_job.name)
            return
        
        #   b) If needs money but can't do paying job => try training job if possible
        if needs_money and not affordable_paying_jobs:
            if affordable_training_jobs:
                chosen_training = self.pick_nearest_job(affordable_training_jobs)
                print(f"{self.name} chooses training job {chosen_training.name}")
                self.target = (chosen_training.position, chosen_training.name)
                return
            else:
                # c) Can't afford training, so maybe collect goods to sell
                if collecting_targets:
                    chosen_good = self.pick_best_good(collecting_targets)
                    print(f"{self.name} collects {chosen_good.name} to sell or use later.")
                    self.target = (chosen_good.position, chosen_good.name)
                    return
        
        # d) If agent does not need money urgently, but sees profitable goods to collect
        if collecting_targets:
            chosen_good = self.pick_best_good(collecting_targets)
            print(f"{self.name} chooses to collect {chosen_good.name}")
            self.target = (chosen_good.position, chosen_good.name)
            return
        
        # e) If none of the above apply, do some default action (idle, wander, etc.)
        print(f"{self.name} has no urgent task. Idling.")
        self.target = None


    def buy_price(self, goods):
        buy_price = self.values_1[goods]
        return buy_price 

    def sell_price(self, goods):
        sell_price = self.values_1[goods] + 1
        return sell_price

    def find_nearby_agents(self, agents, radius):
        nearby_agents = []
        for agent in agents:
            if agent != self:
                distance = np.linalg.norm(agent.position - self.position)
                if distance <= radius:
                    nearby_agents.append(agent)
        return nearby_agents
    def can_perform_job(self, job):
        #"""Check if agent meets skill requirements for a job"""
        return all(self.skills[skill].level >= level 
                  for skill, level in job.required_skills.items())

    def perform_job(self, job):
        #"""Attempt to perform a job with potential machine assistance"""
        if not self.can_perform_job(job):
            return False

        # Calculate effective skill level including machine boosts
        effective_skills = self.calculate_effective_skills()
        completion_speed = 1.0

        # Check for available machines that can help
        usable_machines = [m for m in self.machines 
                          if not m.in_use and m.can_operate(self)]
        
        # Use machines if available and beneficial
        for machine in usable_machines:
            if job.machine_compatible:
                if machine.can_operate(self):
                    machine.in_use = True
                    machine.consume_resources(self)
                    completion_speed += 0.5
                    self.experience_multiplier += 0.2

        # Progress the job
        self.job_progress += completion_speed
        if self.job_progress >= job.completion_time:
            # Complete the job
            if job.reward > 0:  # Paying job
                self.earn(job.reward)
            else:  # Training job
                self.spend(abs(job.reward))

            # Grant experience
            for skill, exp in job.experience_reward.items():
                if skill in self.skills:
                    self.skills[skill].gain_experience(
                        exp * self.experience_multiplier
                    )

            # Reset job progress and machine use
            self.job_progress = 0
            for machine in self.machines:
                machine.in_use = False
            
            return True

        return False

    def calculate_effective_skills(self):
        #"""Calculate total skill levels including machine boosts"""
        effective_skills = {name: skill.level 
                          for name, skill in self.skills.items()}
        
        for machine in self.machines:
            if not machine.in_use:
                for skill, boost in machine.skill_boost.items():
                    if skill in effective_skills:
                        effective_skills[skill] += boost
        
        return effective_skills

    def machine_trade(self, seller, buyer, machine, price):
    #Buyer purchases a machine from Seller.
        if machine in seller.machines and buyer.wealth >= price:
            buyer.spend(price)
            seller.earn(price)
            seller.machines.remove(machine)
            buyer.acquire_machine(machine)
            print(f"{buyer.name} bought {machine.name} from {seller.name} for {price}.")
        else:
            print(f"Trade failed: either {seller.name} does not own {machine.name}, or {buyer.name} lacks wealth.")

    def acquire_machine(self, machine):
        machine.owner = self
        self.machines.append(machine)
        print(f"{self.name} just acquired machine {machine.name}")

    def maintain_machines(self):
        #"""Pay maintenance costs for owned machines"""
        total_maintenance = sum(m.maintenance_cost for m in self.machines)
        self.spend(total_maintenance)

    def __str__(self):
        return f"{self.name} has wealth: {self.wealth:.2f}, inventory: {self.inventory}, at position {self.position}"