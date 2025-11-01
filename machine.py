import numpy as np

class Machine:
    def __init__(self, name, position, skill_boost, resource_consumption):
        self.name = name
        self.position = np.array(position)
        self.skill_boost = skill_boost  # dict of {skill_name: boost_amount}
        self.resource_consumption = resource_consumption  # dict of {resource: amount_needed}
        self.owner = None
        self.in_use = False
        self.maintenance_cost = sum(resource_consumption.values()) * 2

    def can_operate(self, agent):
        """Check if agent has enough resources to operate the machine."""
        return all(
            agent.inventory.get(resource, 0) >= amount
            for resource, amount in self.resource_consumption.items()
        )

    def consume_resources(self, agent):
        """Consume resources needed for operation."""
        for resource, amount in self.resource_consumption.items():
            agent.inventory[resource] = agent.inventory.get(resource, 0) - amount
