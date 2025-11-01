import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from economy import Economy

def update(frame, economy, ax):
    economy.simulate(1)
    # If you have a plotting function:
    economy.plot_market(ax)

def random_position():
    return (random.uniform(-8, 8), random.uniform(-8, 8))

if __name__ == "__main__":
    economy = Economy()

    # Add agents
    economy.add_agent("Alice", 100, (0, 0), {'good1': 10, 'good2': 5, 'good3': 1})
    economy.add_agent("Bob",   50, (5, 5),  {'good1': 2,  'good2': 80, 'good3': 4})
    economy.add_agent("Charlie", 75, (-5, -5), {'good1': 6, 'good2': 3, 'good3': 7})

    # Add some goods
    for _ in range(5):
        economy.add_good('good1', random_position(), 'necessary')
    for _ in range(7):
        economy.add_good('good2', random_position(), 'necessary')
    for _ in range(10):
        economy.add_good('good3', random_position(), 'luxury')

    # Add machines
    # (make sure machine references skill.py if needed)
    from machine import Machine
    machine1 = Machine("AutoFabricator", (1, 1),
                       skill_boost={'manufacturing': 2, 'technology': 1},
                       resource_consumption={'good1': 2, 'good2': 1})
    machine2 = Machine("ResearchAI", (-1, -1),
                       skill_boost={'research': 3, 'technology': 2},
                       resource_consumption={'good2': 2, 'good3': 1})
    economy.add_machine(machine1)
    economy.add_machine(machine2)

    # Add jobs
    economy.add_job('factory_work', (2, 2),
                    reward=20,
                    required_skills={'manufacturing': 1, 'operations': 1})
    economy.add_job('research_project', (-2, -2),
                    reward=35,
                    required_skills={'research': 2, 'technology': 1})
    economy.add_job('manufacturing_training', (3, 3),
                    reward=-10,
                    required_skills={'manufacturing': 0},
                    training_job=True)
    economy.add_job('tech_workshop', (-3, -3),
                    reward=-15,
                    required_skills={'technology': 0},
                    training_job=True)

    print(economy.market)

    # Run animation
    fig, ax = plt.subplots(figsize=(12, 8))
    ani = FuncAnimation(fig, update, fargs=(economy, ax), interval=10)
    plt.show()

    print('\nFinal Results of the Simulation:')
    print(economy.market)
