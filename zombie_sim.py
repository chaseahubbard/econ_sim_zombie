import random
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button

# Reuse your existing economy scaffolding
from agent import EconomicAgent
from market import Market
from job import Job
from good import Good

# ==========================================================
# Zombie Economy v4
#  - Start/Pause: sliders are adjustable **before** the sim runs
#  - Hover tooltips over sliders (inline explainer popups)
#  - Agents keep working/collecting even with 0 zombies
#  - Limited carry capacity; agents must deposit at labeled storage depots
#  - Concentric activity dots (no lines), legend, KPI sliders, dual maps
# ==========================================================

# -------- Color palette (single source of truth for both maps) --------
COLORS = {
    'state': {
        'S': 'tab:blue',      # Human (susceptible)
        'I': 'tab:orange',    # Infected
        'Z': 'tab:red',       # Zombie
        'Q': 'tab:gray',      # Quarantine
        'R': 'tab:green',     # Removed
    },
    'activity': {
        'TRADE': 'tab:purple',
        'HARVEST': 'tab:olive',
        'MINE': 'tab:brown',
        'WORK': 'tab:blue',
        'CRAFT': 'tab:orange',
        'FLEE': 'tab:red',
        'STORE': 'black',
        'DEPOSIT': 'black',
    },
    'nodes': {
        'food_node': 'tab:olive',
        'supplies_node': 'tab:cyan',
        'scrap_node': 'tab:brown',
        'chem_node': 'tab:purple',
        'electronics_node': 'tab:pink',
        'depot': 'black',
    }
}

# ==========================================================
# Core types
# ==========================================================
class HealthState(str, Enum):
    S = "susceptible"
    I = "infected"
    Z = "zombie"
    Q = "quarantine"
    R = "removed"

class Activity(str, Enum):
    IDLE = "idle"
    FLEE = "flee"
    HARVEST = "harvest"   # food/supplies
    MINE = "mine"         # scrap/chems/electronics
    WORK = "work"
    CRAFT = "craft"
    TRADE = "trade"
    STORE = "store"       # walking to storage
    DEPOSIT = "deposit"   # depositing at storage

@dataclass
class Policy:
    # Transmission & progression (SIZR)
    beta: float = 0.0095
    rho: float = 0.005
    alpha: float = 0.005
    zeta: float = 0.0001

    # Quarantine levers (SIZRQ)
    kappa: float = 0.02
    sigma: float = 0.02
    gamma_release: float = 0.0

    # Health system
    hospital_capacity: int = 8
    cure_rate: float = 0.08
    neutralize_rate: float = 0.02

    # Impulsive eradication
    strike_every_steps: int = 200
    kill_ratio_per_strike: float = 0.35

    # Spatial / behavior
    bite_distance: float = 0.5
    notice_radius: float = 3.5
    human_speed: float = 0.12
    zombie_speed: float = 0.08
    flee_speed: float = 0.18

    # Economy / subsistence
    daily_food_need: float = 0.08 
    starvation_limit: int = 3
    trade_per_step: int = 60
    base_prices: Dict[str, float] = None

    # Resource system
    max_resource_nodes: int = 26           # hard cap across all nodes
    spawn_chance_per_step: float = 0.12    # probability to spawn if under cap
    node_capacity_min: int = 2
    node_capacity_max: int = 6
    harvest_radius: float = 0.6            # must be within to continue harvesting

    # Harvesting times (ticks per 1 unit gathered)
    t_harvest_food: int = 8
    t_harvest_supplies: int = 10
    t_mine_scrap: int = 14
    t_mine_chems: int = 16
    t_mine_electronics: int = 18

    # Food regrowth per node (simple probabilistic growth up to max)
    food_regrow_chance: float = 0.20

    # Carry capacity & storage
    carry_capacity: int = 14
    storage_deposit_time: int = 8
    storage_radius: float = 0.8

    def __post_init__(self):
        if self.base_prices is None:
            self.base_prices = {"food": 2.0, "supplies": 3.0, "lux": 5.0}

# ---------------------------------
# Extended agent with health & needs
# ---------------------------------
class EpidemicAgent(EconomicAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.health_state: HealthState = HealthState.S
        self.incubation_clock: float = 0.0
        self.icon = "●"
        self.quarantined: bool = False
        self.assigned_bed: bool = False
        self.hunger_streak: int = 0
        self.metabolism_buf: float = 0.0 
        
        # Activity and work-in-progress
        self.activity: Activity = Activity.IDLE
        self.activity_progress: int = 0       # ticks accrued toward next unit
        self.activity_target: Optional[object] = None  # ResourceNode or depot index
        self.activity_hold_time: int = 0      # configured ticks required
        self.trade_flash: int = 0             # frames to keep trade indicator

        # Ensure inventory keys exist
        for k in ("food", "supplies", "lux", "scrap", "chems", "electronics",
                  "barricade_kit", "med_kit", "trap"):
            if k not in self.inventory:
                self.inventory[k] = 0

    def reset_activity(self):
        self.activity = Activity.IDLE
        self.activity_progress = 0
        self.activity_target = None
        self.activity_hold_time = 0

    def earn(self, amount: float):
        self.wealth = float(self.wealth + amount)

    def spend(self, amount: float) -> bool:
        if self.wealth >= amount:
            self.wealth = float(self.wealth - amount)
            return True
        return False

    def move_step(self, target: np.ndarray, speed: float):
        direction = target - self.position
        d = np.linalg.norm(direction)
        if d > 1e-9:
            self.position += (direction / d) * speed

    def wander(self, bounds: Tuple[float, float, float, float], speed: float):
        x0, x1, y0, y1 = bounds
        jitter = np.random.uniform(-1, 1, size=2)
        target = self.position + jitter
        self.move_step(target, speed)
        self.position[0] = float(np.clip(self.position[0], x0, x1))
        self.position[1] = float(np.clip(self.position[1], y0, y1))

    def carried_items(self) -> int:
        # Total tangible items (exclude cash/wealth)
        keys = ("food","supplies","lux","scrap","chems","electronics",
                "barricade_kit","med_kit","trap")
        return int(sum(max(0, int(self.inventory.get(k,0))) for k in keys))

# Attach capacity & regrowth metadata to Good nodes
class ResourceNode(Good):
    def __init__(self, name: str, position: Tuple[float, float], amount: int, max_amount: Optional[int] = None):
        super().__init__(name, position)
        self.amount = int(amount)
        self.max_amount = int(max_amount if max_amount is not None else amount)

# -----------------
# Simulation system
# -----------------
class ZombieSim:
    def __init__(self, n_agents: int = 80, initial_zombies: int = 2,
                 bounds: Tuple[float, float, float, float] = (-10, 10, -10, 10),
                 policy: Policy = Policy(), seed: int = 7):
        random.seed(seed)
        np.random.seed(seed)
        self.bounds = bounds
        self.t = 0
        self.market = Market()
        self.policy = policy
        self.history = {"S": [], "I": [], "Z": [], "Q": [], "R": []}
        self.avgs = {"food": [], "wealth": [], "supplies": []}

        # Infra built by crafting (affects global rates)
        self.infra = {"barricades": 0, "medkits": 0, "traps": 0}

        # Hospitals and quarantine zone
        self.hospitals = [np.array([0.0, 0.0]), np.array([7.0, -7.0])]
        self.quarantine_box = (-9.0, -5.0, 5.0, 9.0)

        # Storage depots (clearly labeled)
        self.depots = [
            {"pos": np.array([-8.0, -8.0]), "inv": {"food":0,"supplies":0,"scrap":0,"chems":0,"electronics":0}},
            {"pos": np.array([8.0, 8.0]),   "inv": {"food":0,"supplies":0,"scrap":0,"chems":0,"electronics":0}},
        ]

        # Create agents
        for i in range(n_agents):
            p = (np.random.uniform(bounds[0]+1, bounds[1]-1),
                 np.random.uniform(bounds[2]+1, bounds[3]-1))
            a = EpidemicAgent(name=f"A{i}", initial_wealth=float(np.random.randint(40, 120)),
                               position=p, values_1={'good1':5,'good2':5,'good3':3})
            a.inventory["food"] = int(np.random.randint(2, 8))
            a.inventory["supplies"] = int(np.random.randint(0, 5))
            a.inventory["lux"] = int(np.random.randint(0, 3))
            self.market.add_agent(a)

        # Seed zombies
        for z in np.random.choice(self.market.agents, size=min(initial_zombies, len(self.market.agents)), replace=False):
            self._become_zombie(z)

        # Seed nodes under cap
        self._seed_nodes()

        # Jobs
        self.jobs: List[Job] = [
            Job('farm', np.array([4.0, 4.0]), reward=8, required_skills={'manufacturing': 1}),
            Job('factory', np.array([-4.0, -4.0]), reward=10, required_skills={'manufacturing': 1}),
            Job('research_gig', np.array([0.0, -6.0]), reward=14, required_skills={'research': 1}),
        ]

        # Market prices
        self.prices = dict(self.policy.base_prices)

    # ---------
    # Dynamics
    # ---------
    def step(self):
        P = self.policy
        agents = self.market.agents
        humans = [a for a in agents if a.health_state in (HealthState.S, HealthState.I, HealthState.Q)]
        zombies = [a for a in agents if a.health_state == HealthState.Z]

        # Effective rates including crafted infra multipliers
        alpha_eff = P.alpha * (1.0 + 0.02 * self.infra['barricades'])
        cure_eff = P.cure_rate * (1.0 + 0.03 * self.infra['medkits'])
        neut_eff = P.neutralize_rate * (1.0 + 0.03 * self.infra['traps'])
        sigma_eff = P.sigma * (1.0 + 0.02 * self.infra['traps'])

        # 0) Subsistence
        for a in humans:
            if a.health_state == HealthState.R:
                continue
            a.metabolism_buf += self.policy.daily_food_need

            # Pay whole units only when buffer crosses 1.0
            while a.metabolism_buf >= 1.0:
                if a.inventory.get("food", 0) >= 1:
                    a.inventory["food"] -= 1
                    a.hunger_streak = 0
                else:
                    a.hunger_streak += 1
                    if a.hunger_streak >= self.policy.starvation_limit and a.health_state != HealthState.Z:
                        self._neutralize(a)
                a.metabolism_buf -= 1.0

        # 1) Movement/Actions
        for a in agents:
            if a.health_state == HealthState.R:
                continue
            if a.trade_flash > 0:
                a.trade_flash -= 1

            # If over carry capacity, go store
            if a.health_state in (HealthState.S, HealthState.I) and a.carried_items() >= P.carry_capacity:
                self._begin_store(a)
                continue

            # quarantine controls movement
            if a.health_state == HealthState.Q:
                qx0, qx1, qy0, qy1 = self.quarantine_box
                target = np.array([(qx0+qx1)/2, (qy0+qy1)/2])
                a.move_step(target, P.human_speed)
                a.reset_activity()
                continue

            if a.health_state == HealthState.Z:
                targets = [h for h in humans if h.health_state in (HealthState.S, HealthState.I)]
                if targets:
                    t0 = min(targets, key=lambda h: np.linalg.norm(h.position - a.position))
                    a.move_step(t0.position, P.zombie_speed)
                else:
                    a.wander(self.bounds, P.zombie_speed)
                a.reset_activity()
                continue

            # Human: flee if zombie nearby
            nearest_z, d = self._nearest_zombie(a, zombies)
            if nearest_z is not None and d < P.notice_radius:
                away = a.position - nearest_z.position
                if np.linalg.norm(away) < 1e-9:
                    away = np.random.uniform(-1, 1, size=2)
                target = a.position + (away / np.linalg.norm(away)) * 3.0
                a.move_step(target, P.flee_speed)
                a.activity = Activity.FLEE
                a.activity_progress = 0
                a.activity_target = None
                continue

            # Continue an in-place activity if set
            if a.activity in (Activity.HARVEST, Activity.MINE, Activity.WORK, Activity.CRAFT, Activity.DEPOSIT):
                self._tick_stationary_activity(a)
                continue

            # Otherwise decide next target: eat → gather → job → craft
            if a.inventory.get("food", 0) < 2:
                self._begin_harvest(a, preferred="food_node")
            elif self._needs_crafting_mats(a):
                self._begin_mining_or_harvest_mats(a)
            else:
                # Keep economy alive even with 0 zombies: work or wander to job
                job = min(self.jobs, key=lambda j: np.linalg.norm(a.position - j.position))
                a.move_step(job.position, P.human_speed)
                if np.linalg.norm(a.position - job.position) < 0.7:
                    a.activity = Activity.WORK
                    a.activity_hold_time = 10
                    a.activity_progress = 0
                    a.activity_target = None
                else:
                    # If far, occasionally harvest nearby node en route
                    near_food = self._nearest_node(a, 'food_node')
                    if near_food and np.linalg.norm(a.position - near_food.position) < 1.0:
                        self._begin_harvest(a, preferred='food_node')
                    else:
                        a.activity = Activity.IDLE

        # 2) Infection events
        for s in [h for h in agents if h.health_state == HealthState.S]:
            for z in zombies:
                d = np.linalg.norm(z.position - s.position)
                if d < P.bite_distance:
                    p_infect = P.beta / (P.beta + alpha_eff)
                    if np.random.rand() < p_infect:
                        self._become_infected(s)
                    else:
                        if np.random.rand() < alpha_eff:
                            self._neutralize(z)

        zombies = [a for a in agents if a.health_state == HealthState.Z]

        # 3) Progression & resurrection
        for a in agents:
            if a.health_state == HealthState.I:
                a.incubation_clock += 1
                if np.random.rand() < self.policy.rho:
                    self._become_zombie(a)
            elif a.health_state == HealthState.R and np.random.rand() < self.policy.zeta:
                self._become_zombie(a)

        # 4) Quarantine transfers
        for a in agents:
            if a.health_state == HealthState.I and np.random.rand() < self.policy.kappa:
                self._send_to_quarantine(a)
            if a.health_state == HealthState.Z and np.random.rand() < sigma_eff:
                self._send_to_quarantine(a)
        for a in [x for x in agents if x.health_state == HealthState.Q]:
            if np.random.rand() < self.policy.gamma_release:
                a.health_state = HealthState.S
                a.quarantined = False
                a.icon = "●"

        # 5) Health care
        beds = self.policy.hospital_capacity
        triage = [x for x in agents if x.health_state == HealthState.I] + [x for x in agents if x.health_state == HealthState.Z]
        triage = triage[:beds]
        for x in triage:
            x.assigned_bed = True
            if x.health_state == HealthState.I and np.random.rand() < cure_eff:
                x.health_state = HealthState.S
                x.icon = "●"
                x.incubation_clock = 0
            elif x.health_state == HealthState.Z and np.random.rand() < neut_eff:
                self._neutralize(x)
        for a in agents:
            a.assigned_bed = False

        # 6) Impulsive eradication
        if self.policy.kill_ratio_per_strike > 0 and self.policy.strike_every_steps > 0 and (self.t % self.policy.strike_every_steps == 0) and self.t > 0:
            self._impulsive_strike(self.policy.kill_ratio_per_strike)

        # 7) Market
        self._market_clearing()

        # 8) Resource cap & food regrowth & spawn
        self._resource_regrowth_and_spawn()

        # 9) Housekeeping
        self.t += 1
        self._record_counts()
        self._record_avgs()

    # ---------
    # Activity mechanics
    # ---------
    def _begin_store(self, a: EpidemicAgent):
        # Go to nearest depot to deposit items
        depot_idx, depot = min(enumerate(self.depots), key=lambda kv: np.linalg.norm(a.position - kv[1]['pos']))
        a.move_step(depot['pos'], self.policy.human_speed)
        a.activity = Activity.STORE
        a.activity_target = depot_idx
        if np.linalg.norm(a.position - depot['pos']) < self.policy.storage_radius:
            a.activity = Activity.DEPOSIT
            a.activity_hold_time = self.policy.storage_deposit_time
            a.activity_progress = 0

    def _begin_harvest(self, a: EpidemicAgent, preferred: Optional[str] = None):
        kind = preferred or 'food_node'
        node = self._nearest_node(a, kind)
        if node is None:
            node = self._nearest_node(a, 'supplies_node')
            if node is None:
                a.wander(self.bounds, self.policy.human_speed)
                a.activity = Activity.IDLE
                return
        a.move_step(node.position, self.policy.human_speed)
        if np.linalg.norm(a.position - node.position) < self.policy.harvest_radius:
            a.activity = Activity.HARVEST
            a.activity_target = node
            a.activity_progress = 0
            a.activity_hold_time = self._harvest_time_for_node(node)
        else:
            a.activity = Activity.IDLE

    def _begin_mining_or_harvest_mats(self, a: EpidemicAgent):
        choices = []
        for kind in ('scrap_node','chem_node','electronics_node','supplies_node'):
            if self._needs_kind(a, kind):
                node = self._nearest_node(a, kind)
                if node is not None:
                    choices.append((kind, node))
        if not choices:
            a.wander(self.bounds, self.policy.human_speed)
            a.activity = Activity.IDLE
            return
        kind, node = min(choices, key=lambda kv: np.linalg.norm(a.position - kv[1].position))
        a.move_step(node.position, self.policy.human_speed)
        if np.linalg.norm(a.position - node.position) < self.policy.harvest_radius:
            a.activity = Activity.MINE if kind in ('scrap_node','chem_node','electronics_node') else Activity.HARVEST
            a.activity_target = node
            a.activity_progress = 0
            a.activity_hold_time = self._harvest_time_for_node(node)
        else:
            a.activity = Activity.IDLE

    def _tick_stationary_activity(self, a: EpidemicAgent):
        node = a.activity_target
        if a.activity in (Activity.HARVEST, Activity.MINE) and isinstance(node, ResourceNode):
            d = np.linalg.norm(a.position - node.position)
            if d > self.policy.harvest_radius or node.amount <= 0:
                a.reset_activity(); return
            a.activity_progress += 1
            if a.activity_progress >= a.activity_hold_time:
                inv_key = self._mat_name_from_node(node.name)
                a.inventory[inv_key] = a.inventory.get(inv_key, 0) + 1
                node.amount -= 1
                a.activity_progress = 0
                if node.amount <= 0:
                    try: self.market.goods.remove(node)
                    except ValueError: pass
        elif a.activity == Activity.WORK:
            a.activity_progress += 1
            if a.activity_progress >= a.activity_hold_time:
                self._work_payout(a)
                a.reset_activity()
        elif a.activity == Activity.CRAFT:
            a.activity_progress += 1
            if a.activity_progress >= a.activity_hold_time:
                self._finish_craft(a)
                a.reset_activity()
        elif a.activity == Activity.DEPOSIT:
            a.activity_progress += 1
            if a.activity_progress >= a.activity_hold_time:
                self._finish_deposit(a)
                a.reset_activity()

    def _finish_deposit(self, a: EpidemicAgent):
        depot_idx = a.activity_target if isinstance(a.activity_target, int) else 0
        depot = self.depots[depot_idx]
        # Keep *all* food on-person; offload other materials beyond a small cushion
        keep = {"supplies": 1}
        for k in ("supplies", "scrap", "chems", "electronics"):
            qty = int(a.inventory.get(k, 0))
            spare = qty - keep.get(k, 0)
            if spare > 0:
                depot['inv'][k] += spare
                a.inventory[k] -= spare

    def _work_payout(self, a: EpidemicAgent):
        job = min(self.jobs, key=lambda j: np.linalg.norm(a.position - j.position))
        if job.name == 'farm':
            a.inventory['food'] += 1; a.earn(job.reward)
        elif job.name == 'factory':
            a.inventory['supplies'] += 1; a.earn(job.reward)
        else:
            a.earn(job.reward)
        self._maybe_start_craft(a)

    def _maybe_start_craft(self, a: EpidemicAgent):
        if a.activity in (Activity.HARVEST, Activity.MINE, Activity.WORK, Activity.CRAFT, Activity.DEPOSIT):
            return
        if (a.inventory.get('scrap',0) >= 2 and a.inventory.get('supplies',0) >= 1 and a.inventory.get('barricade_kit',0) < 1) or \
           (a.inventory.get('chems',0) >= 2 and a.inventory.get('supplies',0) >= 1 and a.inventory.get('med_kit',0) < 1) or \
           (a.inventory.get('electronics',0) >= 1 and a.inventory.get('scrap',0) >= 1 and a.inventory.get('trap',0) < 1):
            a.activity = Activity.CRAFT
            a.activity_hold_time = 12
            a.activity_progress = 0

    def _finish_craft(self, a: EpidemicAgent):
        if a.inventory.get('barricade_kit',0) < 1 and a.inventory.get('scrap',0) >= 2 and a.inventory.get('supplies',0) >= 1:
            a.inventory['scrap'] -= 2; a.inventory['supplies'] -= 1
            a.inventory['barricade_kit'] += 1; self.infra['barricades'] += 1; return
        if a.inventory.get('med_kit',0) < 1 and a.inventory.get('chems',0) >= 2 and a.inventory.get('supplies',0) >= 1:
            a.inventory['chems'] -= 2; a.inventory['supplies'] -= 1
            a.inventory['med_kit'] += 1; self.infra['medkits'] += 1; return
        if a.inventory.get('trap',0) < 1 and a.inventory.get('electronics',0) >= 1 and a.inventory.get('scrap',0) >= 1:
            a.inventory['electronics'] -= 1; a.inventory['scrap'] -= 1
            a.inventory['trap'] += 1; self.infra['traps'] += 1; return

    # ---------
    # Market mechanics
    # ---------
    def _market_clearing(self):
        P = self.policy
        agents = [a for a in self.market.agents if a.health_state != HealthState.R]

        def wants(a, good):
            if good == 'food':
                return a.inventory.get('food', 0) < 2
            if good == 'supplies':
                return a.inventory.get('supplies', 0) < 2 and a.inventory.get('food', 0) >= 1
            return False

        def surplus(a, good):
            if good == 'food':
                return max(0, a.inventory.get('food', 0) - 3)
            if good == 'supplies':
                return max(0, a.inventory.get('supplies', 0) - 3)
            return 0

        for good in ('food','supplies'):
            buyers = [a for a in agents if wants(a, good) and a.wealth >= self.prices[good]]
            sellers: List[EpidemicAgent] = []
            for a in agents:
                qty = surplus(a, good)
                if qty > 0:
                    sellers.extend([a]*int(qty))

            trades = 0
            while buyers and sellers and trades < P.trade_per_step:
                b = buyers.pop(np.random.randint(0, len(buyers)))
                s_idx = np.random.randint(0, len(sellers))
                s = sellers.pop(s_idx)
                if s is b:
                    continue
                price = self.prices[good]
                if b.spend(price):
                    b.inventory[good] = b.inventory.get(good, 0) + 1
                    s.earn(price)
                    trades += 1
                    b.activity = Activity.TRADE; b.trade_flash = max(b.trade_flash, 6)
                    s.activity = Activity.TRADE; s.trade_flash = max(s.trade_flash, 6)

            excess_demand = len(buyers) - len(sellers)
            self.prices[good] = max(0.5, float(self.prices[good] * (1.0 + 0.002 * np.tanh(excess_demand/10.0))))

    # ---------
    # Resources: cap, regrowth, spawn
    # ---------
    def _seed_nodes(self):
        P = self.policy
        kinds = [
            ('food_node', 6),
            ('supplies_node', 4),
            ('scrap_node', 6),
            ('chem_node', 4),
            ('electronics_node', 4)
        ]
        total = 0
        for kind, n in kinds:
            for _ in range(n):
                if total >= P.max_resource_nodes:
                    return
                amt = np.random.randint(P.node_capacity_min, P.node_capacity_max+1)
                max_amt = amt if kind != 'food_node' else max(amt, P.node_capacity_max + 2)
                self.market.add_good(ResourceNode(kind, self._rand_pos(), amount=amt, max_amount=max_amt))
                total += 1

    def _resource_regrowth_and_spawn(self):
        P = self.policy
        for g in list(self.market.goods):
            if isinstance(g, ResourceNode) and g.name == 'food_node' and g.amount < g.max_amount:
                if np.random.rand() < P.food_regrow_chance:
                    g.amount += 1
        nodes = [g for g in self.market.goods if isinstance(g, ResourceNode)]
        if len(nodes) < P.max_resource_nodes and np.random.rand() < P.spawn_chance_per_step:
            kind = np.random.choice(['scrap_node','chem_node','electronics_node','food_node','supplies_node'], p=[0.22,0.16,0.16,0.30,0.16])
            amt = np.random.randint(P.node_capacity_min, P.node_capacity_max+1)
            max_amt = amt if kind != 'food_node' else max(amt, P.node_capacity_max + 2)
            self.market.add_good(ResourceNode(kind, self._rand_pos(), amount=amt, max_amount=max_amt))

    # ---------
    # Helper utilities
    # ---------
    def _rand_pos(self) -> Tuple[float, float]:
        x0, x1, y0, y1 = self.bounds
        return (np.random.uniform(x0+1, x1-1), np.random.uniform(y0+1, y1-1))

    def _nearest_zombie(self, a: EpidemicAgent, zombies: List[EpidemicAgent]):
        if not zombies:
            return None, math.inf
        z = min(zombies, key=lambda z: np.linalg.norm(z.position - a.position))
        return z, float(np.linalg.norm(z.position - a.position))

    def _nearest_node(self, a: EpidemicAgent, kind: str) -> Optional[ResourceNode]:
        nodes = [g for g in self.market.goods if isinstance(g, ResourceNode) and g.name == kind]
        if not nodes:
            return None
        return min(nodes, key=lambda g: np.linalg.norm(a.position - g.position))

    def _needs_crafting_mats(self, a: EpidemicAgent) -> bool:
        return (a.inventory.get('barricade_kit', 0) < 1 or
                a.inventory.get('med_kit', 0) < 1 or
                a.inventory.get('trap', 0) < 1)

    def _needs_kind(self, a: EpidemicAgent, kind: str) -> bool:
        if kind == 'scrap_node':
            return a.inventory.get('scrap',0) < 2 or a.inventory.get('trap',0) < 1 or a.inventory.get('barricade_kit',0) < 1
        if kind == 'chem_node':
            return a.inventory.get('chems',0) < 2 or a.inventory.get('med_kit',0) < 1
        if kind == 'electronics_node':
            return a.inventory.get('electronics',0) < 1 or a.inventory.get('trap',0) < 1
        if kind == 'supplies_node':
            return a.inventory.get('supplies',0) < 2
        return False

    @staticmethod
    def _mat_name_from_node(kind: str) -> str:
        return {
            'scrap_node': 'scrap',
            'chem_node': 'chems',
            'electronics_node': 'electronics',
            'food_node': 'food',
            'supplies_node': 'supplies',
        }.get(kind, 'supplies')

    def _harvest_time_for_node(self, node: ResourceNode) -> int:
        P = self.policy
        if node.name == 'food_node':
            return P.t_harvest_food
        if node.name == 'supplies_node':
            return P.t_harvest_supplies
        if node.name == 'scrap_node':
            return P.t_mine_scrap
        if node.name == 'chem_node':
            return P.t_mine_chems
        if node.name == 'electronics_node':
            return P.t_mine_electronics
        return 12

    def _become_infected(self, a: EpidemicAgent):
        if a.health_state == HealthState.S:
            a.health_state = HealthState.I
            a.incubation_clock = 0
            a.icon = "◓"
            a.reset_activity()

    def _become_zombie(self, a: EpidemicAgent):
        a.health_state = HealthState.Z
        a.icon = "✖"
        a.quarantined = False
        a.reset_activity()

    def _neutralize(self, a: EpidemicAgent):
        a.health_state = HealthState.R
        a.icon = "✚"
        a.reset_activity()

    def _send_to_quarantine(self, a: EpidemicAgent):
        if a.health_state in (HealthState.I, HealthState.Z):
            a.health_state = HealthState.Q
            a.quarantined = True
            x0, x1, y0, y1 = self.quarantine_box
            a.position = np.array([np.random.uniform(x0, x1), np.random.uniform(y0, y1)])
            a.icon = "□"
            a.reset_activity()

    def _impulsive_strike(self, kill_ratio: float):
        zombies = [a for a in self.market.agents if a.health_state == HealthState.Z]
        if not zombies:
            return
        k = max(1, int(len(zombies) * kill_ratio))
        to_remove = list(np.random.choice(zombies, size=k, replace=False))
        for z in to_remove:
            self._neutralize(z)

    # ----------
    # STATS & DRAWING
    # ----------
    def _record_counts(self):
        counts = {s: 0 for s in self.history}
        for a in self.market.agents:
            if a.health_state == HealthState.S: counts["S"] += 1
            elif a.health_state == HealthState.I: counts["I"] += 1
            elif a.health_state == HealthState.Z: counts["Z"] += 1
            elif a.health_state == HealthState.Q: counts["Q"] += 1
            elif a.health_state == HealthState.R: counts["R"] += 1
        for k, v in counts.items():
            self.history[k].append(v)

    def _record_avgs(self):
        alive = [a for a in self.market.agents if a.health_state != HealthState.R]
        if not alive:
            for k in self.avgs:
                self.avgs[k].append(0.0)
            return
        f = float(np.mean([a.inventory.get('food', 0.0) for a in alive]))
        w = float(np.mean([a.wealth for a in alive]))
        s = float(np.mean([a.inventory.get('supplies', 0.0) for a in alive]))
        self.avgs['food'].append(f)
        self.avgs['wealth'].append(w)
        self.avgs['supplies'].append(s)

    # ---- shared map drawing (used by left & right maps) ----
    def draw_agent_map(self, ax, title=None):
        x0, x1, y0, y1 = self.bounds
        ax.set_xlim([x0, x1]); ax.set_ylim([y0, y1])
        if title:
            ax.set_title(title)
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        # Quarantine
        qx0, qx1, qy0, qy1 = self.quarantine_box
        ax.plot([qx0, qx1, qx1, qx0, qx0], [qy0, qy0, qy1, qy1, qy0], linestyle='--')
        for h in self.hospitals:
            ax.plot(h[0], h[1], marker='P', markersize=8)

        # Storage depots (squares + label)
        for i, d in enumerate(self.depots):
            pos = d['pos']
            ax.scatter([pos[0]],[pos[1]], s=80, marker='s', c=COLORS['nodes']['depot'])
            ax.text(pos[0], pos[1]+0.35, f"Depot {i+1}", ha='center', va='bottom', fontsize=9, color='black')

        # Resource nodes (squares with count)
        for g in [g for g in self.market.goods if isinstance(g, ResourceNode)]:
            color = COLORS['nodes'].get(g.name, 'k')
            ax.scatter([g.position[0]],[g.position[1]], s=55, marker='s', edgecolor='none', alpha=0.9, c=color)
            ax.text(g.position[0], g.position[1]+0.25, f"{g.amount}", ha='center', va='bottom', fontsize=8)

        # Agents with concentric activity indicators
        for a in self.market.agents:
            st_key = a.health_state.name[0]
            base_c = COLORS['state'][st_key]
            ax.text(a.position[0], a.position[1], a.icon, ha='center', va='center', color=base_c)
            if a.health_state != HealthState.R:
                inner_c = None
                if a.activity == Activity.TRADE: inner_c = COLORS['activity']['TRADE']
                elif a.activity == Activity.HARVEST: inner_c = COLORS['activity']['HARVEST']
                elif a.activity == Activity.MINE: inner_c = COLORS['activity']['MINE']
                elif a.activity == Activity.WORK: inner_c = COLORS['activity']['WORK']
                elif a.activity == Activity.CRAFT: inner_c = COLORS['activity']['CRAFT']
                elif a.activity == Activity.FLEE: inner_c = COLORS['activity']['FLEE']
                elif a.activity in (Activity.STORE, Activity.DEPOSIT): inner_c = COLORS['activity']['DEPOSIT']
                if inner_c is not None:
                    ax.scatter([a.position[0]],[a.position[1]], s=24, marker='o', c=inner_c, alpha=0.9)

    def legend_handles(self):
        state_handles = [Patch(color=COLORS['state'][k], label=lbl)
                         for k,lbl in [('S','Human (S)'),('I','Infected (I)'),('Z','Zombie (Z)'),('Q','Quarantine (Q)'),('R','Removed (R)')]]
        activity_handles = [Line2D([0],[0], marker='o', linestyle='None', markersize=8,
                                   markerfacecolor=COLORS['activity'][k], markeredgecolor=COLORS['activity'][k],
                                   label=lbl)
                            for k,lbl in [('TRADE','Trade'),('HARVEST','Harvest'),('MINE','Mine'),('WORK','Work'),('CRAFT','Craft'),('FLEE','Flee'),('DEPOSIT','Store/Deposit')]]
        node_handles = [Patch(color=COLORS['nodes'][k], label=lbl)
                        for k,lbl in [('food_node','Food node'),('supplies_node','Supplies node'),('scrap_node','Scrap node'),('chem_node','Chem node'),('electronics_node','Electronics node'),('depot','Storage depot')]]
        return state_handles, activity_handles, node_handles

# ------------------
# UI (two maps + TS + averages + KPI sliders + start/pause + tooltips)
# ------------------
class ZombieUI:
    def __init__(self, sim: ZombieSim):
        self.sim = sim
        self.fig = plt.figure(figsize=(16,10))

        # Layout
        gs_top = self.fig.add_gridspec(3, 3, height_ratios=[6, 2.2, 2.6], width_ratios=[1.1, 1.0, 1.1], hspace=0.4, wspace=0.25)
        self.ax_map_left = self.fig.add_subplot(gs_top[0,0])
        self.ax_ts = self.fig.add_subplot(gs_top[0,1])
        self.ax_map_right = self.fig.add_subplot(gs_top[0,2])
        self.ax_avg = self.fig.add_subplot(gs_top[1, :])

        # Sliders
        slider_specs = [
            ("beta", 0.0, 0.05, sim.policy.beta, "Transmission probability per bite (higher spreads faster)."),
            ("rho", 0.0, 0.05, sim.policy.rho, "Progression from infected to zombie per tick."),
            ("alpha", 0.0, 0.05, sim.policy.alpha, "Chance a bite attempt gets countered (zombie neutralized)."),
            ("kappa", 0.0, 0.3, sim.policy.kappa, "Rate infected are moved to quarantine."),
            ("sigma", 0.0, 0.3, sim.policy.sigma, "Rate zombies are moved to quarantine."),
            ("cure_rate", 0.0, 0.6, sim.policy.cure_rate, "Hospital cure rate for infected per tick."),
            ("neutralize_rate", 0.0, 0.6, sim.policy.neutralize_rate, "Hospital neutralization rate for zombies per tick."),
            ("notice_radius", 0.5, 8.0, sim.policy.notice_radius, "Humans start fleeing if a zombie is within this radius."),
            ("human_speed", 0.02, 0.4, sim.policy.human_speed, "Human walking speed."),
            ("zombie_speed", 0.02, 0.4, sim.policy.zombie_speed, "Zombie walking speed."),
            ("daily_food_need", 0.0, 3.0, sim.policy.daily_food_need, "Food consumed per tick by each human."),
            ("starvation_limit", 0, 10, sim.policy.starvation_limit, "Ticks a human can go hungry before removal."),
            ("trade_per_step", 0, 200, sim.policy.trade_per_step, "Max market trades per tick."),
            ("food_regrow", 0.0, 0.8, sim.policy.food_regrow_chance, "Per-node chance food regenerates per tick."),
            ("spawn_chance", 0.0, 0.8, sim.policy.spawn_chance_per_step, "Probability a new resource node spawns per tick."),
            ("harvest_radius", 0.2, 1.5, sim.policy.harvest_radius, "Distance within which harvesting/mining progresses."),
            ("t_harvest_food", 1, 30, sim.policy.t_harvest_food, "Ticks to gather 1 food when harvesting."),
            ("t_harvest_supplies", 1, 30, sim.policy.t_harvest_supplies, "Ticks to gather 1 supplies."),
            ("t_mine_scrap", 1, 40, sim.policy.t_mine_scrap, "Ticks to mine 1 scrap."),
            ("t_mine_chems", 1, 40, sim.policy.t_mine_chems, "Ticks to mine 1 chems."),
            ("t_mine_elec", 1, 50, sim.policy.t_mine_electronics, "Ticks to mine 1 electronics."),
            ("carry_capacity", 4, 40, sim.policy.carry_capacity, "Max items an agent can carry before depositing."),
            ("deposit_time", 1, 40, sim.policy.storage_deposit_time, "Ticks spent depositing at a storage depot."),
        ]

        ncols = 6
        nrows = int(math.ceil(len(slider_specs) / ncols))
        gs_sl = self.fig.add_gridspec(nrows, ncols, left=0.06, right=0.96, bottom=0.06, top=0.28, hspace=0.55, wspace=0.35)

        self.sliders: Dict[str, Slider] = {}
        self.slider_help: Dict[str, str] = {}
        self.slider_axes = []
        for i, (name, lo, hi, val, helptext) in enumerate(slider_specs):
            r, c = divmod(i, ncols)
            ax = self.fig.add_subplot(gs_sl[r, c])
            valfmt = '%0.0f' if isinstance(val, int) or name.startswith('t_') or name in ('trade_per_step','starvation_limit','carry_capacity','deposit_time') else None
            s = Slider(ax, name, lo, hi, valinit=val, valfmt=valfmt)
            s.on_changed(self._on_slider)
            self.sliders[name] = s
            self.slider_help[name] = helptext
            self.slider_axes.append((ax, name))

        # Hover tooltip annotation (hidden by default)
        self.tooltip = self.ax_avg.annotate("", xy=(0,0), xytext=(15,15), textcoords='offset points',
                                            bbox=dict(boxstyle='round', fc='w', ec='0.5', alpha=0.95),
                                            ha='left', va='bottom')
        self.tooltip.set_visible(False)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        # Start/Pause button (simulation is PAUSED initially)
        ax_btn = self.fig.add_axes([0.86, 0.91, 0.1, 0.05])
        self.btn = Button(ax_btn, 'Start / Pause')
        self.btn.on_clicked(self._toggle_run)
        self.running = False

        # Legends on right map (state + nodes + activity)
        st_h, act_h, node_h = self.sim.legend_handles()
        leg1 = self.ax_map_right.legend(handles=st_h, loc='upper left', title='Agent state')
        self.ax_map_right.add_artist(leg1)
        leg2 = self.ax_map_right.legend(handles=node_h, loc='lower left', title='Resource nodes')
        leg3 = self.ax_map_right.legend(handles=act_h, loc='center left', title='Activity')
        for lg in (leg1, leg2, leg3):
            lg.get_frame().set_alpha(0.9)

        # Animation; start stopped so user can tune first
        self.ani = FuncAnimation(self.fig, self._tick, interval=30)
        self.ani.event_source.stop()
        self._draw_once()  # render initial frame with current params

    def _on_motion(self, event):
        # Show tooltip if hovering a slider axis
        vis = False
        if event.inaxes is not None:
            for ax, name in self.slider_axes:
                if event.inaxes is ax:
                    txt = self.slider_help.get(name, "")
                    if txt:
                        self.tooltip.xy = (event.xdata, event.ydata)
                        self.tooltip.set_text(f"{name}: {txt}")
                        self.tooltip.set_visible(True)
                        vis = True
                        break
        if not vis:
            self.tooltip.set_visible(False)
        self.fig.canvas.draw_idle()

    def _toggle_run(self, _):
        self.running = not self.running
        if self.running:
            self.ani.event_source.start()
        else:
            self.ani.event_source.stop()

    def _on_slider(self, _):
        p = self.sim.policy
        # Epidemic
        p.beta = float(self.sliders['beta'].val)
        p.rho = float(self.sliders['rho'].val)
        p.alpha = float(self.sliders['alpha'].val)
        p.kappa = float(self.sliders['kappa'].val)
        p.sigma = float(self.sliders['sigma'].val)
        p.cure_rate = float(self.sliders['cure_rate'].val)
        p.neutralize_rate = float(self.sliders['neutralize_rate'].val)
        # Movement/behavior
        p.notice_radius = float(self.sliders['notice_radius'].val)
        p.human_speed = float(self.sliders['human_speed'].val)
        p.zombie_speed = float(self.sliders['zombie_speed'].val)
        # Economy
        p.daily_food_need = float(self.sliders['daily_food_need'].val)
        p.starvation_limit = int(self.sliders['starvation_limit'].val)
        p.trade_per_step = int(self.sliders['trade_per_step'].val)
        # Resources
        p.food_regrow_chance = float(self.sliders['food_regrow'].val)
        p.spawn_chance_per_step = float(self.sliders['spawn_chance'].val)
        p.harvest_radius = float(self.sliders['harvest_radius'].val)
        p.t_harvest_food = int(self.sliders['t_harvest_food'].val)
        p.t_harvest_supplies = int(self.sliders['t_harvest_supplies'].val)
        p.t_mine_scrap = int(self.sliders['t_mine_scrap'].val)
        p.t_mine_chems = int(self.sliders['t_mine_chems'].val)
        p.t_mine_electronics = int(self.sliders['t_mine_elec'].val)
        p.carry_capacity = int(self.sliders['carry_capacity'].val)
        p.storage_deposit_time = int(self.sliders['deposit_time'].val)
        # Redraw static frame if paused
        if not self.running:
            self._draw_once()

    def _tick(self, _):
        if not self.running:
            return
        self.sim.step()
        self._draw_once()

    def _draw_once(self):
        # LEFT MAP (main)
        self.ax_map_left.clear()
        title_left = f"Map A — t={self.sim.t}  |  Infra: B={self.sim.infra['barricades']} M={self.sim.infra['medkits']} T={self.sim.infra['traps']}"
        self.sim.draw_agent_map(self.ax_map_left, title_left)

        # CENTER TS
        self.ax_ts.clear()
        for key, label in [("S","Humans"),("I","Infected"),("Z","Zombies"),("Q","Quarantine"),("R","Removed")]:
            self.ax_ts.plot(self.sim.history.get(key, []), label=label, color=COLORS['state'][key[0]])
        self.ax_ts.legend(loc='upper right')
        self.ax_ts.set_xlabel("Step"); self.ax_ts.set_ylabel("Count")
        self.ax_ts.set_title("Population states")

        # RIGHT MAP (mirror + legend)
        self.ax_map_right.clear()
        self.sim.draw_agent_map(self.ax_map_right, "Map B — legend synced")
        st_h, act_h, node_h = self.sim.legend_handles()
        leg1 = self.ax_map_right.legend(handles=st_h, loc='upper left', title='Agent state')
        self.ax_map_right.add_artist(leg1)
        leg2 = self.ax_map_right.legend(handles=node_h, loc='lower left', title='Resource nodes')
        leg3 = self.ax_map_right.legend(handles=act_h, loc='center left', title='Activity')
        for lg in (leg1, leg2, leg3):
            lg.get_frame().set_alpha(0.9)

        # AVERAGES (bottom row)
        self.ax_avg.clear()
        self.ax_avg.plot(self.sim.avgs['food'], label='Avg Food')
        self.ax_avg.plot(self.sim.avgs['supplies'], label='Avg Supplies')
        self.ax_avg.plot(self.sim.avgs['wealth'], label='Avg Wealth')
        self.ax_avg.legend(loc='upper right')
        self.ax_avg.set_xlabel("Step"); self.ax_avg.set_ylabel("Average per alive agent")
        self.ax_avg.set_title("Economic averages")

        self.fig.canvas.draw_idle()

# ------------------
# Runner / animator
# ------------------

def run_demo():
    sim = ZombieSim(n_agents=85, initial_zombies=16)  # start with 0 to test pre-run behavior
    _ = ZombieUI(sim)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()
