"""
Batch experiments for the Zombie Economy sim — with extra spacing and a scrollable viewer.

What changed in this version
----------------------------
• Increased subplot spacing (larger hspace/wspace, bigger figure) so labels/legends don't overlap.
• Added a Tkinter-based scrollable viewer with two tabs:
    1) Per‑Run Time Series (epidemic + economic)
    2) Summary Bar Charts
  Each tab embeds a full‑resolution Matplotlib figure in a vertically scrollable canvas.

How to run
----------
python zombie_batch_runner.py

If your sim module is named differently, update the import line:
from zombie_sim_with_pause_tooltips_storage import ZombieSim, Policy, HealthState
"""

import math
import copy
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use TkAgg to enable the scrollable Tk viewer
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# <<< UPDATE THIS IMPORT IF YOUR FILE NAME IS DIFFERENT >>>
from zombie_sim import ZombieSim, Policy, HealthState

# ------------------------------------------------------------
# Utility: run one simulation for T steps and collect metrics
# ------------------------------------------------------------

def run_one(label: str, overrides: Dict[str, Any], n_agents: int = 85,
            initial_zombies: int = 3, steps: int = 110, seed: int = 7):
    """Run a single sim with policy overrides.
    Special override keys handled here (not part of Policy):
      - initial_zombies: int (overrides function arg)
      - n_agents: int (overrides function arg)
      - seed: int (overrides function arg)
    """
    ov = dict(overrides) if overrides else {}
    if 'initial_zombies' in ov:
        initial_zombies = int(ov.pop('initial_zombies'))
    if 'n_agents' in ov:
        n_agents = int(ov.pop('n_agents'))
    if 'seed' in ov:
        seed = int(ov.pop('seed'))

    base_p = Policy()
    for k, v in ov.items():
        if hasattr(base_p, k):
            setattr(base_p, k, v)

    sim = ZombieSim(n_agents=n_agents, initial_zombies=initial_zombies, policy=base_p, seed=seed)

    # Track unique infected
    orig_become_infected = sim._become_infected
    def hook_become_infected(agent):
        agent.ever_infected = True
        return orig_become_infected(agent)
    sim._become_infected = hook_become_infected
    for a in sim.market.agents:
        a.ever_infected = False

    for _ in range(steps):
        sim.step()

    final_S = sim.history['S'][-1] if sim.history['S'] else 0
    final_I = sim.history['I'][-1] if sim.history['I'] else 0
    final_Z = sim.history['Z'][-1] if sim.history['Z'] else 0
    final_Q = sim.history['Q'][-1] if sim.history['Q'] else 0
    final_R = sim.history['R'][-1] if sim.history['R'] else 0

    final_humans = final_S + final_I + final_Q
    final_zombies = final_Z
    final_total_wealth = float(sum(a.wealth for a in sim.market.agents))
    total_unique_infected = int(sum(1 for a in sim.market.agents if getattr(a, 'ever_infected', False)))

    return {
        'label': label,
        'overrides_effective': overrides,
        'sim': sim,
        'history': copy.deepcopy(sim.history),
        'avgs': copy.deepcopy(sim.avgs),
        'final': {
            'S': final_S, 'I': final_I, 'Z': final_Z, 'Q': final_Q, 'R': final_R,
            'humans': final_humans, 'zombies': final_zombies,
            'total_wealth': final_total_wealth,
            'unique_infected': total_unique_infected,
        }
    }

# ------------------------------------------------------------
# Experiment catalog (includes zombie‑dominance scenarios)
# ------------------------------------------------------------

EXPERIMENTS_BASE: List[Tuple[str, Dict[str, Any]]] = [
    ("Baseline", {}),
    ("High Transmission (beta↑)", {"beta": 0.02}),
    ("Fast Progression (rho↑)", {"rho": 0.02}),
    ("Strong Quarantine (kappa↑, sigma↑)", {"kappa": 0.08, "sigma": 0.08}),
    ("Strong Hospitals (cure↑, neut↑)", {"cure_rate": 0.25, "neutralize_rate": 0.25}),
    ("High Regrowth/Spawn", {"food_regrow_chance": 0.5, "spawn_chance_per_step": 0.4}),
    ("Tight Carry Cap (cap↓)", {"carry_capacity": 6}),
    ("Fast Humans (speed↑)", {"human_speed": 0.25, "flee_speed": 0.35}),
]

EXPERIMENTS_OVERRUN: List[Tuple[str, Dict[str, Any]]] = [
    ("Overrun: Super‑spreaders", {"beta": 0.04, "rho": 0.03, "alpha": 0.0005, "kappa": 0.0, "sigma": 0.0,
                                   "cure_rate": 0.0, "neutralize_rate": 0.0, "human_speed": 0.09, "flee_speed": 0.12,
                                   "zombie_speed": 0.22, "notice_radius": 1.2, "bite_distance": 0.8,
                                   "initial_zombies": 10}),
    ("Overrun: Fast Z, Slow H", {"beta": 0.03, "rho": 0.02, "alpha": 0.0, "kappa": 0.0, "sigma": 0.0,
                                  "cure_rate": 0.0, "neutralize_rate": 0.0, "human_speed": 0.08, "flee_speed": 0.12,
                                  "zombie_speed": 0.30, "notice_radius": 1.0, "bite_distance": 0.9,
                                  "initial_zombies": 15, "hospital_capacity": 0}),
    ("Overrun: Resurrection Plague", {"beta": 0.028, "rho": 0.02, "zeta": 0.02, "alpha": 0.001, "kappa": 0.0, "sigma": 0.0,
                                       "cure_rate": 0.0, "neutralize_rate": 0.0, "human_speed": 0.10, "zombie_speed": 0.24,
                                       "initial_zombies": 12}),
    ("Overrun: Resource Stress + Spread", {"beta": 0.03, "rho": 0.02, "alpha": 0.001, "kappa": 0.0, "sigma": 0.0,
                                            "cure_rate": 0.0, "neutralize_rate": 0.0, "human_speed": 0.10, "flee_speed": 0.14,
                                            "zombie_speed": 0.24, "daily_food_need": 1.6, "carry_capacity": 8,
                                            "initial_zombies": 12}),
    ("Overrun: Tiny Human Edge Removed", {"beta": 0.026, "rho": 0.018, "alpha": 0.0, "kappa": 0.0, "sigma": 0.0,
                                           "cure_rate": 0.0, "neutralize_rate": 0.0, "human_speed": 0.11, "zombie_speed": 0.26,
                                           "notice_radius": 0.9, "bite_distance": 0.85, "initial_zombies": 20, "n_agents": 80}),
]
EXPERIMENTS_HUMAN_SURVIVE = [
    ("Humans Win: Strong Quarantine + Care",
     {"beta": 0.006, "rho": 0.003, "zeta": 0.0,
      "kappa": 0.14, "sigma": 0.10,                # aggressive quarantine of I and Z
      "hospital_capacity": 30, "cure_rate": 0.35, "neutralize_rate": 0.25,
      "bite_distance": 0.45, "notice_radius": 5.0,
      "human_speed": 0.22, "flee_speed": 0.35, "zombie_speed": 0.10,
      "food_regrow_chance": 0.55, "spawn_chance_per_step": 0.35,
      "carry_capacity": 18, "daily_food_need": 0.85,
      "initial_zombies": 2}),

    ("Humans Win: Contact Avoidance",
     {"beta": 0.005, "rho": 0.004, "zeta": 0.0,
      "kappa": 0.10, "sigma": 0.08,
      "bite_distance": 0.35, "notice_radius": 6.0, # see early, bite needs close contact
      "human_speed": 0.26, "flee_speed": 0.38, "zombie_speed": 0.09,
      "hospital_capacity": 20, "cure_rate": 0.20, "neutralize_rate": 0.18,
      "food_regrow_chance": 0.6, "spawn_chance_per_step": 0.3,
      "carry_capacity": 20, "daily_food_need": 0.9,
      "initial_zombies": 1}),

    ("Humans Win: Hospital Surge + Traps",
     {"beta": 0.007, "rho": 0.004, "zeta": 0.0,
      "kappa": 0.08, "sigma": 0.12,                # capture zombies fast
      "hospital_capacity": 40, "cure_rate": 0.45, "neutralize_rate": 0.35,
      "human_speed": 0.24, "flee_speed": 0.36, "zombie_speed": 0.12,
      "bite_distance": 0.5, "notice_radius": 4.5,
      "food_regrow_chance": 0.5, "spawn_chance_per_step": 0.35,
      "carry_capacity": 16, "daily_food_need": 0.95,
      "initial_zombies": 3}),

    ("Humans Win: Impulsive Strikes",
     {"beta": 0.008, "rho": 0.004, "zeta": 0.0,
      "kappa": 0.10, "sigma": 0.08,
      "hospital_capacity": 25, "cure_rate": 0.30, "neutralize_rate": 0.30,
      "strike_every_steps": 45, "kill_ratio_per_strike": 0.28,  # periodic eradication
      "human_speed": 0.23, "flee_speed": 0.34, "zombie_speed": 0.12,
      "bite_distance": 0.5, "notice_radius": 4.0,
      "food_regrow_chance": 0.55, "spawn_chance_per_step": 0.32,
      "carry_capacity": 18, "daily_food_need": 0.9,
      "initial_zombies": 3}),

    ("Humans Win: Resource Abundance (less exposure)",
     {"beta": 0.007, "rho": 0.0035, "zeta": 0.0,
      "kappa": 0.09, "sigma": 0.08,
      "hospital_capacity": 18, "cure_rate": 0.25, "neutralize_rate": 0.20,
      "human_speed": 0.22, "flee_speed": 0.34, "zombie_speed": 0.11,
      "bite_distance": 0.5, "notice_radius": 4.2,
      "food_regrow_chance": 0.7, "spawn_chance_per_step": 0.45,  # fewer trips needed
      "carry_capacity": 22, "daily_food_need": 0.85,
      "initial_zombies": 2}),

    ("Humans Win: Low Progression + Capture",
     {"beta": 0.009, "rho": 0.002, "zeta": 0.0,
      "kappa": 0.12, "sigma": 0.12,                 # both I and Z get removed fast
      "hospital_capacity": 22, "cure_rate": 0.30, "neutralize_rate": 0.28,
      "human_speed": 0.24, "flee_speed": 0.36, "zombie_speed": 0.11,
      "bite_distance": 0.45, "notice_radius": 5.0,
      "food_regrow_chance": 0.55, "spawn_chance_per_step": 0.35,
      "carry_capacity": 18, "daily_food_need": 0.9,
      "initial_zombies": 3}),

    ("Humans Win: Zero Resurrection + Counter-bite",
     {"beta": 0.008, "rho": 0.0035, "zeta": 0.0,
      "alpha": 0.02,                                 # human counter-bite success up
      "kappa": 0.10, "sigma": 0.09,
      "hospital_capacity": 24, "cure_rate": 0.30, "neutralize_rate": 0.22,
      "human_speed": 0.24, "flee_speed": 0.36, "zombie_speed": 0.12,
      "bite_distance": 0.48, "notice_radius": 4.8,
      "food_regrow_chance": 0.55, "spawn_chance_per_step": 0.34,
      "carry_capacity": 18, "daily_food_need": 0.9,
      "initial_zombies": 2}),

    ("Humans Win: Quarantine + Release (rehab)",
     {"beta": 0.0075, "rho": 0.0035, "zeta": 0.0,
      "kappa": 0.14, "sigma": 0.10, "gamma_release": 0.02,   # some Q return to S
      "hospital_capacity": 26, "cure_rate": 0.32, "neutralize_rate": 0.22,
      "human_speed": 0.23, "flee_speed": 0.35, "zombie_speed": 0.11,
      "bite_distance": 0.48, "notice_radius": 4.5,
      "food_regrow_chance": 0.6, "spawn_chance_per_step": 0.32,
      "carry_capacity": 20, "daily_food_need": 0.9,
      "initial_zombies": 3}),
]
EXPERIMENTS: List[Tuple[str, Dict[str, Any]]] = EXPERIMENTS_BASE + EXPERIMENTS_OVERRUN + EXPERIMENTS_HUMAN_SURVIVE

# ------------------------------------------------------------
# Figures (spacing bumped) + Scrollable viewer
# ------------------------------------------------------------

def make_figures(results: List[Dict[str, Any]], steps: int):
    # ---- Figure 1: Per‑run time series with more spacing
    n = len(results)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig1 = plt.figure(figsize=(16, 5.6*rows))
    gs1 = fig1.add_gridspec(rows, cols, hspace=0.6, wspace=0.35)

    for i, r in enumerate(results):
        row, col = divmod(i, cols)
        ax1 = fig1.add_subplot(gs1[row, col])
        H = r['history']
        ax1.plot(H['S'], label='Humans (S)')
        ax1.plot(H['I'], label='Infected (I)')
        ax1.plot(H['Z'], label='Zombies (Z)')
        ax1.plot(H['Q'], label='Quarantine (Q)')
        ax1.plot(H['R'], label='Removed (R)')
        ax1.set_title(f"{r['label']} — Epidemic & Economics (T={steps})")
        ax1.set_xlabel('Step'); ax1.set_ylabel('Count')

        ax2 = ax1.twinx()
        A = r['avgs']
        ax2.plot(A['food'], linestyle='--', label='Avg Food')
        ax2.plot(A['supplies'], linestyle='--', label='Avg Supplies')
        ax2.plot(A['wealth'], linestyle='--', label='Avg Wealth')
        ax2.set_ylabel('Economic avgs')

        # Combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=8, framealpha=0.9)

        # Footer subtitle with tested overrides
        ov_pairs = ", ".join(f"{k}={v}" for k,v in r['overrides_effective'].items()) if r['overrides_effective'] else "(default policy)"
        ax1.text(0.02, -0.24, f"Tested: {ov_pairs}", transform=ax1.transAxes, fontsize=9, va='top')

    fig1.suptitle("Zombie Economy — Per‑Run Time Series", y=0.995)

    # ---- Figure 2: Summary dashboards laid out 2x2 with generous spacing
    # Panels:
    #  (0,0) Humans vs Zombies (stacked)
    #  (0,1) Final Total Wealth
    #  (1,0) Total Unique Infected (ever)
    #  (1,1) Final Death Causes (Zombified vs Removed)
    fig2, axes = plt.subplots(2, 2, figsize=(22, 12))
    plt.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.12, top=0.90)

    labels = [r['label'] for r in results]
    humans = [r['final']['humans'] for r in results]
    zombies = [r['final']['zombies'] for r in results]
    wealths = [r['final']['total_wealth'] for r in results]
    uniq_inf = [r['final']['unique_infected'] for r in results]

    # For "death causes": treat final zombies as "Zombified" and final removed as "Removed"
    removed = [r['final']['R'] for r in results]
    zombified = zombies  # interpret becoming/remaining Z at T as a fatal outcome category

    x = np.arange(len(labels))

    # (0,0) Humans vs Zombies (stacked)
    ax = axes[0,0]
    ax.bar(x, humans, label='Humans')
    ax.bar(x, zombies, bottom=humans, label='Zombies')
    ax.set_title('Final counts: Humans vs Zombies')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Count'); ax.legend()

    # (0,1) Final total wealth
    ax = axes[0,1]
    ax.bar(x, wealths)
    ax.set_title('Final Total Wealth by Run')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Wealth (sum)')

    # (1,0) Unique infected over run
    ax = axes[1,0]
    ax.bar(x, uniq_inf)
    ax.set_title('Total Unique Infected (ever) by Run')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Agents')

    # (1,1) Death causes summary (final state interpretation)
    ax = axes[1,1]
    ax.bar(x, removed, label='Removed (neutralized/other)')
    ax.bar(x, zombified, bottom=removed, label='Zombified (final Z)')
    ax.set_title('Final “Deaths” by Cause (interpretation at T)')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Agents'); ax.legend()

    fig2.suptitle(f"Summary after {steps} steps per run — arranged 2×2 with extra spacing")

    return fig1, fig2

# ------------------------------------------------------------
# Tk Scrollable viewer with tabs
# ------------------------------------------------------------

class ScrollableFigureApp:
    def __init__(self, fig_list: List[Tuple[str, plt.Figure]], window_title: str = "Zombie Batch Results"):
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1200x800")  # starting size; content can be taller and scrolled

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True)

        for tab_title, fig in fig_list:
            self._add_tab(tab_title, fig)

    def _add_tab(self, title: str, fig: plt.Figure):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=title)

        # Create a canvas + vertical scrollbar
        canvas = tk.Canvas(frame, borderwidth=0)
        vscroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame inside the canvas to hold the figure canvas widget
        inner = ttk.Frame(canvas)
        # Add the inner frame to the canvas
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        # Embed the Matplotlib figure into the inner frame
        fig_canvas = FigureCanvasTkAgg(fig, master=inner)
        widget = fig_canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        fig_canvas.draw()

        # Update scrollregion when the inner frame size changes
        def _configure_inner(event):
            # Update the scrollregion to match the size of the inner frame
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind('<Configure>', _configure_inner)

        # Also update the inner window's width when the canvas width changes (so it resizes horizontally with the window)
        def _configure_canvas(event):
            canvas.itemconfig(inner_id, width=event.width)
        canvas.bind('<Configure>', _configure_canvas)

        # Mousewheel scrolling on Windows/Mac/Linux
        def _on_mousewheel(event):
            # On Windows, event.delta is multiples of 120; on Mac, it can be small; on X11, use Button-4/5
            delta = -1 * (event.delta // 120 if event.delta else 1)
            canvas.yview_scroll(delta, "units")
        # Bind for Windows and MacOS
        widget.bind("<Enter>", lambda e: widget.bind_all("<MouseWheel>", _on_mousewheel))
        widget.bind("<Leave>", lambda e: widget.unbind_all("<MouseWheel>"))
        # Bind for Linux (wheel up/down)
        widget.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        widget.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    def run(self):
        self.root.mainloop()

# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------

def run_all_and_plot(experiments=None, steps=210, n_agents=85, initial_zombies=3, seed=7, use_scroll_viewer=True):
    if experiments is None:
        experiments = EXPERIMENTS

    results = []
    for label, overrides in experiments:
        r = run_one(label, overrides, n_agents=n_agents, initial_zombies=initial_zombies, steps=steps, seed=seed)
        results.append(r)

    fig1, fig2 = make_figures(results, steps)

    if use_scroll_viewer:
        app = ScrollableFigureApp([
            ("Per‑Run Time Series", fig1),
            ("Summary Bar Charts", fig2),
        ])
        app.run()
    else:
        # Fall back to standard plt.show() for both figures
        plt.show()


if __name__ == "__main__":
    run_all_and_plot()
