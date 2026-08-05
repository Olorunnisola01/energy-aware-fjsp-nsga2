from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import numpy as np

# Define simple FJSP problem
class TinyFJSP(Problem):
    def __init__(self):
        # 4 operations: J1O1, J1O2, J2O1, J2O2
        # Each operation will have a machine assignment [0 or 1]
        # and a position in the operation sequence (0-3)
        super().__init__(n_var=8, n_obj=2, n_constr=0,
                         xl=0, xu=3, type_var=int)

        # Operation details: (job_id, op_id, [M1_time, M2_time])
        self.operations = [
            (0, 0, [3, 5]),  # J1O1
            (0, 1, [2, 4]),  # J1O2
            (1, 0, [4, 3]),  # J2O1
            (1, 1, [2, 5])   # J2O2
        ]

        self.energy_rate = [1, 2]  # M1 = 1 unit/time, M2 = 2 units/time

    def decode(self, X):
        # Split decision vector into machine assignments and operation order
        machine_assign = [int(m) for m in X[:4]]
        order = [int(o) for o in X[4:]]

        # Map operation index to order
        op_order = sorted([(idx, order[idx]) for idx in range(4)], key=lambda x: x[1])
        sequence = [idx for idx, _ in op_order]  # sequence of op indices

        return machine_assign, sequence

    def simulate(self, machine_assign, sequence):
        machine_available = [0, 0]  # M1, M2
        job_ready = [0, 0]  # J1, J2
        end_times = [0] * 4
        energy_used = [0, 0]

        for op_idx in sequence:
            job_id, op_id, times = self.operations[op_idx]
            machine = int(machine_assign[op_idx] % 2)  # Ensure machine is an int
            proc_time = times[machine]

            start_time = max(machine_available[machine], job_ready[job_id])
            end_time = start_time + proc_time

            machine_available[machine] = end_time
            job_ready[job_id] = end_time

            end_times[op_idx] = end_time
            energy_used[machine] += proc_time * self.energy_rate[machine]

        makespan = max(end_times)
        total_energy = sum(energy_used)
        return makespan, total_energy

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for row in X:
            ma, seq = self.decode(row)
            ms, energy = self.simulate(ma, seq)
            F.append([ms, energy])
        out["F"] = np.array(F)


# Local search operator using greedy refinement strategy
def local_search(problem, solution):
    # solution is a 1D array of length 8 (4 machine assignments + 4 operation orders)
    best_solution = solution.copy()
    best_obj = problem.evaluate_single(best_solution)

    improved = True
    while improved:
        improved = False
        # Try swapping machine assignments for each operation
        for i in range(4):
            neighbor = best_solution.copy()
            # Flip machine assignment between 0 and 1 (mod 2)
            neighbor[i] = (neighbor[i] + 1) % 2
            obj = problem.evaluate_single(neighbor)
            if dominates(obj, best_obj):
                best_solution = neighbor
                best_obj = obj
                improved = True
                break  # restart search after improvement

        if improved:
            continue

        # Try swapping operation order positions
        for i in range(4):
            for j in range(i+1, 4):
                neighbor = best_solution.copy()
                # Swap order positions in the second half of the solution vector
                neighbor[4+i], neighbor[4+j] = neighbor[4+j], neighbor[4+i]
                obj = problem.evaluate_single(neighbor)
                if dominates(obj, best_obj):
                    best_solution = neighbor
                    best_obj = obj
                    improved = True
                    break
            if improved:
                break

    return best_solution

def dominates(obj1, obj2):
    # Check if obj1 dominates obj2 (minimization)
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))

# Extend NSGA2 to include local search
class NSGA2WithLocalSearch(NSGA2):
    def _next(self):
        # Generate offspring population
        off = super()._next()

        # Apply local search to each offspring
        for i in range(len(off)):
            off[i] = local_search(self.problem, off[i])

        return off


# Run NSGA-II with local search on TinyFJSP
problem = TinyFJSP()

task = get_termination("n_gen", 50)

algorithm = NSGA2WithLocalSearch(pop_size=20)

res = minimize(problem,
               algorithm,
               termination=task,
               seed=1,
               verbose=True)

# Print final Pareto front
print("\nPareto Front:")
for f in res.F:
    print(f"Makespan: {f[0]}, Energy: {f[1]}")


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gantt(solution, problem):
    ma, seq = problem.decode(solution)
    machine_available = [0, 0]
    job_ready = [0, 0]
    tasks = []

    for op_idx in seq:
        job_id, op_id, times = problem.operations[op_idx]
        machine = int(ma[op_idx] % 2)
        proc_time = times[machine]
        start_time = max(machine_available[machine], job_ready[job_id])
        end_time = start_time + proc_time

        machine_available[machine] = end_time
        job_ready[job_id] = end_time

        tasks.append({
            'Job': f'J{job_id+1}-O{op_id+1}',
            'Machine': machine,
            'Start': start_time,
            'End': end_time,
            'Color': f'C{job_id}'
        })

    fig, ax = plt.subplots()
    yticks = []
    ylabels = []

    for task in tasks:
        y = task['Machine']
        ax.barh(y, task['End'] - task['Start'], left=task['Start'],
                height=0.4, align='center', color=task['Color'], edgecolor='black')
        ax.text(task['Start'] + 0.1, y, task['Job'], va='center', ha='left', fontsize=8, color='white')
        if y not in yticks:
            yticks.append(y)
            ylabels.append(f'M{y+1}')

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('Time')
    ax.set_title('Gantt Chart - Selected Schedule')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# 🔧 Choose one solution (e.g., best makespan)
best_index = np.argmin(res.F[:, 0])  # or choose manually
best_solution = res.X[best_index]

# Plot the Gantt chart
plot_gantt(best_solution, problem)
