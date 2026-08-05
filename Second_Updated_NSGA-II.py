from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# Load electricity price data from CSV and preprocess
price_df = pd.read_csv("belgium_prices_feb2022.csv", parse_dates=["timestamp"])
price_df["minute"] = price_df["timestamp"].dt.floor("min")
price_per_minute = price_df.set_index("minute")["price_eur_per_kWh"].resample("min").mean().ffill()
price_array = price_per_minute.values

def generate_fjsp_instance(n_jobs=15, n_machines=4, min_ops=5, max_ops=10, min_time=5, max_time=10):
    """
    Generate a random large-scale Flexible Job Shop Problem (FJSP) instance.

    Parameters:
        n_jobs (int): Number of jobs.
        n_machines (int): Number of machines.
        min_ops (int): Minimum operations per job.
        max_ops (int): Maximum operations per job.
        min_time (int): Minimum processing time.
        max_time (int): Maximum processing time.

    Returns:
        operations (list): List of tuples (job_id, op_id, proc_times).
        n_jobs (int): Number of jobs.
        n_machines (int): Number of machines.
    """
    operations = []
    for job_id in range(n_jobs):
        n_ops = random.randint(min_ops, max_ops)
        for op_id in range(n_ops):
            # Each operation can be processed on 2-3 randomly selected machines
            machines = random.sample(range(n_machines), random.randint(2, 3))
            proc_times = [None] * n_machines
            for m in machines:
                proc_times[m] = random.randint(min_time, max_time)
            operations.append((job_id, op_id, proc_times))
    return operations, n_jobs, n_machines

class LargeFJSP(Problem):
    """
    Large-scale Flexible Job Shop Problem (FJSP) definition for multi-objective optimization.

    Objectives:
        1. Minimize makespan (total completion time).
        2. Minimize total energy cost based on electricity prices.

    Decision variables:
        - Machine assignments for each operation.
        - Operation sequencing order.
    """
    def __init__(self, operations, num_jobs, num_machines):
        self.operations = operations
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.n_ops = len(operations)
        self.energy_rate = [i + 1 for i in range(num_machines)]  # Energy rate per machine (units/time)

        super().__init__(n_var=2 * self.n_ops, n_obj=2, n_constr=0,
                         xl=0, xu=self.n_ops - 1, type_var=int)

    def decode(self, X):
        """
        Decode the decision vector into machine assignments and operation sequence.

        Parameters:
            X (array-like): Decision vector.

        Returns:
            machine_assign (list): Machine assignments for each operation.
            sequence (list): Operation execution sequence.
        """
        machine_assign = [int(m) for m in X[:self.n_ops]]
        order = [int(o) for o in X[self.n_ops:]]
        op_order = sorted([(idx, order[idx]) for idx in range(self.n_ops)], key=lambda x: x[1])
        sequence = [idx for idx, _ in op_order]
        return machine_assign, sequence

    def simulate(self, machine_assign, sequence):
        """
        Simulate the schedule to compute makespan and total energy cost.

        Parameters:
            machine_assign (list): Machine assignments for each operation.
            sequence (list): Operation execution sequence.

        Returns:
            makespan (int): Total completion time.
            total_energy (float): Total energy cost based on electricity prices.
        """
        machine_available = [0] * self.num_machines
        job_ready = [0] * self.num_jobs
        end_times = [0] * self.n_ops
        total_energy = 0.0

        for op_idx in sequence:
            job_id, op_id, times = self.operations[op_idx]
            available_machines = [m for m, t in enumerate(times) if t is not None]
            m_idx = machine_assign[op_idx] % len(available_machines)
            selected_machine = available_machines[m_idx]
            proc_time = times[selected_machine]

            start_time = max(machine_available[selected_machine], job_ready[job_id])
            end_time = start_time + proc_time

            # Calculate energy cost using electricity price per minute
            for t in range(start_time, end_time):
                if t < len(price_array):
                    total_energy += price_array[t]
                else:
                    total_energy += price_array[-1]

            machine_available[selected_machine] = end_time
            job_ready[job_id] = end_time
            end_times[op_idx] = end_time

        makespan = max(end_times)
        return makespan, total_energy

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the population of solutions.

        Parameters:
            X (array-like): Population decision vectors.
            out (dict): Output dictionary to store objective values.
        """
        F = []
        for row in X:
            ma, seq = self.decode(row)
            ms, energy = self.simulate(ma, seq)
            F.append([ms, energy])
        out["F"] = np.array(F)

def dominates(obj1, obj2):
    """
    Check if obj1 dominates obj2 in multi-objective minimization.

    Parameters:
        obj1 (list or array): Objective values of solution 1.
        obj2 (list or array): Objective values of solution 2.

    Returns:
        bool: True if obj1 dominates obj2, False otherwise.
    """
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 < o2 for o1, o2 in zip(obj1, obj2))

def local_search(problem, solution):
    """
    Perform a greedy local search to improve a solution.

    Parameters:
        problem (LargeFJSP): The FJSP problem instance.
        solution (array-like): Current solution vector.

    Returns:
        best_solution (array-like): Improved solution vector.
    """
    best_solution = solution.copy()
    best_obj = problem.evaluate_single(best_solution)
    improved = True

    while improved:
        improved = False
        # Try changing machine assignments
        for i in range(problem.n_ops):
            neighbor = best_solution.copy()
            neighbor[i] = (neighbor[i] + 1) % problem.num_machines
            obj = problem.evaluate_single(neighbor)
            if dominates(obj, best_obj):
                best_solution = neighbor
                best_obj = obj
                improved = True
                break
        if improved:
            continue
        # Try swapping operation orders
        for i in range(problem.n_ops):
            for j in range(i + 1, problem.n_ops):
                neighbor = best_solution.copy()
                neighbor[problem.n_ops + i], neighbor[problem.n_ops + j] = \
                    neighbor[problem.n_ops + j], neighbor[problem.n_ops + i]
                obj = problem.evaluate_single(neighbor)
                if dominates(obj, best_obj):
                    best_solution = neighbor
                    best_obj = obj
                    improved = True
                    break
            if improved:
                break
    return best_solution

class NSGA2WithLocalSearch(NSGA2):
    """
    NSGA-II algorithm extended with a local search operator applied to offspring.
    """
    def _next(self):
        off = super()._next()
        for i in range(len(off)):
            off[i] = local_search(self.problem, off[i])
        return off

def plot_gantt(solution, problem):
    """
    Plot a Gantt chart for a given solution.

    Parameters:
        solution (array-like): Solution vector.
        problem (LargeFJSP): Problem instance.
    """
    ma, seq = problem.decode(solution)
    machine_available = [0] * problem.num_machines
    job_ready = [0] * problem.num_jobs
    tasks = []

    for op_idx in seq:
        job_id, op_id, times = problem.operations[op_idx]
        available_machines = [m for m, t in enumerate(times) if t is not None]
        machine = ma[op_idx] % len(available_machines)
        selected_machine = available_machines[machine]
        proc_time = times[selected_machine]

        start_time = max(machine_available[selected_machine], job_ready[job_id])
        end_time = start_time + proc_time

        machine_available[selected_machine] = end_time
        job_ready[job_id] = end_time

        tasks.append({
            'Job': f'J{job_id+1}-O{op_id+1}',
            'Machine': selected_machine,
            'Start': start_time,
            'End': end_time,
            'Color': f'C{job_id % 10}'
        })

    fig, ax = plt.subplots()
    yticks = []
    ylabels = []

    # Create legend handles for jobs
    job_ids = sorted(set(task['Job'] for task in tasks))
    colors = {task['Job']: task['Color'] for task in tasks}
    legend_handles = [plt.Rectangle((0,0),1,1, color=colors[job]) for job in job_ids]

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
    ax.set_title('Gantt Chart - Large FJSP')
    ax.grid(False)  # Remove gridlines
    ax.legend(legend_handles, job_ids, title="Jobs", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_convergence(history):
    """
    Plot the convergence of the optimization algorithm over generations.

    Parameters:
        history (History): pymoo optimization history object.
    """
    gen = []
    best_makespan = []
    best_energy = []

    # pymoo history stores results in a list of Result objects
    for record in history:
        gen.append(record.n_gen)
        F = record.opt.get("F")
        best_makespan.append(np.min(F[:, 0]))
        best_energy.append(np.min(F[:, 1]))

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Makespan', color='tab:blue')
    ax1.plot(gen, best_makespan, color='tab:blue', label='Best Makespan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Best Energy', color='tab:orange')
    ax2.plot(gen, best_energy, color='tab:orange', label='Best Energy')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.suptitle('Convergence Plot')
    fig.tight_layout()
    plt.show()


def plot_pareto_front(F):
    """
    Plot the Pareto front of solutions and save as 'pareto_front.png'.

    Parameters:
        F (array-like): Objective values of solutions.
    """
    plt.figure()
    plt.scatter(F[:, 0], F[:, 1], c='blue', marker='o')
    plt.xlabel('Makespan')
    plt.ylabel('Energy Cost')
    plt.title('Pareto Front')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pareto_front.png')
    plt.show()

# Generate and solve the FJSP problem
operations, njobs, nmachs = generate_fjsp_instance()
problem = LargeFJSP(operations, njobs, nmachs)

algorithm = NSGA2WithLocalSearch(pop_size=20)
termination = get_termination("n_gen", 2000)

result = minimize(problem, algorithm, termination, seed=42, verbose=True, save_history=True)

# Print Pareto front results
print("\nPareto Front:")
for f in result.F:
    print(f"Makespan: {f[0]}, Energy: {f[1]}")

# Plot Gantt chart for best makespan solution
best_idx = np.argmin(result.F[:, 0])
plot_gantt(result.X[best_idx], problem)

# Plot convergence
plot_convergence(result.history)

# Plot Pareto front
plot_pareto_front(result.F)
