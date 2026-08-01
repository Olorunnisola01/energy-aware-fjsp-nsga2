"""
Energy-Aware Flexible Job-Shop Scheduling via NSGA-II with Local Search
========================================================================

Mini project: bi-objective optimisation of the Flexible Job-Shop Scheduling
Problem (FJSP), minimising

    (1) makespan            - total time to complete all jobs, and
    (2) energy cost         - integral of real day-ahead electricity prices
                               (Belgium, February 2022, ENTSO-E) over each
                               operation's processing window.

Method
------
A random FJSP instance (configurable number of jobs / machines / operations)
is solved with NSGA-II (Deb et al., 2002), augmented with a greedy local
search ("memetic" step) applied to every offspring each generation.

Electricity price data provides a realistic, time-varying energy cost signal
instead of a fixed per-machine energy rate, so scheduling operations during
cheap price windows is directly rewarded by the second objective.

Outputs
-------
Running this script produces, in the `outputs/` directory:
    - gantt_chart.png        Schedule for the minimum-makespan solution
    - convergence_plot.png   Best makespan / energy per generation
    - pareto_front.png       Final non-dominated front
    - pareto_front.csv       Numeric objective values of the front
    - run_summary.txt        Run configuration and headline results

Reproducibility
----------------
Both the random FJSP instance and the NSGA-II run are seeded, so results are
deterministic given the same RNG seed and pymoo version.

Usage
-----
    python nsga2_fjsp_energy.py
    python nsga2_fjsp_energy.py --n-jobs 20 --n-machines 5 --n-gen 300 --seed 7
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

SCRIPT_DIR = Path(__file__).resolve().parent
PRICE_CSV = SCRIPT_DIR / "belgium_prices_feb2022.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
BENCHMARK_DIR = SCRIPT_DIR / "benchmarks" / "brandimarte"


def load_price_series(csv_path: Path) -> np.ndarray:
    """
    Load the Belgium day-ahead electricity price series and resample it to
    a per-minute array indexed by minute offset from the start of the data.

    Parameters
    ----------
    csv_path : Path
        Path to a CSV with columns ["timestamp", "price_eur_per_kWh"].

    Returns
    -------
    np.ndarray
        Price in EUR/kWh for each minute, minute 0 being the series start.
    """
    price_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    price_df["minute"] = price_df["timestamp"].dt.floor("min")
    price_per_minute = (
        price_df.set_index("minute")["price_eur_per_kWh"]
        .resample("min")
        .mean()
        .ffill()
    )
    return price_per_minute.values


def generate_fjsp_instance(rng, n_jobs=15, n_machines=4, min_ops=5, max_ops=10,
                            min_time=5, max_time=10):
    """
    Generate a random Flexible Job-Shop Scheduling Problem (FJSP) instance.

    Parameters
    ----------
    rng : random.Random
        Seeded RNG used for all random draws, for reproducibility.
    n_jobs, n_machines : int
        Number of jobs and machines.
    min_ops, max_ops : int
        Range for the number of operations per job.
    min_time, max_time : int
        Range for per-machine processing time (minutes).

    Returns
    -------
    operations : list[tuple[int, int, list[int | None]]]
        Each entry is (job_id, op_id, proc_times) where proc_times[m] is the
        processing time on machine m, or None if that machine cannot run it.
    n_jobs, n_machines : int
        Echoed back for convenience.
    """
    operations = []
    for job_id in range(n_jobs):
        n_ops = rng.randint(min_ops, max_ops)
        for op_id in range(n_ops):
            machines = rng.sample(range(n_machines), rng.randint(2, 3))
            proc_times = [None] * n_machines
            for m in machines:
                proc_times[m] = rng.randint(min_time, max_time)
            operations.append((job_id, op_id, proc_times))
    return operations, n_jobs, n_machines


def parse_fjs_instance(path):
    """
    Parse a Flexible Job-Shop Scheduling instance in the standard (Hurink)
    ``.fjs``/text format used by the Brandimarte (1993) benchmark suite:

        <n_jobs> <n_machines>
        <n_ops_job1> <n_alt_op1> <machine> <time> [<machine> <time> ...] <n_alt_op2> ...
        ...(one line per job)

    Machine indices in the file are 0-based. Returns the same
    ``(operations, n_jobs, n_machines)`` structure as
    `generate_fjsp_instance`, so it is a drop-in replacement for random
    instance generation.

    Parameters
    ----------
    path : Path
        Path to the instance file.

    Returns
    -------
    operations : list[tuple[int, int, list[int | None]]]
    n_jobs, n_machines : int
    """
    lines = [line.split() for line in Path(path).read_text().strip().splitlines()]
    n_jobs, n_machines = int(lines[0][0]), int(lines[0][1])

    operations = []
    for job_id in range(n_jobs):
        toks = [int(t) for t in lines[1 + job_id]]
        idx = 0
        n_ops = toks[idx]; idx += 1
        for op_id in range(n_ops):
            n_alt = toks[idx]; idx += 1
            proc_times = [None] * n_machines
            for _ in range(n_alt):
                machine, time = toks[idx], toks[idx + 1]
                idx += 2
                proc_times[machine] = time
            operations.append((job_id, op_id, proc_times))

    return operations, n_jobs, n_machines


class EnergyAwareFJSP(Problem):
    """
    Bi-objective FJSP: minimise (makespan, electricity cost).

    Decision vector layout (length 2 * n_ops):
        X[0 : n_ops]        machine-choice index per operation (encoded
                             modulo the number of eligible machines for
                             that operation, so any integer is valid)
        X[n_ops : 2*n_ops]  sort key used to derive the operation execution
                             sequence (ascending order of these keys)
    """

    def __init__(self, operations, num_jobs, num_machines, price_array):
        self.operations = operations
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.n_ops = len(operations)
        self.price_array = price_array

        super().__init__(n_var=2 * self.n_ops, n_obj=2, n_constr=0,
                          xl=0, xu=self.n_ops - 1, type_var=int)

    def decode(self, x):
        """Split a decision vector into (machine_assign, sequence)."""
        machine_assign = [int(m) for m in x[:self.n_ops]]
        order = [int(o) for o in x[self.n_ops:]]
        op_order = sorted(range(self.n_ops), key=lambda idx: order[idx])
        return machine_assign, op_order

    def simulate(self, machine_assign, sequence):
        """
        Simulate a schedule and compute (makespan, energy_cost).

        Energy cost is the sum, over every operation, of the real
        electricity price integrated (per-minute) across that operation's
        [start, end) processing window.
        """
        machine_available = [0] * self.num_machines
        job_ready = [0] * self.num_jobs
        end_times = [0] * self.n_ops
        total_energy = 0.0
        price_len = len(self.price_array)

        for op_idx in sequence:
            job_id, op_id, times = self.operations[op_idx]
            available_machines = [m for m, t in enumerate(times) if t is not None]
            m_idx = machine_assign[op_idx] % len(available_machines)
            selected_machine = available_machines[m_idx]
            proc_time = times[selected_machine]

            start_time = max(machine_available[selected_machine], job_ready[job_id])
            end_time = start_time + proc_time

            if end_time <= price_len:
                total_energy += self.price_array[start_time:end_time].sum()
            else:
                in_range = max(0, price_len - start_time)
                total_energy += self.price_array[start_time:start_time + in_range].sum()
                total_energy += self.price_array[-1] * (proc_time - in_range)

            machine_available[selected_machine] = end_time
            job_ready[job_id] = end_time
            end_times[op_idx] = end_time

        return max(end_times), total_energy

    def evaluate_single(self, x):
        """Evaluate one decision vector, returning objectives as an array."""
        machine_assign, sequence = self.decode(x)
        makespan, energy = self.simulate(machine_assign, sequence)
        return np.array([makespan, energy])

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.evaluate_single(row) for row in X])


def dominates(obj1, obj2):
    """Return True if obj1 Pareto-dominates obj2 (minimisation)."""
    return all(a <= b for a, b in zip(obj1, obj2)) and any(a < b for a, b in zip(obj1, obj2))


def local_search(problem, solution):
    """
    Greedy neighbourhood local search ("memetic" refinement step).

    Explores two neighbourhoods in turn until neither yields a dominating
    move: (1) re-assigning each operation's machine, and (2) swapping the
    sequencing key of each pair of operations.

    Parameters
    ----------
    problem : EnergyAwareFJSP
    solution : np.ndarray
        Decision vector to refine.

    Returns
    -------
    np.ndarray
        A solution that is not dominated by any explored neighbour.
    """
    best_solution = solution.copy()
    best_obj = problem.evaluate_single(best_solution)
    improved = True

    while improved:
        improved = False
        for i in range(problem.n_ops):
            neighbor = best_solution.copy()
            neighbor[i] = (neighbor[i] + 1) % problem.num_machines
            obj = problem.evaluate_single(neighbor)
            if dominates(obj, best_obj):
                best_solution, best_obj, improved = neighbor, obj, True
                break
        if improved:
            continue

        for i in range(problem.n_ops):
            for j in range(i + 1, problem.n_ops):
                neighbor = best_solution.copy()
                neighbor[problem.n_ops + i], neighbor[problem.n_ops + j] = \
                    neighbor[problem.n_ops + j], neighbor[problem.n_ops + i]
                obj = problem.evaluate_single(neighbor)
                if dominates(obj, best_obj):
                    best_solution, best_obj, improved = neighbor, obj, True
                    break
            if improved:
                break

    return best_solution


class NSGA2WithLocalSearch(NSGA2):
    """NSGA-II with a greedy local-search refinement applied to every offspring."""

    def _next(self):
        off = super()._next()
        for i in range(len(off)):
            off[i].X = local_search(self.problem, off[i].X)
        return off


def _job_color_map(num_jobs):
    """
    Build a dict {job_id: RGBA color} with one visually distinct color per
    job. `tab20` (20 qualitative colors) is used directly when it covers all
    jobs; beyond that, colors are sampled evenly around the HSV wheel so
    every job still gets its own hue rather than colors repeating (as they
    would with matplotlib's 10-color default cycle via `job_id % 10`).
    """
    if num_jobs <= 20:
        palette = plt.get_cmap("tab20").colors[:num_jobs]
    else:
        cmap = plt.get_cmap("hsv")
        palette = [cmap(i / num_jobs) for i in range(num_jobs)]
    return {job_id: palette[job_id] for job_id in range(num_jobs)}


def _label_color_for(rgba):
    """Pick black or white label text for readable contrast against `rgba`."""
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.6 else "white"


def plot_gantt(solution, problem, out_path):
    """Save a Gantt chart of the schedule encoded by `solution`."""
    machine_assign, sequence = problem.decode(solution)
    machine_available = [0] * problem.num_machines
    job_ready = [0] * problem.num_jobs
    tasks = []
    job_colors = _job_color_map(problem.num_jobs)

    for op_idx in sequence:
        job_id, op_id, times = problem.operations[op_idx]
        available_machines = [m for m, t in enumerate(times) if t is not None]
        selected_machine = available_machines[machine_assign[op_idx] % len(available_machines)]
        proc_time = times[selected_machine]

        start_time = max(machine_available[selected_machine], job_ready[job_id])
        end_time = start_time + proc_time

        machine_available[selected_machine] = end_time
        job_ready[job_id] = end_time

        tasks.append({
            "Job": f"J{job_id + 1}-O{op_id + 1}",
            "Machine": selected_machine,
            "Start": start_time,
            "End": end_time,
            "Color": job_colors[job_id],
        })

    makespan = max(t["End"] for t in tasks)
    fig_width = max(10, makespan / 12)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    yticks, ylabels = [], []
    job_ids = sorted(set(t["Job"].split("-")[0] for t in tasks), key=lambda j: int(j[1:]))
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=job_colors[int(j[1:]) - 1]) for j in job_ids]

    # Only label a bar with its job/op id if it is wide enough for the text
    # to fit without overlapping its neighbours - avoids the illegible
    # smear of overlapping labels that shows up on dense schedules.
    min_label_width = makespan * 0.012

    for task in tasks:
        y = task["Machine"]
        width = task["End"] - task["Start"]
        ax.barh(y, width, left=task["Start"], height=0.6, align="center",
                color=task["Color"], edgecolor="black", linewidth=0.6)
        if width >= min_label_width:
            ax.text(task["Start"] + width / 2, y, task["Job"], va="center",
                    ha="center", fontsize=6.5, color=_label_color_for(task["Color"]),
                    clip_on=True)
        if y not in yticks:
            yticks.append(y)
            ylabels.append(f"M{y + 1}")

    ax.set_yticks(sorted(yticks))
    ax.set_yticklabels([f"M{y + 1}" for y in sorted(yticks)])
    ax.set_xlim(0, makespan * 1.01)
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Gantt Chart - Minimum-Makespan Solution")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(legend_handles, job_ids, title="Jobs", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_single_convergence(gens, values, out_path, *, title, ylabel, color,
                              marker, linestyle, value_fmt):
    """Save one standalone convergence curve (helper for `plot_convergence`)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_kwargs = dict(color=color, linewidth=2.0, linestyle=linestyle)
    if marker is not None:
        plot_kwargs.update(marker=marker, markevery=max(1, len(gens) // 25), markersize=5)
    ax.plot(gens, values, **plot_kwargs)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.annotate(value_fmt.format(values[-1]), xy=(gens[-1], values[-1]),
                xytext=(-8, 8), textcoords="offset points", ha="right",
                color=color, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(history, makespan_path, energy_path):
    """
    Save two separate, visually distinct convergence plots (best makespan
    and best energy cost per generation): different colours, line/marker
    styles, and file each, so the two objectives are never mistaken for one
    another when viewed independently (e.g. in a report or slide deck).
    """
    gens, best_makespan, best_energy = [], [], []
    for record in history:
        gens.append(record.n_gen)
        F = record.opt.get("F")
        best_makespan.append(np.min(F[:, 0]))
        best_energy.append(np.min(F[:, 1]))

    _plot_single_convergence(
        gens, best_makespan, makespan_path,
        title="NSGA-II Convergence - Best Makespan",
        ylabel="Best Makespan (minutes)",
        color="tab:blue", marker=None, linestyle="-", value_fmt="{:.0f} min",
    )
    _plot_single_convergence(
        gens, best_energy, energy_path,
        title="NSGA-II Convergence - Best Energy Cost",
        ylabel="Best Energy Cost (EUR)",
        color="tab:red", marker=None, linestyle="-", value_fmt="EUR {:.1f}",
    )


def plot_pareto_front(F, out_path):
    """Save a scatter plot of the final Pareto front."""
    order = np.argsort(F[:, 0])
    F_sorted = F[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(F_sorted[:, 0], F_sorted[:, 1], c="tab:blue", edgecolor="black")
    ax.plot(F_sorted[:, 0], F_sorted[:, 1], c="tab:blue", alpha=0.3, linewidth=1)
    ax.set_xlabel("Makespan (minutes)")
    ax.set_ylabel("Energy Cost (EUR)")
    ax.set_title("Pareto Front - Makespan vs. Energy Cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


BENCHMARK_CHOICES = [f"mk{i:02d}" for i in range(1, 11)]


def parse_args():
    parser = argparse.ArgumentParser(description="NSGA-II energy-aware FJSP mini project")
    parser.add_argument("--instance", choices=BENCHMARK_CHOICES, default="mk01",
                         help="Brandimarte (1993) FJSP benchmark instance to solve "
                              "(default: mk01, 10 jobs / 6 machines). Ignored if --random is set.")
    parser.add_argument("--random", action="store_true",
                         help="Generate a random FJSP instance instead of using a benchmark "
                              "(uses --n-jobs/--n-machines/--min-ops/--max-ops/--min-time/--max-time).")
    parser.add_argument("--n-jobs", type=int, default=15)
    parser.add_argument("--n-machines", type=int, default=4)
    parser.add_argument("--min-ops", type=int, default=5)
    parser.add_argument("--max-ops", type=int, default=10)
    parser.add_argument("--min-time", type=int, default=5)
    parser.add_argument("--max-time", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-gen", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42, help="Seed for both instance generation and NSGA-II")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    price_array = load_price_series(PRICE_CSV)

    if args.random:
        rng = random.Random(args.seed)
        operations, n_jobs, n_machines = generate_fjsp_instance(
            rng, n_jobs=args.n_jobs, n_machines=args.n_machines,
            min_ops=args.min_ops, max_ops=args.max_ops,
            min_time=args.min_time, max_time=args.max_time,
        )
        instance_label = f"random (seed={args.seed})"
    else:
        instance_path = BENCHMARK_DIR / f"{args.instance}.txt"
        operations, n_jobs, n_machines = parse_fjs_instance(instance_path)
        instance_label = f"Brandimarte {args.instance}"

    problem = EnergyAwareFJSP(operations, n_jobs, n_machines, price_array)

    algorithm = NSGA2WithLocalSearch(pop_size=args.pop_size)
    termination = get_termination("n_gen", args.n_gen)

    result = minimize(problem, algorithm, termination,
                       seed=args.seed, verbose=True, save_history=True)

    # NSGA-II's population can contain duplicate objective points (e.g. two
    # genotypes that decode to the same schedule); collapse those before
    # reporting the front so it reads as a clean, strictly non-dominated set.
    F_unique, unique_idx = np.unique(result.F, axis=0, return_index=True)
    X_unique = result.X[unique_idx]

    best_idx = int(np.argmin(F_unique[:, 0]))
    plot_gantt(X_unique[best_idx], problem, OUTPUT_DIR / "gantt_chart.png")
    plot_convergence(result.history,
                      OUTPUT_DIR / "convergence_makespan.png",
                      OUTPUT_DIR / "convergence_energy.png")
    plot_pareto_front(F_unique, OUTPUT_DIR / "pareto_front.png")

    pd.DataFrame(F_unique, columns=["makespan", "energy_cost_eur"]).sort_values(
        "makespan"
    ).to_csv(OUTPUT_DIR / "pareto_front.csv", index=False)

    summary_lines = [
        "NSGA-II Energy-Aware FJSP - Run Summary",
        "========================================",
        f"Instance: {instance_label}",
        f"Jobs: {n_jobs}, Machines: {n_machines}, Operations: {problem.n_ops}",
        f"Population size: {args.pop_size}, Generations: {args.n_gen}, Seed: {args.seed}",
        f"Pareto front size: {len(F_unique)}",
        f"Best makespan: {F_unique[:, 0].min():.2f} minutes",
        f"Best energy cost: {F_unique[:, 1].min():.4f} EUR",
        f"Min-makespan solution energy cost: {F_unique[best_idx, 1]:.4f} EUR",
    ]
    (OUTPUT_DIR / "run_summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    print(f"\nFigures and results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
