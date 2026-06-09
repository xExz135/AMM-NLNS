import subprocess
import sys
from pathlib import Path
import re
import pickle

ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "trained_models" / "cvrp"
INSTANCE_PATH = ROOT / "instances" / "zomato_chunked.pkl" # 32, 64, 128, 256, 20, 50, 100

if not MODELS_ROOT.exists():
    raise SystemExit(f"Models folder not found: {MODELS_ROOT}")
if not INSTANCE_PATH.exists():
    raise SystemExit(f"Instance file not found: {INSTANCE_PATH}")

model_dirs = sorted([p for p in MODELS_ROOT.iterdir() if p.is_dir()])
if not model_dirs:
    raise SystemExit(f"No model directories found under {MODELS_ROOT}")

results = []

for model_dir in model_dirs:
    print(f"Running model {model_dir.name}...")
    # determine number of instances and choose a cpu count that divides it
    try:
        with INSTANCE_PATH.open('rb') as f:
            data = pickle.load(f)
        nb_instances = len(data)
    except Exception:
        nb_instances = None

    if nb_instances is None or nb_instances == 0:
        chosen_cpus = 1
        chosen_batch = 1
    else:
        DEFAULT_BATCH = 300
        max_cpus = min(10, nb_instances)

        chosen_cpus = None
        chosen_batch = None

        # Try cpu counts from max to 1 and find a batch size divisor of test_size_per_cpu
        for c in range(max_cpus, 0, -1):
            if nb_instances % c != 0:
                continue
            test_size = nb_instances // c
            # find largest batch <= DEFAULT_BATCH (and <= test_size) that divides test_size
            max_batch = min(DEFAULT_BATCH, test_size)
            batch = next((b for b in range(max_batch, 0, -1) if test_size % b == 0), None)
            if batch is not None:
                chosen_cpus = c
                chosen_batch = batch
                break

        # fallback: use single CPU and choose batch divisor of nb_instances
        if chosen_cpus is None:
            chosen_cpus = 1
            max_batch = min(DEFAULT_BATCH, nb_instances)
            chosen_batch = next((b for b in range(max_batch, 0, -1) if nb_instances % b == 0), 1)

    print(f"  Instances: {nb_instances}, using lns_nb_cpus={chosen_cpus}, lns_batch_size={chosen_batch}")

    cmd = [sys.executable, str(ROOT / "main.py"),
            "--mode", "eval_batch",
            "--device", "cpu",
            "--model_path", str(model_dir),
            "--instance_path", str(INSTANCE_PATH),
            "--instance_blueprint", model_dir.name,
            "--lns_nb_cpus", str(),
            "--lns_batch_size", str(chosen_batch)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(f"  FAILED: return code {proc.returncode}")
        print(stdout)
        continue

    # Try to locate the recently created run directory and plot results
    try:
        runs_dir = ROOT / 'runs'
        result_path = None
        if runs_dir.exists():
            # Find most recent run folder that contains search/results.txt
            run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            for rd in run_dirs:
                candidate = rd / 'search' / 'results.txt'
                if candidate.exists():
                    result_path = candidate
                    plots_outdir = rd / 'search' / 'plots'
                    break

        if result_path is not None:
            plots_outdir.mkdir(parents=True, exist_ok=True)
            print(f"  Generating plots for results: {result_path}")
            plot_cmd = [sys.executable, str(ROOT / 'plot_results.py'), '--results', str(result_path), '--instances', str(INSTANCE_PATH), '--outdir', str(plots_outdir)]
            pproc = subprocess.run(plot_cmd, capture_output=True, text=True)
            print(pproc.stdout)
            if pproc.returncode != 0:
                print('  Plotting failed:')
                print(pproc.stderr)
        else:
            print('  No results.txt found to plot for this run')
    except Exception as e:
        print('  Exception while plotting:', e)

    # Try multiple possible log messages for mean/test costs
    cost = None
    for pattern in [r"Mean Costs:\s*([0-9]+\.?[0-9]*)", r"Test set costs:\s*([0-9]+\.?[0-9]*)"]:
        m = re.search(pattern, stdout)
        if m:
            try:
                cost = float(m.group(1))
            except Exception:
                cost = None
            break

    if cost is not None:
        print(f"  OK: Mean/Test Costs = {cost}")
        results.append((model_dir.name, cost, stdout))
    else:
        print("  WARNING: could not parse mean/test cost from output")
        results.append((model_dir.name, float("inf"), stdout))

print("\nSummary:")
for name, cost, _ in results:
    print(f"{name}: {cost}")

if results:
    best = min(results, key=lambda x: x[1])
    print(f"\nBest model: {best[0]} with mean cost {best[1]}")
else:
    print("No successful model evaluations.")
