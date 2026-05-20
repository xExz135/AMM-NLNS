import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_results(path):
    try:
        data = np.loadtxt(path, delimiter=',', dtype=str)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print('Failed to load results:', e)
        raise


def interpret_results(data):
    # Batch results: instance_id,cost
    # Single-search results: name,cost,runtime
    if data.shape[1] == 2:
        ids = data[:, 0].astype(int)
        costs = data[:, 1].astype(float)
        labels = None
        runtimes = None
    elif data.shape[1] == 3:
        labels = data[:, 0]
        costs = data[:, 1].astype(float)
        runtimes = data[:, 2].astype(float)
        ids = np.arange(len(costs))
    else:
        raise ValueError('Unsupported result format with %d columns' % data.shape[1])
    return ids, costs, labels, runtimes


def plot_histogram(costs, outpath):
    plt.figure(figsize=(6,4))
    plt.hist(costs, bins=50, color='C0', edgecolor='k')
    plt.xlabel('Cost')
    plt.ylabel('Count')
    plt.title('Distribution of NLNS costs')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_scatter(ids, costs, outpath):
    plt.figure(figsize=(8,4))
    plt.scatter(ids, costs, s=8, alpha=0.6)
    plt.xlabel('Instance ID')
    plt.ylabel('Cost')
    plt.title('Cost per instance')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_instance_map(instances_pkl, idx, outpath):
    with open(instances_pkl,'rb') as f:
        insts = pickle.load(f)
    if idx < 0 or idx >= len(insts):
        raise IndexError('instance index out of range')
    depot, locs, demands, cap = insts[idx]
    locs = np.array(locs)
    plt.figure(figsize=(6,6))
    plt.scatter(locs[:,0], locs[:,1], c='C1', label='customers')
    plt.scatter([depot[0]], [depot[1]], c='C0', marker='s', s=80, label='depot')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f'Instance {idx} map (cap={cap})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_bar_costs(ids, costs, labels, outpath):
    plt.figure(figsize=(10, 4))
    if labels is not None:
        plt.bar(ids, costs, color='C0')
        plt.xticks(ids, labels, rotation=90, fontsize=8)
    else:
        plt.bar(ids, costs, color='C0')
        plt.xlabel('Instance ID')
    plt.ylabel('Cost')
    plt.title('Costs')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Path to results.txt')
    parser.add_argument('--instances', required=False, help='Path to instances .pkl (optional)')
    parser.add_argument('--outdir', required=False, help='Directory to save plots', default=None)
    parser.add_argument('--show-instance', type=int, default=None, help='Index of instance to map-plot')
    args = parser.parse_args()

    results = load_results(args.results)
    ids, costs, labels, runtimes = interpret_results(results)

    outdir = args.outdir or os.path.dirname(args.results)
    os.makedirs(outdir, exist_ok=True)

    hist_path = os.path.join(outdir, 'cost_histogram.png')
    scatter_path = os.path.join(outdir, 'cost_scatter.png')
    bar_path = os.path.join(outdir, 'cost_bar.png')

    plot_histogram(costs, hist_path)
    plot_scatter(ids, costs, scatter_path)
    plot_bar_costs(ids, costs, labels, bar_path)
    print('Saved:', hist_path, scatter_path, bar_path)

    if args.show_instance is not None:
        if args.instances is None:
            print('No instances file provided to plot instance map')
        else:
            inst_path = args.instances
            map_path = os.path.join(outdir, f'instance_{args.show_instance}_map.png')
            plot_instance_map(inst_path, args.show_instance, map_path)
            print('Saved:', map_path)

if __name__ == '__main__':
    main()
