import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_results(path):
    try:
        data = np.loadtxt(path, delimiter=',')
        # if single row, ensure shape
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print('Failed to load results:', e)
        raise


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Path to results.txt')
    parser.add_argument('--instances', required=False, help='Path to instances .pkl (optional)')
    parser.add_argument('--outdir', required=False, help='Directory to save plots', default=None)
    parser.add_argument('--show-instance', type=int, default=None, help='Index of instance to map-plot')
    args = parser.parse_args()

    results = load_results(args.results)
    # results for batch: instance_id,cost
    ids = results[:,0].astype(int)
    costs = results[:,1].astype(float)

    outdir = args.outdir or os.path.dirname(args.results)
    os.makedirs(outdir, exist_ok=True)

    hist_path = os.path.join(outdir, 'cost_histogram.png')
    scatter_path = os.path.join(outdir, 'cost_scatter.png')

    plot_histogram(costs, hist_path)
    plot_scatter(ids, costs, scatter_path)
    print('Saved:', hist_path, scatter_path)

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
