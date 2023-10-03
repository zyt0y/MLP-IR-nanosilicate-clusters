
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from silicate.plot.ase_plot import plot_atoms


def ax_plot_hessian_dipole(ax, atoms, view='top'):
    ax_x = 8
    ax_y = 6
    ax_z = 4
    com = atoms.get_center_of_mass()
    
    color_scheme = np.zeros((len(atoms), 4), dtype=object)
    color_scheme[:, 0] = 1
    color_scheme[:, 2] = 1 # alpha values for atoms color
    color_scheme[[3, 5], 3] = 1 # dashed for atoms Circle boundaries

    bbox_top = (com[0]-ax_x*0.5, com[1]-ax_y/2, com[0]+ax_x*0.5, com[1]+ax_y/2)
    bbox_side = (com[0]-ax_x*0.5, com[2]-ax_z/2, com[0]+ax_x*0.5, com[2]+ax_z/2)
    if view == 'top':
        bbox = bbox_top
        rotation = ''
    else:
        bbox = bbox_side
        rotation = '-90x'
    ax, mpl_object = plot_atoms(atoms, ax, return_object=True, color_scheme=color_scheme, bbox=bbox, show_vector=1, show_bonds=1, radii=0.7, plot_bond_radii=0.9, rotation=rotation)
    
    distance = atoms.get_distance(5, 3)
    ax.text(0.47, 0.9, f'{distance:.2f}Å', transform=ax.transAxes, fontsize='x-small')
    #ax.plot(mpl_object.positions[[3, 5], 0],mpl_object.positions[[3, 5], 1], color='gray', ls='--')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def get_plot_data(data_path):
    plot_indices = np.load(f'{data_path}/mode_plot_indices.npy')
    selected_indices = np.load(f'{data_path}/farthest_selected_indices.npy')
    reduced_global_features = np.load(f'{data_path}/global_features_PCA.npy')

    trajs = read(f'{data_path}/farthest_selected_non_opt_with_velocities.traj', ':')
    return trajs, plot_indices, selected_indices, reduced_global_features

def ax_plot_sampling(ax, plot_indices, selected_indices, labeled_indices, reduced_global_features):
    ax.scatter(reduced_global_features[:, 0], reduced_global_features[:, 1], s=5, c=plot_indices, cmap='tab20b')
    ax.scatter(reduced_global_features[selected_indices, 0], reduced_global_features[selected_indices, 1], s=20, c='red', marker='x')
    texts = ['b', 'c', 'd', 'e', 'f', 'g']
    for i, index in enumerate(labeled_indices):
        ax.text(reduced_global_features[index, 0], reduced_global_features[index, 1]+0.0004, texts[i])
    ax.set_aspect('equal')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_xticks([])
    ax.set_yticks([])

    pos = ax.get_position()
    pos.y0 -= 0.15
    ax.set_position(pos)

def axes_plot_trajs(axes, trajs):
    texts = ['b', 'c', 'd', 'e', 'f', 'g']
    axes[0].text(0.01, 3.75, '(a)', transform=axes[0].transAxes)
    for i in range(len(axes)):
        ax_plot_hessian_dipole(axes[i], trajs[i])
        axes[i].text(0.01, 0.85, f'({texts[i]})', transform=axes[i].transAxes)

def plot_all_in_one(saving_path, trajs, plot_indices, selected_indices, labeled_indices, reduced_global_features):
    print('Plot all in one')
    plt.switch_backend('Agg')
    fig = plt.figure(constrained_layout=False, figsize=(6, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[4, 1, 1], hspace=0)
    ax_sampling = fig.add_subplot(gs[0, :])
    axes_trajs = []
    for i in range(2):
        for j in range(3):
            axes_trajs.append(fig.add_subplot(gs[i+1, j]))
    ax_plot_sampling(ax_sampling,  plot_indices, selected_indices, labeled_indices, reduced_global_features)
    axes_plot_trajs(axes_trajs, trajs)

    plt.savefig(f'{saving_path}/fig1.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{saving_path}/fig1.pdf', dpi=200, bbox_inches='tight', transparent=True)

data_path = '../../data/PCA/P2'
saving_path = '../../figures'
labeled_indices = [212, 113, 194, 161, 160, 158]
trajs, plot_indices, selected_indices, reduced_global_features = get_plot_data(data_path)
plottted_trajs = []
for i in range(len(labeled_indices)):
    for j in range(len(selected_indices)):
        if labeled_indices[i] == selected_indices[j]:
            plottted_trajs.append(trajs[j])
plot_all_in_one(saving_path, plottted_trajs, plot_indices, selected_indices, labeled_indices, reduced_global_features)
