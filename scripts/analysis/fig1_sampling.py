
import os
import numpy as np

from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel

from ase import Atoms
from ase.io import read, write

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA


def read_orca_hess(fname):
    ''' return ASE-trajectories and freq
    '''
    with open(fname) as fileobj:
        lines = fileobj.readlines()
        natoms = int(lines[0])
        nimages = len(lines) // (natoms + 2)
        trajs = []
        for i in range(nimages):
            symbols = []
            positions = []
            velocities = []
            n = i * (natoms + 2) + 2

            if i == 1:
                freq = float(lines[n-1].split()[-1])
            for line in lines[n:n + natoms]:
                symbol, x, y, z, vx, vy, vz = line.split()#[:4]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
                velocities.append([float(vx), float(vy), float(vz)])
            atoms = Atoms(symbols=symbols, positions=positions, velocities=velocities)
            trajs.append(atoms)
    return trajs, freq

def farthest_point_sampling(similarities, n_points=None):
    N = similarities.shape[0]
    farthest_indices = np.zeros(N, int)
    ds = similarities[0, :]
    for i in range(1, N):
        idx = np.argmin(ds)
        farthest_indices[i] = idx
        ds = np.maximum(ds, similarities[idx, :])
    #print(distances, farthest_indices)
    if n_points is None:
        return farthest_indices
    else:
        return farthest_indices[:n_points]

def one_sampling(base_path, saving_path):
    # We will compare two similar molecules
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    opt_atoms = read(f'{base_path}/orca.xyz')
    n_atoms = len(opt_atoms)
    trajs = [opt_atoms]
    indices = range(6, 3*n_atoms)
    n_points = 1 + 3*n_atoms
    plot_indices = [0]
    for i, idx in enumerate(indices):
        #trajs += read(f'{base_path}/orca.hess.v{idx:03}.xyz', '1:10')
        trajs += read_orca_hess(f'{base_path}/orca.hess.v{idx:03}.xyz')[0][1:10]
        plot_indices += [i+1]*9
    print(f'number of structures: {len(trajs)}')
    print(f'number of atoms: {n_atoms}')
    np.save(f'{saving_path}/mode_plot_indices.npy', plot_indices)

    # First we will have to create the features for atomic environments. Lets
    # use SOAP.
    local_desc = SOAP(species=["Mg", "Si", "O"], r_cut=5.0, n_max=5, l_max=6, sigma=0.2)
    global_desc = SOAP(species=["Mg", "Si", "O"], r_cut=5.0, n_max=5, l_max=6, sigma=0.2, average='inner')

    print(f'calculating {local_desc.get_number_of_features()} local features ')
    all_local_features = local_desc.create(trajs)
    print(f'calculating {global_desc.get_number_of_features()} global features ')
    all_global_features = global_desc.create(trajs)

    # Before passing the features we normalize them. Depending on the metric, the
    # REMatch kernel can become numerically unstable if some kind of normalization
    # is not done.
    print('normalizing local features')
    all_normalized_features = [normalize(features) for features in all_local_features]
    

    # Calculates the similarity with the REMatch kernel and a linear metric. The
    # result will be a full similarity matrix.
    print('REMatchKernel')
    re = REMatchKernel(metric="linear", alpha=1e-2, threshold=1e-7, normalize_kernel=True)
    re_kernel = re.create(all_normalized_features)
    #distances = StandardScaler().fit_transform(re_kernel)
    all_indices = farthest_point_sampling(re_kernel)
    selected_indices = all_indices[:n_points]
    selected_trajs = [trajs[i] for i in selected_indices]
    print(len(all_indices), len(selected_indices))
    np.save(f'{saving_path}/farthest_all_indices.npy', all_indices)
    np.save(f'{saving_path}/farthest_selected_indices.npy', selected_indices)

    #write(f'{saving_path}/farthest_selected_non_opt.traj', selected_trajs)
    write(f'{saving_path}/farthest_selected_non_opt_with_velocities.traj', selected_trajs)
    
    # PCA analysis
    print('PCA analysis')
    model = PCA(n_components=2, random_state=1000).fit(all_global_features)
    reduced_global_features = model.transform(all_global_features)
    np.save(f'{saving_path}/global_features_PCA.npy', reduced_global_features)

#one_sampling('../../data/orca/P2', 'test')
one_sampling('../../data/orca/P2', '../../data/PCA/P2')
