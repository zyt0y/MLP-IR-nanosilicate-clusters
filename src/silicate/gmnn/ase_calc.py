import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from gmnn.utils import *
from gmnn.layers import *
from gmnn.neighborlist import *
from ase import units
from gmnn.calculators import NNCalculator


class GMNNCalculator(NNCalculator):
    def __init__(self, model_path: str, config: str = "pes_training.txt", use_all_features: bool = True,
                 device_number: int = 0, wrap_positions: bool = False, skin: float = 0.0, compute_stress: bool = False,
                 pbc: int = 0,
                 set_units: str = 'kcal/mol -> eV', dtype=tf.float32):
        super().__init__(model_path=model_path, config=config, use_all_features=use_all_features,
                         device_number=device_number)
        self.skin = skin
        self.wrap_positions = wrap_positions
        self.nl = NeighborListTF(cutoff=self.args.cutoff + self.skin, wrap_positions=self.wrap_positions)
        self.compute_stress = compute_stress
        self.pbc = pbc
        self._set_units(set_units)
        self.config = config
        self.dtype = dtype

    def _scale_shift_model(self, layers):
        return ScaleShiftModel(Sequential(layers), atomic_energy_std=1.0, atomic_energy_regression=np.zeros(119))

    def _build_neighbors(self, atoms: Atoms):
        self.idx_i, self.idx_j, self.offsets = self.nl.neighbor_list(atoms)

    def _set_units(self, set_units: str):
        if set_units == 'kcal/mol to eV':
            self.rescale_units = units.kcal / units.mol
        elif set_units == 'Hartree to eV':
            self.rescale_units = units.Hartree
        elif set_units == 'eV to kcal/mol':
            self.rescale_units = units.mol / units.kcal
        elif set_units == 'eV to Hartree':
            self.rescale_units = 1.0 / units.Hartree
        elif set_units == 'kcal/mol to Hartree':
            self.rescale_units = units.kcal / units.mol / units.Hartree
        elif set_units == 'eV to eV':
            self.rescale_units = units.eV
        else:
            raise NotImplementedError()
        
    @tf.function(input_signature=[
        tf.TensorSpec([None, ], dtype=tf.int32),                # data['Z']
        tf.TensorSpec([None, None], dtype=tf.float32),          # data['R']
        tf.TensorSpec([None, None, None], dtype=tf.float32),    # data['C']
        tf.TensorSpec([None, ], dtype=tf.int32),                # data['idx_i']
        tf.TensorSpec([None, ], dtype=tf.int32),                # data['idx_j']
        tf.TensorSpec([None, None], dtype=tf.float32),          # data['offsets']
        tf.TensorSpec([None, ], dtype=tf.int32)                 # data['batch_seg']
    ])   
    def _inference(self, Z, R, C, idx_i, idx_j, offsets, batch_seg):
        E_bp_pred = []
        F_at_pred = []
        W_at_pred = []
        for model in self.models:
            with tf.GradientTape(watch_accessed_variables=False) as force_tape:
                force_tape.watch(R)
                R_ij = get_distance_vectors(R, C, idx_i, idx_j, offsets, batch_seg, pbc=self.pbc)
                if self.pbc and self.compute_stress:
                    with tf.GradientTape(watch_accessed_variables=False) as stress_tape:
                        stress_tape.watch(R_ij)
                        x = (R_ij, Z, idx_i, idx_j)
                        E_at_pred = tf.cast(model(x, training=False), dtype=self.dtype)
                        E_bp_pred.append(tf.reduce_sum(E_at_pred))
                        E_total = tf.reduce_sum(E_at_pred)
                    F_ij_pred = tf.convert_to_tensor(stress_tape.gradient(E_total, R_ij))
                    W_at_pred.append(- 1.0 * tf.reduce_sum(F_ij_pred[:, None, :] * R_ij[:, :, None], 0)
                                     / tf.abs(tf.linalg.det(C)))
                else:
                    x = (R_ij, Z, idx_i, idx_j)
                    E_at_pred = tf.cast(model(x, training=False), dtype=self.dtype)
                    E_bp_pred.append(tf.reduce_sum(E_at_pred))
                    E_total = tf.reduce_sum(E_at_pred)
            F_at_pred.append(-tf.convert_to_tensor(force_tape.gradient(E_total, R)))

        # mean force used to propagate the structure
        F_at_ens = sum(F_at_pred) / len(self.models)
        E_bp_ens = sum(E_bp_pred) / len(self.models)
        if self.pbc:
            W_at_ens = sum(W_at_pred) / len(self.models)

        # force variance to check the accuracy of the potential
        F_at_var = tf.math.reduce_std(F_at_pred, axis=0) ** 2
        E_bp_var = tf.math.reduce_std(E_bp_pred, axis=0) ** 2
        if self.pbc:
            W_at_var = tf.math.reduce_std(W_at_pred, axis=0) ** 2
        
        if self.pbc and self.compute_stress:
            results = {"energy_mean": E_bp_ens,
                       "energy_var": E_bp_var,
                       "force_mean": F_at_ens,
                       "force_var": F_at_var,
                       "stress_mean": W_at_ens,
                       "stress_var": W_at_var}
        else:
            results = {"energy_mean": E_bp_ens,
                       "energy_var": E_bp_var,
                       "force_mean": F_at_ens,
                       "force_var": F_at_var}

        return results

    def calculate_all_properties(self, atoms):
        self._build_neighbors(atoms)

        batch_seg = np.zeros(len(atoms), dtype=np.int64)
        # move everything to TensorFlow
        Z = tf.convert_to_tensor(atoms.get_atomic_numbers(), dtype=tf.int32)
        R = tf.convert_to_tensor(atoms.get_positions(), dtype=tf.float32)
        C = tf.convert_to_tensor(atoms.get_cell()[None, :, :], dtype=tf.float32)
        idx_i = tf.convert_to_tensor(self.idx_i, dtype=tf.int32)
        idx_j = tf.convert_to_tensor(self.idx_j, dtype=tf.int32)
        offsets = tf.convert_to_tensor(self.offsets, dtype=tf.float32)
        batch_seg = tf.convert_to_tensor(batch_seg, dtype=tf.int32)

        self.results = self._inference(Z, R, C, idx_i, idx_j, offsets, batch_seg)

    def get_potential_energy(self, atoms: Atoms, force_consistent: bool = False) -> float:
        """
        :param atoms: Atomistic system.
        :return: Potential energy of the simulated system.
        """
        self.calculate_all_properties(atoms)
        return self.results["energy_mean"].numpy() * self.rescale_units

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """
        :param atoms: Atomistic system.
        :return: Atomic forces of the simulated system.
        """
        self.calculate_all_properties(atoms)
        return self.results["force_mean"].numpy() * self.rescale_units