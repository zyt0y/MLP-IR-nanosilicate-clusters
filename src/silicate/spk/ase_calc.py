
import numpy as np
import torch
from schnetpack.interfaces.ase_interface import SpkCalculatorError

from ase.calculators.calculator import Calculator, all_changes
from schnetpack import Properties
from schnetpack.md.utils import MDUnits
from schnetpack.data.atoms import AtomsConverter
from schnetpack.environment import SimpleEnvironmentProvider

from schnetpack.data.atoms import _convert_atoms, torchify_dict, get_center_of_mass

class AtomsConverter:
    """
    Convert ASE atoms object to an input suitable for the SchNetPack
    ML models.

    Args:
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        device (str): Device for computation (default='cpu')
    """

    def __init__(
        self,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        device=torch.device("cpu"),
    ):
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples

        # Get device
        self.device = device

    def __call__(self, atoms):
        """
        Args:
            atoms (ase.Atoms): Atoms object of molecule

        Returns:
            dict of torch.Tensor: Properties including neighbor lists and masks
                reformated into SchNetPack input format.
        """
        inputs = _convert_atoms(atoms, self.environment_provider, self.collect_triples, centering_function=get_center_of_mass)
        inputs['charges'] = np.zeros(1, np.float32)
        inputs = torchify_dict(inputs)

        # Calculate masks
        inputs[Properties.atom_mask] = torch.ones_like(inputs[Properties.Z]).float()
        mask = inputs[Properties.neighbors] >= 0
        inputs[Properties.neighbor_mask] = mask.float()
        inputs[Properties.neighbors] = (
            inputs[Properties.neighbors] * inputs[Properties.neighbor_mask].long()
        )

        if self.collect_triples:
            mask_triples = torch.ones_like(inputs[Properties.neighbor_pairs_j])
            mask_triples[inputs[Properties.neighbor_pairs_j] < 0] = 0
            mask_triples[inputs[Properties.neighbor_pairs_k] < 0] = 0
            inputs[Properties.neighbor_pairs_mask] = mask_triples.float()

        # Add batch dimension and move to CPU/GPU
        for key, value in inputs.items():
            inputs[key] = value.unsqueeze(0).to(self.device)

        return inputs

class SpkDipoleCalculator(Calculator):
    """
    ASE calculator for schnetpack machine learning models.

    Args:
        ml_model (schnetpack.AtomisticModel): Trained model for
            calculations
        device (str): select to run calculations on 'cuda' or 'cpu'
        collect_triples (bool): Set to True if angular features are needed,
            for example, while using 'wascf' models
        environment_provider (callable): Provides neighbor lists
        pair_provider (callable): Provides list of neighbor pairs. Only
            required if angular descriptors are used. Default is none.
        **kwargs: Additional arguments for basic ase calculator class
    """

    energy = Properties.energy
    forces = Properties.forces
    stress = Properties.stress
    implemented_properties = [energy, forces, stress, "dipole", "charges"]

    def __init__(
        self,
        model,
        device="cpu",
        collect_triples=False,
        environment_provider=SimpleEnvironmentProvider(),
        energy=None,
        forces=None,
        stress=None,
        dipole=None,
        charges=None,
        energy_units="eV",
        forces_units="eV/Angstrom",
        stress_units="eV/Angstrom/Angstrom/Angstrom",
        pos_units=1,
        dipole_units=1,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.model = model
        self.model.to(device)

        self.atoms_converter = AtomsConverter(
            environment_provider=environment_provider,
            collect_triples=collect_triples,
            device=device,
        )

        self.model_energy = energy
        self.model_forces = forces
        self.model_stress = stress
        self.model_dipole = dipole
        self.model_charges = charges

        # Convert to ASE internal units (energy=eV, length=A)
        self.energy_units = MDUnits.unit2unit(energy_units, "eV")
        self.forces_units = MDUnits.unit2unit(forces_units, "eV/Angstrom")
        self.stress_units = MDUnits.unit2unit(stress_units, "eV/A/A/A")
        self.pos_units = pos_units
        self.dipole_units = dipole_units

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)
            # Convert to schnetpack input format
            model_inputs = self.atoms_converter(atoms)
            # Convert to Bohr
            model_inputs["_positions"] *= self.pos_units

            # Call model
            model_results = self.model(model_inputs)

            results = {}
            # Convert outputs to calculator format
            if self.model_energy is not None:
                if self.model_energy not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model "
                        "properties!".format(self.model_energy)
                    )
                energy = model_results[self.model_energy].cpu().data.numpy()
                results[self.energy] = (
                    energy.item() * self.energy_units
                )  # ase calculator should return scalar energy

            if self.model_forces is not None:
                if self.model_forces not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model"
                        "properties!".format(self.model_forces)
                    )
                forces = model_results[self.model_forces].cpu().data.numpy()
                results[self.forces] = (
                    forces.reshape((len(atoms), 3)) * self.forces_units
                )

            if self.model_stress is not None:
                if atoms.cell.volume <= 0.0:
                    raise SpkCalculatorError(
                        "Cell with 0 volume encountered for stress computation"
                    )

                if self.model_stress not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model"
                        "properties! If desired, stress tensor computation can be "
                        "activated via schnetpack.utils.activate_stress_computation "
                        "at ones own risk.".format(self.model_stress)
                    )
                stress = model_results[self.model_stress].cpu().data.numpy()
                results[self.stress] = stress.reshape((3, 3)) * self.stress_units

            if self.model_dipole is not None:
                if self.model_dipole not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model"
                        "properties!".format(self.model_dipole)
                    )
                dipole = model_results[self.model_dipole].cpu().data.numpy().flatten()
                results["dipole"] = dipole * self.dipole_units

            if self.model_charges is not None:
                if self.model_charges not in model_results.keys():
                    print(model_results.keys())
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model"
                        "properties!".format(self.model_charges)
                    )
                charges = model_results[self.model_charges].cpu().data.numpy()
                results["charges"] = charges.reshape(len(atoms))
            self.results = results
