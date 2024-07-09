import numpy as np
from ase.io import read, write
import time
import csv
import os
import argparse
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
from quippy.potential import Potential
from ase import Atoms
from ase.constraints import ExpCellFilter
import io
import contextlib
import pickle
from distutils.util import strtobool
from math import ceil
import pandas as pd
import sys

import warnings
for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")

class Converged(Exception):
    pass

class OptimizerConvergenceError(Exception):
    pass

class SciPyOptimizer(Optimizer):
    """General interface for SciPy optimizers

    Only the call to the optimizer is still needed
    """

    def __init__(self, atoms, logfile='-', trajectory=None,
                 callback_always=False, alpha=70.0, master=None,
                 force_consistent=None):
        """Initialize object

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        callback_always: book
            Should the callback be run after each force call (also in the
            linesearch)

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """
        restart = None
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent=force_consistent)
        self.force_calls = 0
        self.callback_always = callback_always
        self.H0 = alpha

    def x0(self):
        """Return x0 in a way SciPy can use

        This class is mostly usable for subclasses wanting to redefine the
        parameters (and the objective function)"""
        return self.atoms.get_positions().reshape(-1)

    def f(self, x):
        """Objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        # Scale the problem as SciPy uses I as initial Hessian.
        return (self.atoms.get_potential_energy(
                force_consistent=self.force_consistent) / self.H0)

    def fprime(self, x):
        """Gradient of the objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        self.force_calls += 1

        if self.callback_always:
            self.callback(x)

        # Remember that forces are minus the gradient!
        # Scale the problem as SciPy uses I as initial Hessian.
        return - self.atoms.get_forces().reshape(-1) / self.H0

    def callback(self, x):
        """Callback function to be run after each iteration by SciPy

        This should also be called once before optimization starts, as SciPy
        optimizers only calls it after each iteration, while ase optimizers
        call something similar before as well.

        :meth:`callback`() can raise a :exc:`Converged` exception to signal the
        optimisation is complete. This will be silently ignored by
        :meth:`run`().
        """
        f = self.atoms.get_forces()
        self.log(f)
        self.call_observers()
        if self.converged(f):
            raise Converged
        self.nsteps += 1

    def run(self, fmax=0.05, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        try:
            # As SciPy does not log the zeroth iteration, we do that manually
            self.callback(None)
            # Scale the problem as SciPy uses I as initial Hessian.
            self.call_fmin(fmax / self.H0, steps)
        except Converged:
            pass

    def dump(self, data):
        pass

    def load(self):
        pass

    def call_fmin(self, fmax, steps):
        raise NotImplementedError

class SciPyFminCG_no_crash(SciPyOptimizer):
    """
    Non-linear (Polak-Ribiere) conjugate gradient algorithm
    
    OptimizerConvergenceError Exception commented out to continue relaxations 
    instead of ending the run
    """

    def call_fmin(self, fmax, steps):
        output = opt.fmin_cg(self.f,
                             self.x0(),
                             fprime=self.fprime,
                             # args=(),
                             gtol=fmax * 0.1,  # Should never be reached
                             norm=np.inf,
                             # epsilon=
                             maxiter=steps,
                             full_output=1,
                             disp=0,
                             # retall=0,
                             callback=self.callback)
        warnflag = output[-1]
        if warnflag == 2:
            print("OptimizerConvergenceError")
            #raise OptimizerConvergenceError(
            #    'Warning: Desired error not necessarily achieved '
            #    'due to precision loss')

class Relaxer_GAP:
    """
    Relaxer_GAP is a class for structural relaxation
    """

    def __init__(
        self,
        potential: Potential = None,
        optimizer: Optimizer = None,
        relax_cell: bool = None
    ):
        """

        Args:
            potential: quippy Potential object (ase calculator object)
            optimizer: ase Optimizer object.
            relax_cell (bool): whether to relax the lattice cell
        """
        if potential is None:
            raise ValueError("Potential cannot be None")

        if optimizer is None:
            raise ValueError("Optimizer cannot be None")

        self.opt_class = optimizer
        self.calculator = potential
        self.relax_cell = relax_cell

    def relax(
        self,
        atoms: Atoms,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str = None,
        interval=1,
        verbose=False,
        use_soft_fmax=False,
        **kwargs,
    ):
        """

        Args:
            atoms (Atoms): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
                Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            use_soft_fmax: saves the energies, forces, and stresses in the trajectory
                for use with the soft_fmax convergence citeria
            **kwargs:
        Returns:
            final structure: ase Atoms object of final structure
            trajectory: TrajectoryObserver object
        """
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms, use_soft_fmax)
            if self.relax_cell:
                atoms = ExpCellFilter(atoms)
            optimizer = self.opt_class(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file is not None:
            obs.save(traj_file)
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms

        return {
            "final_structure": atoms,
            "trajectory": obs,
        }

class TrajectoryObserver:
    """
    Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures
    """

    def __init__(self, atoms: Atoms, use_soft_fmax):
        """
        Args:
            atoms (Atoms): the structure to observe
        """
        self.use_soft_fmax = use_soft_fmax
        self.atoms = atoms
        if self.use_soft_fmax:
            self.energies: list[float] = []
            self.forces: list[np.ndarray] = []
            self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """
        The logic for saving the properties of an Atoms during the relaxation
        Returns:
        """
        if self.use_soft_fmax:
            self.energies.append(self.compute_energy())
            self.forces.append(self.atoms.get_forces())
            self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def compute_energy(self) -> float:
        """
        calculate the energy, here we just use the potential energy
        Returns:
        """
        energy = self.atoms.get_potential_energy()
        return energy

    def save(self, filename: str):
        """
        Save the trajectory to file
        Args:
            filename (str): filename to save the trajectory
        Returns:
        """
        traj_dict = {"atom_positions": self.atom_positions,
                    "cell": self.cells,
                    "atomic_number": self.atoms.get_atomic_numbers()}
        if self.use_soft_fmax:
            traj_dict.update({"energy": self.energies,
                              "forces": self.forces,
                              "stresses": self.stresses,}
                             )
        with open(filename, "wb") as f:
            pickle.dump(traj_dict, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_batches', default=True, type=lambda x:bool(strtobool(x)), help='split relaxation into batches')
    parser.add_argument('--n_strucs_per_batch', type=int, default=100, help='number of structures per batch')
    parser.add_argument('--batch', type=int, default=1, help='batch number')
    parser.add_argument('--data_path', type=str, default='gen_clean.extxyz', help='path to data (extxyz file)')
    parser.add_argument('--model_path', type=str, default='/path/to/data/potentials/GAP_PRX_Si/gp_iter6_sparse9k.xml', help='path to GAP model')
    parser.add_argument('--f_max', type=float, default=0.05, help='maximum allowed force (length of force vector) for relaxation convergence')
    parser.add_argument('--max_steps', type=int, default=1000, help='maximum number of relaxation steps allowed')
    parser.add_argument('--use_soft_fmax', default=False, type=lambda x:bool(strtobool(x)), help='use soft_fmax convergence criteria')
    parser.add_argument('--soft_fmax', type=float, default=0.1, help='if the relaxation does not converge within max_steps, ' + 
                                                                     'the step with the minimum fmax will be saved if below this threshold')
    
    args = parser.parse_args()
    batch = args.batch
    
    if args.split_batches:
        data = read(filename='../'+args.data_path, index=':', format='extxyz')
    else:
        data = read(filename=args.data_path, index=':', format='extxyz')
    relax_atoms_fname = args.data_path.split('.')[0] + '_relaxed.' + args.data_path.split('.')[1]  # For saving relaxed structures
    csv_fname = 'relax_results.csv'   # For saving relaxation results
    
    # Get starting and ending indexes of structures to relax
    if args.split_batches:
        n_strucs_per_batch = args.n_strucs_per_batch
        i_start = (batch - 1)*n_strucs_per_batch   # Starting index
        if batch == ceil(len(data) / n_strucs_per_batch):   # last batch may have less than n_strucs_per_batch
            i_end = len(data) - 1   # Ending index
        else:
            i_end = batch*n_strucs_per_batch - 1   # Ending index
    else:
        i_start = 0
        i_end = len(data) - 1 
    
    # Restart relaxation if previous run crashed (skips the crashed structure)
    if os.path.exists(csv_fname) or os.path.exists("discarded.csv"):
        print('Previous run detected')
        last_index = []
        if os.path.exists(csv_fname):
            df_results = pd.read_csv(csv_fname)
            last_index.append(df_results['structure_number'].iloc[-1] - 1)
        if os.path.exists("discarded.csv"):
            df_discarded = pd.read_csv("discarded.csv")
            last_index.append(df_discarded['structure_number'].iloc[-1] - 1)
        last_index = max(last_index)
        
        print('Logging last crashed structure')
        if not os.path.exists("discarded.csv"):
            with open("discarded.csv", "a", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['batch', 'structure_number', 'n_steps', 'time', 
                                    'E_initial (eV/atom)', 'E_final (eV/atom)', 'fmax_final'])
        with open("discarded.csv", "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Details omitted from entry for simplicity, can be pulled from job output file if desired
            csvwriter.writerow([batch, last_index + 2, '', '', 
                                '', '', ''])
        
        print('Restarting relaxations')
        i_start = last_index + 2   # Skip crashed structure
    else:
        # Write csv header
        with open(csv_fname, "a", newline='') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['batch', 'structure_number', 'n_steps', 'time (s)', 
                                'E_initial (eV/atom)', 'E_final (eV/atom)', 'convergence'])
    
    # Initialize GAP potential
    Si_potential = Potential(init_args='Potential xml_label="GAP_2017_6_17_60_4_3_56_165"',
                             param_filename=args.model_path)
    
    for i in range(i_start, i_end+1):
        print("Structure number", i+1)
        struc = data[i].copy()
        relaxer = Relaxer_GAP(potential=Si_potential, optimizer=SciPyFminCG_no_crash, relax_cell=False)
        end = time.time()
        relax_results = relaxer.relax(struc, verbose=True, fmax=args.f_max, steps=args.max_steps, use_soft_fmax=args.use_soft_fmax)
        relax_time = time.time() - end
        
        trajectory = relax_results['trajectory']
        n_steps = len(trajectory.atom_positions) - 2  # Removes first entry (original structure) and last entry (final structure energy calculated twice)
        
        soft_step_saved = False   # Indicates if the step corresponding to the soft_fmax convergence criteria is saved
        if args.use_soft_fmax:
            if n_steps == args.max_steps:
                ## Maximum number of relaxation steps reached. Check the minimum fmax throughout the relaxation.
                forces_all = np.array(trajectory.forces)[0:-1]  # Remove last entry (final structure was calculated twice)
                fmax_all = []
                for j in range(len(forces_all)):
                    forces = forces_all[j]
                    forces_norm = np.linalg.norm(forces, axis=1)
                    fmax = np.max(forces_norm)
                    fmax_all.append(fmax)
                min_fmax = np.min(np.array(fmax_all))
                index_min_fmax = np.argmin(fmax_all)
                
                if args.f_max+0.001 < min_fmax <= args.soft_fmax+0.001:
                    ## min_fmax did not converge to target fmax, but converged to soft_fmax. Save the step with min_fmax.
                    # Retrieve atoms object of last step and replace with the step with minimum f_max
                    atoms_obj = trajectory.atoms
                    atoms_obj.set_positions(trajectory.atom_positions[index_min_fmax])
                    calculator = atoms_obj.get_calculator()
                    calculator.results['energy'] = trajectory.energies[index_min_fmax]
                    calculator.results['free_energy'] = trajectory.energies[index_min_fmax]
                    calculator.results['forces'] = trajectory.forces[index_min_fmax]
                    calculator.results['stress'] = trajectory.stresses[index_min_fmax]
                    atoms_obj.set_calculator(calculator)
                    
                    # Get energy per atom
                    initial_energy_per_atom = float(trajectory.energies[0] / len(struc))
                    step_energy_per_atom = float(trajectory.energies[index_min_fmax] / len(struc))
                    
                    with open(csv_fname, "a", newline='') as csvfile: 
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([batch, i+1, n_steps, np.round(relax_time,2), 
                                            initial_energy_per_atom, step_energy_per_atom, 'soft'])
                    
                    write(filename=relax_atoms_fname, images=atoms_obj, format='extxyz', append=True)
                    soft_step_saved = True
           
        if soft_step_saved is False:
            # Get initial energy per atom
            initial_atoms_obj = data[i].copy()
            initial_atoms_obj.set_calculator(Si_potential)
            initial_energy_per_atom = initial_atoms_obj.get_potential_energy() / len(struc)
            
            # Get final structure and check fmax
            final_atoms_obj = trajectory.atoms
            final_energy_per_atom = final_atoms_obj.get_potential_energy() / len(struc)
            forces_last = final_atoms_obj.get_forces()
            forces_last_norm = np.linalg.norm(forces_last, axis=1)
            fmax_last = np.max(forces_last_norm)
            
            if fmax_last > args.f_max+0.001:
                ## Did not converge, discard structure
                if not os.path.exists("discarded.csv"):
                    with open("discarded.csv", "a", newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(['batch', 'structure_number', 'n_steps', 'time', 
                                            'E_initial (eV/atom)', 'E_final (eV/atom)', 'fmax_final'])
                with open("discarded.csv", "a", newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([batch, i+1, n_steps, np.round(relax_time,2), 
                                        initial_energy_per_atom, final_energy_per_atom, fmax_last])            
            else:
                ## Converged to target fmax, save last structure
                # Add stress to calculator results
                calculator = final_atoms_obj.get_calculator()
                stress = final_atoms_obj.get_stress()
                calculator.results['stress'] = stress
                final_atoms_obj.set_calculator(calculator)
                
                with open(csv_fname, "a", newline='') as csvfile: 
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([batch, i+1, n_steps, np.round(relax_time,2), 
                                        initial_energy_per_atom, final_energy_per_atom, 'yes'])
                    
                write(filename=relax_atoms_fname, images=final_atoms_obj, format='extxyz', append=True)


if  __name__ == '__main__':
    main()

    
