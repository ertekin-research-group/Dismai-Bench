import numpy as np
from ase.io import read, write
from ase.stress import full_3x3_to_voigt_6_stress
from m3gnet.models import Relaxer
import time
import csv
import os
import argparse
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
from distutils.util import strtobool
from math import ceil
import pandas as pd

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_batches', default=True, type=lambda x:bool(strtobool(x)), help='split relaxation into batches')
    parser.add_argument('--n_strucs_per_batch', type=int, default=100, help='number of structures per batch')
    parser.add_argument('--batch', type=int, default=1, help='batch number')
    parser.add_argument('--data_path', type=str, default='gen_clean.extxyz', help='path to generated structures (extxyz file)')
    parser.add_argument('--model_path', type=str, default='/path/to/data/potentials/m3gnet_int', help='path to m3gnet model')
    parser.add_argument('--f_max', type=float, default=0.05, help='maximum allowed force (magnitude of force vector) for relaxation convergence')
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
    
    for i in range(i_start, i_end+1):
        print("Structure number", i+1)
        struc = data[i].copy()
        relaxer = Relaxer(potential=args.model_path, optimizer=SciPyFminCG_no_crash, relax_cell=False, stress_weight=1)
        end = time.time()
        relax_results = relaxer.relax(struc, verbose=True, fmax=args.f_max, steps=args.max_steps)
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
            initial_energy_per_atom = float(trajectory.energies[0] / len(struc))
            
            # Get final structure and check fmax
            final_atoms_obj = trajectory.atoms
            final_energy_per_atom = float(trajectory.energies[-1] / len(struc))
            forces_last = trajectory.forces[-1]
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
                # Convert stress from 3x3 matrix to Voigt form so that the atoms object can be written out to extxyz
                calculator = final_atoms_obj.get_calculator()
                stress = calculator.results['stress']
                stress_voigt = full_3x3_to_voigt_6_stress(stress)
                calculator.results['stress'] = stress_voigt
                final_atoms_obj.set_calculator(calculator)
                
                with open(csv_fname, "a", newline='') as csvfile: 
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([batch, i+1, n_steps, np.round(relax_time,2), 
                                        initial_energy_per_atom, final_energy_per_atom, 'yes'])
                    
                write(filename=relax_atoms_fname, images=final_atoms_obj, format='extxyz', append=True)


if  __name__ == '__main__':
    main()

    
