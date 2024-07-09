import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model
from torch.nn import functional as F
from torch_scatter import scatter
from distutils.util import strtobool


def predict_composition(loader, model, ld_kwargs):
    """
    Predict the compositions of structures in <loader>
    """
    composition_all = []
    num_atoms_all = []
    
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx+1} in {len(loader)}')

        _, _, z = model.encode(batch)
        num_atoms, lengths_and_angles, lengths, angles, comp_decoded = model.decode_stats(
                                                                            z, gt_num_atoms=None)
        if model.hparams.pred_comp_using_atom_types:
            comp_atom_types = comp_decoded
            current_batch_size = comp_atom_types.size(dim=0)
            comp_atom_types = comp_atom_types.reshape((current_batch_size*model.hparams.max_atoms, model.hparams.data.max_atomic_num + 1))
            # obtain atom types and convert to composition
            comp_atom_types_probs = F.softmax(comp_atom_types, dim=-1)   # Probability of each atom type
            comp_atom_types_probs = F.one_hot(comp_atom_types_probs.argmax(dim=-1), 
                                              num_classes=model.hparams.data.max_atomic_num+1).float()   # One-hot vector of each predicted atom type
            comp_atom_types_probs = comp_atom_types_probs[:,:-1]   # Remove non-atom column
            batch_padded = torch.arange(current_batch_size, device=comp_atom_types_probs.device).repeat_interleave(model.hparams.max_atoms)
            composition = scatter(comp_atom_types_probs, batch_padded, dim=0, reduce='sum')   # yet to be normalized
            composition_num_atoms = composition.sum(dim=-1)   # Number of atoms of each crystal in composition prediction
            composition_num_atoms = composition_num_atoms.unsqueeze(1).repeat_interleave(model.hparams.data.max_atomic_num, dim=1)
            composition = composition / composition_num_atoms
        else:
            composition_per_atom = comp_decoded
            composition_prob = F.softmax(composition_per_atom, dim=-1)
            batch = torch.arange(
                len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
            assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
            composition = scatter(composition_prob, batch, dim=0, reduce='mean')
        
        composition_all.append(composition.detach().cpu())
        num_atoms_all.append(num_atoms.detach().cpu())
    
    composition_all = torch.cat(composition_all, dim=0)
    num_atoms_all = torch.cat(num_atoms_all, dim=0)
    
    return (composition_all, num_atoms_all)


def reconstructon(loader, model, ld_kwargs, num_evals,
                  force_num_atoms=False, force_atom_types=False, down_sample_traj_step=1):
    """
    Reconstruct the crystals in <loader>
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx+1} in {len(loader)}')
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # only sample one z, multiple evals for stochasticity in langevin dynamics
        _, _, z = model.encode(batch)

        for eval_idx in range(num_evals):
            gt_num_atoms = batch.num_atoms if force_num_atoms else None
            gt_atom_types = batch.atom_types if force_atom_types else None
            outputs = model.langevin_dynamics(
                z, ld_kwargs, gt_num_atoms, gt_atom_types)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lengths.append(outputs['lengths'].detach().cpu())
            batch_angles.append(outputs['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    outputs['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    outputs['all_atom_types'][::down_sample_traj_step].detach().cpu())
        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))
        # Save the ground truth structure
        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack, input_data_batch)


def generation(model, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    """
    Generate structures
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        print(f'batch {z_idx+1} in {num_batches_to_sample}')
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.latent_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)


def optimization(model, ld_kwargs, data_loader,
                 num_starting_points=100, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=10):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.latent_dim,
                        device=model.device)
        z.requires_grad = True

    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    interval = num_gradient_steps // (num_saved_crys-1)
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        loss = model.fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps-1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}


def main(args):
    model_path = Path(args.model_path)
    model, data_loader, cfg = load_model(model_path, 
                                         load_data=('recon' in args.tasks) or
                                                   ('pred_comp' in args.tasks) or
                                                   ('opt' in args.tasks and args.start_from == 'data'),
                                         testing=args.testing,
                                         load_saved_datasets=args.load_saved_datasets)
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')
        
    if 'pred_comp' in args.tasks:
        print('Predicting compositions.')
        (composition, num_atoms) = predict_composition(data_loader, model, ld_kwargs)
        
        if args.label == '':
            pred_comp_out_name = 'eval_pred_comp.pt'
        else:
            pred_comp_out_name = f'eval_pred_comp_{args.label}.pt'

        torch.save({'composition': composition,
                    'num_atoms': num_atoms}, 
                   model_path / pred_comp_out_name)

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        start_time = time.time()
        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack, input_data_batch) = reconstructon(
            data_loader, model, ld_kwargs, args.num_evals,
            args.force_num_atoms, args.force_atom_types, args.down_sample_traj_step)

        if args.label == '':
            recon_out_name = 'eval_recon.pt'
        else:
            recon_out_name = f'eval_recon_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / recon_out_name)

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
            model, ld_kwargs, args.num_batches_to_sample, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)

    if 'opt' in args.tasks:
        print('Evaluate model on the property optimization task.')
        start_time = time.time()
        if args.start_from == 'data':
            loader = data_loader
        else:
            loader = None
        optimized_crystals = optimization(model, ld_kwargs, loader)
        optimized_crystals.update({'eval_setting': args,
                                   'time': time.time() - start_time})

        if args.label == '':
            gen_out_name = 'eval_opt.pt'
        else:
            gen_out_name = f'eval_opt_{args.label}.pt'
        torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to saved model (.ckpt file)')
    parser.add_argument('--tasks', nargs='+', default=['gen'], help='task(s) to perform')
    parser.add_argument('--n_step_each', default=100, type=int, help='number of diffusion steps per noise level')
    parser.add_argument('--step_lr', default=1e-4, type=float, help='step size ε as defined in CDVAE paper')
    parser.add_argument('--min_sigma', default=0, type=float, help='minimum sigma to use in annealed langevin dynamics')
    parser.add_argument('--save_traj', default=False, type=lambda x:bool(strtobool(x)), help='if true, save the diffusion trajectory')
    parser.add_argument('--disable_bar', default=False, type=lambda x:bool(strtobool(x)), help='if true, disables the progress bar of langevin dynamics')
    parser.add_argument('--num_evals', default=1, type=int, help='number of times the evaluation task is performed')
    parser.add_argument('--num_batches_to_sample', default=40, type=int, help='number of batches to sample')
    parser.add_argument('--start_from', default='data', type=str, help='if set to "data", perform optimization task on structures in the dataloader')
    parser.add_argument('--batch_size', default=25, type=int, help='batch size')
    parser.add_argument('--force_num_atoms', action='store_true', help='perform reconstruction task using ground truth number of atoms')
    parser.add_argument('--force_atom_types', action='store_true', help='perform reconstruction task using ground truth atom types')
    parser.add_argument('--down_sample_traj_step', default=10, type=int, help='downsample the trajectory to every n steps')
    parser.add_argument('--label', default='', help='suffix label added to saved files')
    parser.add_argument('--testing', default=True, type=lambda x:bool(strtobool(x)), help='if true, load the test set; if false, loads the validation set')
    parser.add_argument('--load_saved_datasets', default=True, type=lambda x:bool(strtobool(x)), help='if true, load dataset from saved .pt file')

    args = parser.parse_args()

    main(args)
    