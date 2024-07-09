import time
import argparse
import torch

from pathlib import Path
from torch_geometric.data import Batch, DataLoader
from torch.utils.data import Dataset

from eval_utils import load_model, lattices_to_params_shape
from distutils.util import strtobool


def diffusion(loader, model, num_evals, step_lr, timesteps):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        for eval_idx in range(num_evals):
            print(f'batch {idx+1} / {len(loader)}, sample {eval_idx+1} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr = step_lr, timesteps = timesteps)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

        input_data_list = input_data_list + batch.to_data_list()


    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)


    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch
    )


class SampleDataset(Dataset):
    def __init__(self, data, n_strucs):
        super().__init__()
        self.data = data
        self.n_strucs = n_strucs

    def __len__(self) -> int:
        return self.n_strucs

    def __getitem__(self, index):
        return self.data


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=True)
    
    if args.sample_composition:
        # Sample a fixed composition with the specified batch size
        test_dataset = SampleDataset(test_loader.dataset[0], args.sampling_batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.sampling_batch_size)

    if torch.cuda.is_available():
        model.to('cuda')

    print('Sampling structures based on test set...')

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch) = diffusion(
        test_loader, model, args.num_evals, args.step_lr, args.timesteps)

    if args.label == '':
        diff_out_name = 'eval_gen.pt'
    else:
        diff_out_name = f'eval_gen_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'input_data_batch': input_data_batch,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lattices': lattices,
        'lengths': lengths,
        'angles': angles,
        'time': time.time() - start_time,
    }, model_path / diff_out_name)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to saved model (.ckpt file)')
    parser.add_argument('--step_lr', default=1e-5, type=float, help='step size γ as defined in DiffCSP paper')
    parser.add_argument('--timesteps', default=1000, type=int, help='number of diffusion steps')
    parser.add_argument('--num_evals', default=50, type=int, help='number of times the evaluation task is performed')
    parser.add_argument('--label', default='', help='suffix label added to saved files')
    parser.add_argument('--sample_composition', default=True, type=lambda x:bool(strtobool(x)), help='if true, sample structures of a fixed composition')
    parser.add_argument('--sampling_batch_size', default=20, type=int, help='if sample_composition is true, this sets the size of each batch (total number of samples = num_evals * sampling_batch_size)')
    
    args = parser.parse_args()
    main(args)
