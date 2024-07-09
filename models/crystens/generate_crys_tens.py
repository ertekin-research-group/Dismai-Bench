import argparse
import torch
from imagen_pytorch import Unet, SRUnet256, Imagen, ImagenTrainer
import json


def imagen_model(args):
    """
    Constructs an Imagen model
    """
    unet1 = Unet(
                 dim = args.dim,
                 dim_mults = (1, 2, 4),
                 num_resnet_blocks = 3,
                 layer_attns = (False, True, True),
                 layer_cross_attns = (False, True, True),
                 use_linear_attn = True,
                 channels = 4,
                 channels_out = 4
                 )

    unet2 = SRUnet256(
                      dim = args.dim, 
                      dim_mults = (1, 2, 4),
                      num_resnet_blocks = (2, 4, 8),
                      layer_attns = (False, False, True),
                      layer_cross_attns = (False, False, True),
                      channels = 4,
                      channels_out = 4
                      )

    imagen = Imagen(
                    condition_on_text = False,
                    unets = (unet1, unet2),
                    image_sizes = (64, args.img_size),
                    timesteps = args.timesteps,
                    channels = 4,
                    )
    
    return imagen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', type=str, help='path to checkpoint to be loaded')
    parser.add_argument('--dim', type=int, default=64, help='number of base channels of the U-Net')
    parser.add_argument('--param', type=str, default='/path/to/data/train_val_data/crystens/int/param.json', help='path to dataset parameters (json file)')
    parser.add_argument('--num_batches_to_sample', type=int, default=40, help='number of batches to sample')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps')
    
    args = parser.parse_args()
    
    with open(args.param) as f:
        param = json.load(f)
    args.img_size = 11 + param['padding'] + param['max_n_atoms']
    imagen = imagen_model(args)
    trainer = ImagenTrainer(imagen)
    trainer.load(args.load_ckpt)
    
    images_all = []
    for i in range(args.num_batches_to_sample):
        print(f'Batch {i+1} in {args.num_batches_to_sample}')
        images = trainer.sample(batch_size=args.batch_size)
        images = images.cpu().detach()
        images_all.append(images)
    
    images_all = torch.cat(images_all, axis=0)
    torch.save(images_all, 'crys_tens_gen.pt')


if __name__ == '__main__':
    main()




