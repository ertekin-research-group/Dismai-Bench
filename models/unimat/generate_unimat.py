import argparse
import torch
from imagen_pytorch import Unet3D, Imagen, ImagenTrainer
import json


def imagen_model(args):
    """
    Constructs an Imagen model
    """
    unet = Unet3D(
                  dim = args.dim, 
                  dim_mults = (1, 2, 4),
                  num_resnet_blocks = 3,
                  layer_attns = (False, True, True),
                  layer_cross_attns = (False, True, True)
                  )

    imagen = Imagen(
                    condition_on_text = False,
                    unets = unet,
                    image_sizes = args.img_size,
                    timesteps = args.timesteps,
                    channels = 3,
                    auto_normalize_img = False
                    )
    
    return imagen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', type=str, help='path to checkpoint to be loaded')
    parser.add_argument('--dim', type=int, default=64, help='number of base channels of the U-Net')
    parser.add_argument('--param', type=str, default='/path/to/data/train_val_data/unimat/int/param.json', help='path to dataset parameters (json file)')
    parser.add_argument('--num_batches_to_sample', type=int, default=40, help='number of batches to sample')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps')
    
    args = parser.parse_args()
    
    with open(args.param) as f:
        param = json.load(f)
    args.img_size = param['img_len']
    imagen = imagen_model(args)
    trainer = ImagenTrainer(imagen)
    trainer.load(args.load_ckpt)
    
    videos_all = []
    for i in range(args.num_batches_to_sample):
        print(f'Batch {i+1} in {args.num_batches_to_sample}')
        videos = trainer.sample(batch_size=args.batch_size, video_frames=param['n_atoms'])
        videos = videos.cpu().detach()
        videos_all.append(videos)
    
    videos_all = torch.cat(videos_all, axis=0)
    torch.save(videos_all, 'unimat_gen.pt')


if __name__ == '__main__':
    main()




