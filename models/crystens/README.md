# CrysTens

## Table of contents
- [Prerequisites](#prerequisites)
- [Training CrysTens](#training-crystens)
- [Restarting training](#restarting-training)
- [Generating structures](#generating-structures)

## Prerequisites
CrysTens requires the following packages:
- pytorch = 1.12.1
- imagen-pytorch = 1.26.2
- ase = 3.22.0
- tqdm = 4.64.1

Note that these are only suggested package versions; other versions may also be compatible. 
We suggest installing the `pytorch` version closest to one listed above, that is compatible with your CUDA version.

## Training CrysTens
CrysTens uses a cascaded diffusion model (Imagen) consisting of two U-Nets. 

To train the 1st U-Net, run the following command:
```
python train_imagen.py --data_dir /path/to/data/train_val_data/crystens/int --unet_number 1
```
Model checkpoints will be saved to `$save_dir` (`model_saves` by default).

After training the 1st U-Net, load the saved checkpoint and train the 2nd U-Net using the following command:
```
python train_imagen.py --data_dir /path/to/data/train_val_data/crystens/int --unet_number 2 --load_ckpt /path/to/checkpoint_best.pt
```
We suggest training the two U-Nets in separate directories.

## Restarting training
To restart training for a certain U-Net (for example, the 1st U-Net), load the last checkpoint and use the `cont_train_same` tag as follows:
```
python train_imagen.py --data_dir /path/to/data/train_val_data/crystens/int --unet_number 1 --n_steps 300001 --cont_train_same --load_ckpt /path/to/checkpoint_last.pt
```

## Generating structures
After training both U-Nets, use the [generate_crys_tens.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/crystens/generate_crys_tens.py) script to generate structures:
```
python generate_crys_tens.py --load_ckpt /path/to/checkpoint_best.pt --param /path/to/data/train_val_data/crystens/int/param.json
```
Please remember to set the path to the `param.json` file found in our provided dataset. 

The generated structures are saved as a `crys_tens_gen.pt` file. 
Convert the saved `.pt` file to a `.extxyz` file using the [convert_crys_tens_to_extxyz.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/crystens/convert_crys_tens_to_extxyz.py) script:
```
python convert_crys_tens_to_extxyz.py --crys_tens_data crys_tens_gen.pt --train_data_dir /path/to/data/train_val_data/crystens/int
```
Please remember to set the path to the directory containing the training dataset; files in the directory are used as reference to reconstruct the structures.

The script will also output a `statistics.csv` that lists the reconstruction errors.
