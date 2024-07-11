# UniMat

## Table of contents
- [Prerequisites](#prerequisites)
- [Training UniMat](#training-unimat)
- [Restarting training](#restarting-training)
- [Generating structures](#generating-structures)

## Prerequisites
UniMat requires the following packages:
- pytorch = 1.12.1
- imagen-pytorch = 1.26.2
- ase = 3.22.0
- tqdm = 4.64.1

Note that these are only suggested package versions; other versions may also be compatible. 
We suggest installing the `pytorch` version closest to one listed above, that is compatible with your CUDA version.

## Training UniMat
To train a UniMat model, run the following command:
```
python train_video_diff.py --data_dir /path/to/data/train_val_data/unimat/int
```
Model checkpoints will be saved to `$save_dir` (`model_saves` by default).

## Restarting training
To restart training, load the last checkpoint as follows:
```
python train_video_diff.py --data_dir /path/to/data/train_val_data/unimat/int --n_steps 300001 --load_ckpt /path/to/checkpoint_last.pt
```

## Generating structures
Use the [generate_unimat.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/unimat/generate_unimat.py) script to generate structures:
```
python generate_unimat.py --load_ckpt /path/to/checkpoint_best.pt --param /path/to/data/train_val_data/unimat/int/param.json
```
Please remember to set the path to the `param.json` file found in our provided dataset. 

The generated structures are saved as a `unimat_gen.pt` file. 
Convert the saved `.pt` file to a `.extxyz` file using the [convert_unimat_to_extxyz.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/unimat/convert_unimat_to_extxyz.py) script:
```
python convert_unimat_to_extxyz.py --unimat_data unimat_gen.pt --train_data_dir /path/to/data/train_val_data/unimat/int
```
Please remember to set the path to the directory containing the training dataset; files in the directory are used as reference to reconstruct the structures.

The script will also output an `element_accuracy.csv` that lists the atomic species and composition accuracies. 
Only structures with the correct composition will be saved into the `.extxyz` file.
