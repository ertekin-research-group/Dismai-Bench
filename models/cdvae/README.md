# CDVAE
## Prerequisites
CDVAE requires the following packages:
- pytorch = 1.13.0
- pytorch-lightning = 2.0.9
- torch-geometric = 2.3.1
- ase = 3.22.1
- autopep8 = 2.0.4
- seaborn = 0.12.2
- tqdm = 4.66.1
- nglview = 3.0.8
- higher = 0.2.1
- hydra-core = 1.1.0
- hydra-joblib-launcher = 1.1.5
- p-tqdm = 1.3.3
- pytest = 7.4.2
- python-dotenv = 1.0.0
- smact = 2.2.1
- streamlit = 0.79.0
- torchdiffeq = 0.2.3
- wandb = 0.15.10
- matminer = 0.7.3
- protobuf = 3.20.3

Note that these are only suggested package versions; other versions may also be compatible (e.g., `pytorch=1.10.2` + `pytorch-lightning=1.9.4` + `torch-geometric=2.1.0`). 
We suggest installing the `pytorch` version closest to one listed above, that is compatible with your CUDA version.
You will also need to install the appropriate `torch-geometric` version based on your `pytorch` version.

We suggest installing the packages in the following sequence, note that the installation will take some time:
```
conda create --name cdvae_env python=3.9
conda activate cdvae_env
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge pytorch-lightning
conda install -c conda-forge ase autopep8 seaborn tqdm nglview
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install higher hydra-core==1.1.0 hydra-joblib-launcher==1.1.5 p-tqdm==1.3.3 pytest python-dotenv smact==2.2.1 streamlit==0.79.0 torchdiffeq wandb
pip install matminer==0.7.3
pip install protobuf==3.20.3
```
Please change the lines corresponding to [pytorch](https://pytorch.org/get-started/locally/) and [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) based on the versions you want to install.

## Setting up CDVAE
1. If you have just created the conda environment, login to your [wanb](https://docs.wandb.ai/quickstart) account by running:
    ```
    wandb login
    ```

2. Enter the directory where the cdvae files are located, and install the cdvae package using the following command:
    ```
    pip install -e .
    ```

3. Place your datasets somewhere, and set the correct paths to the datasets in the data configuration [YAML files](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/cdvae/conf/data).
   Set the paths to the `root_path` key in the YAML files.

4. Create a directory for running a job (for example, `my_run`), and enter the directory.<br/>

5. a) Create a `.env` file (see [.env.template](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/cdvae/.env.template) for an example), and set the correct paths.<br />
 `PROJECT_ROOT` should point to the cdvae directory.<br />
 `HYDRA_JOBS` and `WABDB_DIR` should point to directories within `my_run` (where the job will be run).<br />
 Please use absolute paths instead of relative paths.<br />
   b) Alternatively, you can simply set the environment variables directly (for example, in your job submission script).
      ```
      export PROJECT_ROOT="/path/to/cdvae"
      export HYDRA_JOBS="/path/to/my_run/hydra"
      export WABDB_DIR="/path/to/my_run/wabdb"
      ```

## Configuring CDVAE training runs
All parameters for configuring CDVAE can be set in the configuration [YAML files](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/cdvae/conf). 

The parameters have self-explanatory names, please refer to the original [paper](https://arxiv.org/abs/2110.06197).

Note that `denoise_atom_types` in [vae.yaml](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/cdvae/conf/model/vae.yaml) 
turns atomic species denoising on or off. 

To benchmark CDVAE as done in the Dismai-Bench paper, simply leave the parameters unchanged.

## (Optional) Saving graphs for faster dataset loading
*If you have downloaded the datasets provided by us, you don't need to perform this step.*

When CDVAE is run, it initially builds the graphs from the input csv datasets, which can be time-consuming especially for structures with large number of atoms. 
You can save the constructed graphs using the [save_datasets.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/cdvae/cdvae/save_datasets.py) script, 
and load the graphs directly the next time.

Create your dataset in `.csv` format (see [CDVAE datasets](https://github.com/txie-93/cdvae/tree/main/data) for reference).
Then, create the corresponding data configuration [YAML file](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/cdvae/conf/data).
We suggest setting `preprocess_workers` in the YAML file to 1-2x the number of cpus.

Then, use the following command to save the graphs (alloy dataset as example here):
```
python /path/to/cdvae/save_datasets.py data=alloy_300K_narrow
```
`data` should be set based on the name of the data configuration YAML file.

The graphs will be saved as `.pt` files in the same directory as your `.csv` datasets. 

Ensure that `load_saved_datasets` is set to true in the data configuration YAML file, and the graphs will be loaded the next time you train CDVAE on the same dataset.

## Training CDVAE
To train a CDVAE model, run the following command:
```
python path/to/cdvae/run.py data=alloy_300K_narrow expname=alloy_300K_narrow_model_1 runname=run_1
```
`data` should be set based on the name of the data configuration YAML file. 

After training, model checkpoints can be found in `$HYDRA_JOBS/singlerun/YYYY-MM-DD/expname`.

## Restarting training
Simply train CDVAE as usual and it will automatically look for the last checkpoint to load.

Alternatively, you can also supply a checkpoint to load by setting the `load_checkpoint_path` in the train configuration [YAML file](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/cdvae/conf/train/default.yaml). 

## Generating structures
To generate structures, you will need the files saved in `$HYDRA_JOBS/singlerun/YYYY-MM-DD/expname`, specifically `hparams.yaml`,
`lattice_scaler.pt`, `prop_scaler.pt` and a `.ckpt` file.

You will also need to set the environment variable for `PROJECT_ROOT`:
```
export PROJECT_ROOT="/path/to/cdvae"
```

Use the [evaluate.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/cdvae/scripts/evaluate.py) script to generate structures:
```
python /path/to/scripts/evaluate.py --model_path /path/to/dir/containing/checkpoint/and/supporting/files --tasks gen --num_batches_to_samples 40 --batch_size 25 --n_step_each 100
```

The generated structures are saved as a `.pt` file. 
Convert the saved `.pt` file to a `.extxyz` file using the [convert_gen_pt_to_extxyz.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/cdvae/scripts/convert_gen_pt_to_extxyz.py) script:
```
python /path/to/scripts/convert_gen_pt_to_extxyz.py --data eval_gen.pt
```
