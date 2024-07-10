# DiffCSP

## Table of contents
- [Prerequisites](#prerequisites)
- [Setting up DiffCSP](#setting-up-diffcsp)
- [Configuring DiffCSP training runs](#configuring-diffcsp-training-runs)
- [Training DiffCSP](#training-diffcsp)
- [Restarting training](#restarting-training)
- [Generating structures](#generating-structures)
- [Note regarding dataset loading](#note-regarding-dataset-loading)

## Prerequisites
DiffCSP requires the same packages as CDVAE, plus a few additional packages. 
Please create a CDVAE environment first as per the instructions [here](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/cdvae),
then install the following additional packages:
- pyshtools = 4.10.4
- pyxtal = 0.6.0
- einops = 0.8.0
- bottleneck = 1.3.6

Note that these are only suggested package versions; other versions may also be compatible.

## Setting up DiffCSP
1. If you have just created the conda environment, login to your [wanb](https://docs.wandb.ai/quickstart) account by running:
    ```
    wandb login
    ```

2. Enter the directory where the diffcsp files are located, and install the diffcsp package using the following command:
    ```
    pip install -e .
    ```

3. Place your datasets somewhere, and set the correct paths to the datasets in the data configuration [YAML files](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/diffcsp/conf/data).
   Set the paths to the `root_path` key in the YAML files.

4. Create a directory for running a job (for example, `my_run`), and enter the directory.<br/>

5. a) Create a `.env` file (see [.env.template](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/diffcsp/.env.template) for an example), and set the correct paths.<br />
 `PROJECT_ROOT` should point to the diffcsp directory.<br />
 `HYDRA_JOBS` and `WABDB_DIR` should point to directories within `my_run` (where the job will be run).<br />
 Please use absolute paths instead of relative paths.<br />
   b) Alternatively, you can simply set the environment variables directly (for example, in your job submission script).
      ```
      export PROJECT_ROOT="/path/to/diffcsp"
      export HYDRA_JOBS="/path/to/my_run/hydra"
      export WABDB_DIR="/path/to/my_run/wabdb"
      ```

## Configuring DiffCSP training runs
All parameters for configuring DiffCSP can be set in the configuration [YAML files](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/diffcsp/conf). 

The parameters have self-explanatory names, please refer to the original [paper](https://arxiv.org/abs/2309.04475).

Note that `teacher_forcing_lattice` in the data configuration [YAML files](https://github.com/ertekin-research-group/Dismai-Bench/tree/main/models/diffcsp/conf/data) 
turns teacher forcing of the lattice parameters on or off. 

To benchmark DiffCSP as done in the Dismai-Bench paper, set `sigma_end` in [wrapped.yaml](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/diffcsp/conf/model/sigma_scheduler/wrapped.yaml)
to 0.1 (default) for the disordered interfaces/alloys, and 0.05 for amorphous silicon. The rest of the parameters can be left unchanged.

## Training DiffCSP
To train a DiffCSP model, run the following command:
```
python path/to/diffcsp/run.py data=alloy_300K_narrow expname=alloy_300K_narrow_model_1 runname=run_1
```
`data` should be set based on the name of the data configuration YAML file. 

After training, model checkpoints can be found in `$HYDRA_JOBS/singlerun/YYYY-MM-DD/expname`.

## Restarting training
Simply train DiffCSP as usual and it will automatically look for the last checkpoint to load.

Alternatively, you can also supply a checkpoint to load by setting the `load_checkpoint_path` in the train configuration [YAML file](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/diffcsp/conf/train/default.yaml). 

## Generating structures
To generate structures, you will need the files saved in `$HYDRA_JOBS/singlerun/YYYY-MM-DD/expname`, specifically `hparams.yaml`,
`lattice_scaler.pt`, `prop_scaler.pt` and a `.ckpt` file.

You will also need to set the environment variable for `PROJECT_ROOT`:
```
export PROJECT_ROOT="/path/to/diffcsp"
```

Use the [evaluate.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/diffcsp/scripts/evaluate.py) script to generate structures:
```
python /path/to/scripts/evaluate.py --model_path /path/to/dir/containing/checkpoint/and/supporting/files --sampling_batch_size 20 --num_evals 50 --timesteps 1000
```

The generated structures are saved as a `.pt` file. 
Convert the saved `.pt` file to a `.extxyz` file using the [convert_gen_pt_to_extxyz.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/models/diffcsp/scripts/convert_gen_pt_to_extxyz.py) script:
```
python /path/to/scripts/convert_gen_pt_to_extxyz.py --data eval_gen.pt
```

## Note regarding dataset loading
*If you have downloaded the datasets provided by us, the datasets already contain the saved graphs.*

If you provide your own csv dataset to train DiffCSP, DiffCSP will first build the graphs from the csv datasets.
We suggest setting `preprocess_workers` in the data configuration YAML file to 1-2x the number of cpus.
Once the graphs are built, DiffCSP will automatically save the graphs as `.pt`files.

The next time DiffCSP is run on the same dataset, the graphs will be directly loaded without needing to be built again.
