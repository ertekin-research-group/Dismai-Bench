# Disordered LSC-LCO interface benchmarking

## Table of contents
- [Structural relaxation](#structural-relaxation)
- [Coordination motif analysis](#coordination-motif-analysis)

## Structural relaxation
All necessary scripts are found in the `relax` directory.

1. Prepare your dataset in `.extxyz` format.

2. Post-process the structures to move apart atoms that are too close together:
   ```
   python /path/to/relax/move_atoms_apart.py --data_path gen.extxyz
   ```
   Any structure that still has atoms too close together (<1.5 Å by default) after `max_iter` interations (100 by default) will be rejected.

3. Download the M3GNet potential provided by us, and the set the `model_path` in 
   [relax_int.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/benchmark/int/relax/relax_int.py) and 
   [calc_int_energy.py](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/benchmark/int/relax/calc_int_energy.py)
   to the directory containing the M3GNet potential.

5. (Optional) Calculate the energy of the structures (no relaxation):
   ```
   python /path/to/relax/calc_int_energy.py --data_path gen_clean.extxyz
   ```

6. Run M3GNet relaxations:
   ```
   python /path/to/relax/relax_int.py --data_path gen_clean.extxyz
   ```
   The relaxations can be split into batches and run in parallel over multiple cpus, if the `split_batches` tag is set to True (default).

   See [batch_relax_slurm.sh](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/benchmark/int/relax/batch_relax_slurm.sh) for an
   example job submission script (written for Slurm job schedulers) to run jobs in parallel.
   A working directory is created for each batch, and the `batch` tag in used by `relax_int.py` to identify the batch number.

   Once all batches have completed their relaxations, collect the results by running the following command:
   ```
   python /path/to/relax/collect_batch_relax_outputs.py --n_strucs_per_batch 100
   ```

   When a structure fails to be relaxed, it would simply be logged in `discarded.csv`, and the relaxations would just continue.
   On rare occassions, especially if the structure is very poor, the entire job can crash.
   If this happens, simply run `relax_int.py` again and it should restart the relaxations where it left off.
   When collecting results using `collect_batch_relax_outputs.py`, it will check if any of the batches had crashed, and return an error message if so.

7. Plot the interface energy distribution:
   ```
   python /path/to/relax/plot_int_energy_distribution.py --gen_data relax_results.csv --train_data /path/to/data/dismai_bench_train_ref_data/int/train_energy.csv
   ```

## Coordination motif analysis
All necessary scripts are found in the `motif` directory.

1. Calculate coordination motif fingerprints:
   ```
   python /path/to/motif/get_int_motifs.py --data_path gen_relaxed.extxyz
   ```
   These calculations can be split into batches and run in parallel over multiple cpus, if the `split_batches` tag is set to True (default).

   See [batch_int_motif_slurm.sh](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/benchmark/int/motif/batch_int_motif_slurm.sh) for an
   example job submission script (written for Slurm job schedulers) to run jobs in parallel.
   A working directory is created for each batch, and the `batch` tag in used by `get_int_motifs.py` to identify the batch number.

   To collect the results once all batches have completed, see [collect_batch_int_motif_outputs.sh](https://github.com/ertekin-research-group/Dismai-Bench/blob/main/benchmark/int/motif/collect_batch_int_motif_outputs.sh) 
   for an example bash script. 

2. Calculate coodination motif metrics:
   ```
   python /path/to/motif/get_int_motif_metrics.py --data_train_dir /path/to/data/dismai_bench_train_ref_data/int/train_motif --n_strucs_ori 1000
   ```
   Remember to download the training dataset motif data and set `data_train_dir` to the correct path.

   Set `n_strucs_ori` to the number of structures that was in your original dataset (before any post-processing and relaxation).

4. Get the distributions of most likely coordination motifs:

   Li motifs as example here,
   ```
   python /path/to/motif/plot_most_likely_motifs/get_most_likely_Li_OP.py --data_train_dir /path/to/data/dismai_bench_train_ref_data/int/train_motif/cnn_stats_Li.csv
   ```

5. Plot the distributions of most likely coordination motifs:

   Li motifs as example here,
   ```
   python /path/to/motif/plot_most_likely_motifs/plot_most_likely_Li_OP.py
   ```
   <p align="center">
     <img src="../assets/int_motifs.png" width="500"> 
   </p>
