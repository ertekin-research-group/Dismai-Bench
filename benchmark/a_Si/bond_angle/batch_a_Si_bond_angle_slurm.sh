## Set job name prefix
job_name=my_job
## Set starting and ending batches
i_batch_start=1
i_batch_end=10

cp /path/to/benchmark/a_Si/bond_angle/get_a_Si_bond_angle.py .

for i in $(seq $i_batch_start $i_batch_end)
do
  mkdir batch_"$i"
  cd batch_"$i"

  echo Batch $i

  cat >submit_script_slurm <<!
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4500
#SBATCH --time=24:00:00
#SBATCH --job-name="$job_name"_batch"$i"
#SBATCH -o job.stdout
#SBATCH -e %j.err

# Module loading/unloading
cd \$SLURM_SUBMIT_DIR
module load conda
conda activate my_env

python ../get_a_Si_bond_angle.py --batch $i --n_strucs_per_batch 100
!

  sbatch submit_script_slurm
  cd ..

done
