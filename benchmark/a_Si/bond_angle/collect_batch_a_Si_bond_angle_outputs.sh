## Set the number of batches
n_batch=10


cp batch_1/bond_angles.csv .

for i in $(seq 2 $n_batch)
do
  tail -n +2 batch_"$i"/bond_angles.csv >> bond_angles.csv
done
