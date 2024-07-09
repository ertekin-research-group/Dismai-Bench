## Set the number of batches
n_batch=10


cp batch_1/cnn_stats_Si.csv .

for i in $(seq 2 $n_batch)
do
  tail -n +2 batch_"$i"/cnn_stats_Si.csv >> cnn_stats_Si.csv
done
