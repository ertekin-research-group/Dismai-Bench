## Set the number of batches
n_batch=10


cp batch_1/cnn_stats_Li.csv .
cp batch_1/cnn_stats_Sc.csv .
cp batch_1/cnn_stats_Co.csv .

for i in $(seq 2 $n_batch)
do
  tail -n +2 batch_"$i"/cnn_stats_Li.csv >> cnn_stats_Li.csv
  tail -n +2 batch_"$i"/cnn_stats_Sc.csv >> cnn_stats_Sc.csv
  tail -n +2 batch_"$i"/cnn_stats_Co.csv >> cnn_stats_Co.csv
done
