#!/bin/bash 

conda activate cl-ica

# ressource specs
TIME=4:0:00
NUM_WORKERS=4
MEM_PER_CPU=5G

# Ablation paramters
SAMPLEs=(1000 10000)
Ns=(5 10 20)
NR_CLUSTERs=(10 100 1000 10000) 
C_PARAMs=(5 10 50)
MPs=(0 1 2)
CPs=(0 1 2)

################################################ NON NORMALIZED W ################################################
# Number of samples
for SAMPLE in "${SAMPLEs[@]}"
do
JOB="python ../main_diet.py --nr-indices $SAMPLE"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Dimensionality of representation
for N in "${Ns[@]}"
do
JOB="python ../main_diet.py --n $N"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Number of clusters
for NR_CLUSTER in "${NR_CLUSTERs[@]}"
do
JOB="python ../main_diet.py --nr-clusters $NR_CLUSTER"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Variance of latent distribution
for C_PARAM in "${C_PARAMs[@]}"
do
JOB="python ../main_diet.py --c-param $C_PARAM"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Type of distribution of cluster vectors
for MP in "${MPs[@]}"
do
JOB="python ../main_diet.py --m-p $MP"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Type of distribution of latent
for CP in "${CPs[@]}"
do
JOB="python ../main_diet.py --c-p $CP"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;


################################################ NORMALIZED W ################################################
# Number of samples
for SAMPLE in "${SAMPLEs[@]}"
do
JOB="python ../main_diet.py --nr-indices $SAMPLE --normalize-rows-W"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Dimensionality of representation
for N in "${Ns[@]}"
do
JOB="python ../main_diet.py --n $N --normalize-rows-W "
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Number of clusters
for NR_CLUSTER in "${NR_CLUSTERs[@]}"
do
JOB="python ../main_diet.py --nr-clusters $NR_CLUSTER --normalize-rows-W"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Variance of latent distribution
for C_PARAM in "${C_PARAMs[@]}"
do
JOB="python ../main_diet.py --c-param $C_PARAM --normalize-rows-W"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Type of distribution of cluster vectors
for MP in "${MPs[@]}"
do
JOB="python ../main_diet.py --m-p $MP --normalize-rows-W"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;

# Type of distribution of latent
for CP in "${CPs[@]}"
do
JOB="python ../main_diet.py --c-p $CP --normalize-rows-W"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="nvidia-smi;$JOB"
done;
