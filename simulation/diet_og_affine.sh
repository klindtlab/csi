#!/bin/bash 

module load stack/2024-06 python/3.11.6
source $HOME/mae/bin/activate

TIME=48:00:00
NUM_WORKERS=2
MEM_PER_CPU=10G

SEEDs=(1 2 3 4)
Ns=(5 10 20)
NR_CLUSTERs=(10 100 1000 10000)
C_PARAMs=(5 10 50)
MPs=(0 1 2)
CPs=(0 1 2)
SAVE_FOLDER="./outputs"

for SEED in "${SEEDs[@]}"
do

for N in "${Ns[@]}"
do
JOB="python main_diet_og.py --seed $SEED --n $N"
echo $JOB
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --save_folder "$SAVE_FOLDER" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1  --wrap="$JOB"
done;

for NR_CLUSTER in "${NR_CLUSTERs[@]}"
do
JOB="python main_diet_og.py --seed $SEED --nr-clusters $NR_CLUSTER"
echo $JOB
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --save_folder "$SAVE_FOLDER" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1  --wrap="$JOB"
done;

for C_PARAM in "${C_PARAMs[@]}"
do
JOB="python main_diet_og.py --seed $SEED --c-param $C_PARAM"
echo $JOB
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --save_folder "$SAVE_FOLDER" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --wrap="$JOB"
done;

for MP in "${MPs[@]}"
do
JOB="python main_diet_og.py --seed $SEED --m-p $MP"
echo $JOB
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --save_folder "$SAVE_FOLDER" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1  --wrap="$JOB"
done;

for CP in "${CPs[@]}"
do
JOB="python main_diet_og.py --seed $SEED --c-p $CP"
echo $JOB
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --save_folder "$SAVE_FOLDER" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1  --wrap="$JOB"
done;

done; 
