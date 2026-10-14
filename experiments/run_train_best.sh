#!/encs/bin/tcsh
#SBATCH -J "RELiq train best"   # job name
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=4  # number of CPU cores per task
#SBATCH --gpus=1   # gpu devices per node
#SBATCH --mem=30G   # memory
#SBATCH --nodelist=speed-[37-42],speed-17,speed-01,speed-03,speed-05
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00   # job time limit
#SBATCH --output=lastlog.out

module load python/3.12.0/default
mkdir -p /speed-scratch/$USER/tmp
setenv TMP /speed-scratch/$USER/tmp
setenv TMPDIR /speed-scratch/$USER/tmp
python -m venv $TMPDIR/reliqvenv
module load cuda/12.8/default
setenv PIP_CACHE_DIR /speed-scratch/$USER/tmp/reliqvenv/cache
source /speed-scratch/$USER/QRL/RELiQ/Reliqvenv/bin/activate.csh
pip install -r /speed-scratch/r_albuq/QRL/replica/RELiQ/requirements.txt

echo "Starting Script"

cd /speed-scratch/$USER/QRL/replica/RELiQ
bash experiments/train_best.sh
exit $status
