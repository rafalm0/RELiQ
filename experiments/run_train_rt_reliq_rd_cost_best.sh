#!/encs/bin/tcsh
#SBATCH -J "rt-reliq-rd-cost"   # job name
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=4  # number of CPU cores per task
#SBATCH --gpus=1   # gpu devices per node
#SBATCH --mem=20G   # memory
#SBATCH --nodelist=speed-[37-42],speed-17,speed-01,speed-03,speed-05
#SBATCH --mail-type=ALL
#SBATCH --time=7-00:00:00   # job time limit
#SBATCH --output=lastlog_%x_%j.out

module load python/3.12.0/default
mkdir -p /speed-scratch/$USER/tmp
setenv TMP /speed-scratch/$USER/tmp
setenv TMPDIR /speed-scratch/$USER/tmp
module load cuda/12.8/default
setenv PIP_CACHE_DIR /speed-scratch/$USER/tmp/reliqvenv/cache

# Use the venv if it is ready, otherwise create it. The mkdir lock lets only one job
# create/install, so jobs starting at the same time cannot install into it together.
set VENV = /speed-scratch/$USER/QRL/RELiQ/Reliqvenv
if ( ! -f $VENV/.ready ) then
  mkdir $VENV.lock >& /dev/null
  if ( $status == 0 ) then
    if ( ! -f $VENV/bin/activate.csh ) python -m venv $VENV
    source $VENV/bin/activate.csh
    pip install -r /speed-scratch/$USER/QRL/replica/RELiQ/requirements.txt
    if ( $status != 0 ) then
      rmdir $VENV.lock
      exit 1
    endif
    touch $VENV/.ready
    rmdir $VENV.lock
  else
    echo "another job is setting up the venv, waiting"
    while ( ! -f $VENV/.ready && -d $VENV.lock )
      sleep 30
    end
    if ( ! -f $VENV/.ready ) exit 1
  endif
endif
source $VENV/bin/activate.csh

echo "Starting Script"

cd /speed-scratch/$USER/QRL/replica/RELiQ
setenv METHODS rt-reliq-rd-cost
setenv PHASES "train validate"
bash experiments/train_best.sh
exit $status
