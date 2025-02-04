#!/bin/bash
#SBATCH --job-name test_pyrosetta
#SBATCH --output=mood_seqs.out
#SBATCH --error=mood_seqs.err
#SBATCH --ntasks=5
#SBATCH --time=00-02:00:00
#SBATCH --qos=gp_debug

module purge

# ml ucx/1.16.0-gcc
# ml openmpi/5.0.5-gcc
ml ucx/1.15.0-gcc
ml mpich/4.2.2-gcc

module load anaconda

source activate /gpfs/projects/bsc72/conda_envs/mood

mpirun -n 4 /gpfs/projects/bsc72/conda_envs/mood/bin/python /gpfs/projects/bsc72/acanella/Repos/multiObjectiveOptimizationDesign/tests/metrics/test_pyrosetta.py