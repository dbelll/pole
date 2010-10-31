#!/bin/bash
#
### Using bash
#$ -S /bin/bash
#
#
### Combine output and error messages
#$ -j y
#$ -o testing.outerr.$JOB_ID
##$ -e testing.outerr.$JOB_ID
#
### Work in the directory where submitted
#$ -cwd
#
### request 8 GPU cores == four compute hosts ( 8-way MPI job)
#$ -pe ortegpu 8
#
## submit to special GPU queue (uses the above 'ortegpu' PE)
#$ -q gpubatch.q
#
### request maximum of 30 minutes of compute time
#$ -l h_rt=00:30:00
#

echo " ############### Script Started ############"


#
# Use modules to setup the runtime environment
#
. /etc/profile

# These need to be the same as when the executable was compiled:
module load  packages/cuda/2.3
module load compilers/gcc/4.3.3 
#module load mpi/openmpi/1.2.8/gnu


#
# Execute the job
#

#mpirun -np $NSLOTS ./bin/mpi_boxcar -s 100
./scripts/res_differ_400ms.sh

#
# finish up
#
date 

exit

############################################
