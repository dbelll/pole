#!/bin/bash
# Testing script for res computer
#
#  Test restart interavl
#

_trials="--TRIALS=1024"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"

#_restart="--RESTART_INTERVAL=8192"
_test="--TEST_REPS=8192 --TEST_INTERVAL=16384"
_sharing="--SHARING_INTERVAL=16384"  # use large value to maximize chunk size
_time="--TIME_STEPS=524288"
_grpsize="--AGENT_GROUP_SIZE=1"

_a=.10
_e=.00
_l=.70


for _re in 65536 131072 262144
do
_restart="--RESTART_INTERVAL=$_re"
_parms="--EPSILON=$_e --LAMBDA=$_l --ALPHA=$_a"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common $_grpsize $_time $_test $_restart
 
done

