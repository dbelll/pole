#!/bin/bash
# Testing script for res computer
#
#  Test restart interval
#

_trials="--TRIALS=512"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"
_test="--TEST_REPS=4096"

_restart="--RESTART_INTERVAL=8192"
#_sharing="--SHARING_INTERVAL=1024"

_a=0.20
_e=0.00
_g=0.90
_l=0.70

for _sharing in "--SHARING_INTERVAL=64" "--SHARING_INTERVAL=256" "--SHARING_INTERVAL=1024" "--SHARING_INTERVAL=4096" 
do

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=4096
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=8192 --TEST_INTERVAL=512

done
