#!/bin/bash
#
# Baseline runs for res computer
#
#

_trials="--TRIALS=2048"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"

_test="--TEST_REPS=8192"

#_restart="--RESTART_INTERVAL=8192"
_sharing="--SHARING_INTERVAL=1024"

_a=0.90
_e=0.00
_g=0.90
_l=0.70

for _restart in "--RESTART_INTERVAL=2048" "--RESTART_INTERVAL=4096" "--RESTART_INTERVAL=8192"

do

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=4096
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=8192 --TEST_INTERVAL=512

done


for _restart in "--RESTART_INTERVAL=16384" "--RESTART_INTERVAL=32768"
do

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=4096
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=1024
#$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=8192 --TEST_INTERVAL=512

done
