#!/bin/bash
#
# Baseline runs for res computer
#
#

_trials="--TRIALS=1024"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"

_test="--TEST_REPS=8192"

_restart="--RESTART_INTERVAL=8192"
_sharing="--SHARING_INTERVAL=1024"


_a=0.90
_e=0.00
_g=0.90
_l=0.70


_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=65536 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=16384 --TEST_INTERVAL=512
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=4096 --TEST_INTERVAL=256
$_location/pole $_common --AGENT_GROUP_SIZE=256 --TIME_STEPS=1024 --TEST_INTERVAL=64

