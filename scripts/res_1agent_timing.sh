#!/bin/bash
#
# 
# Single agent timing runs on CPU and GPU for 1024 trials
#
# 
#

_trials="--TRIALS=$1"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"

_test="--TEST_REPS=8192"

_restart="--RESTART_INTERVAL=2048"
_sharing="--SHARING_INTERVAL=2048"

_wgtinfo="--INIT_SHARING_WGT=0.50 --INIT_THETA_MIN=-0.1 --INIT_THETA_MAX=0.1"

_a=0.05
_e=0.00
_g=0.90
_l=0.70

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=262144 --TEST_INTERVAL=262144



#  CPU run

_run="--RUN_ON_CPU=1 --RUN_ON_GPU=0"

_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=262144 --TEST_INTERVAL=262144
