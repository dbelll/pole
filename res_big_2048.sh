#!/bin/bash
#
# Baseline runs for res computer, big group sizes
#
#


_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"

_test="--TEST_REPS=8192"

_restart="--RESTART_INTERVAL=2048"

_sharing="--SHARING_INTERVAL=1024"

_wgtinfo="--INIT_SHARING_WGT=0.50 --INIT_THETA_MIN=-0.1 --INIT_THETA_MAX=0.1"


_a=0.90
_e=0.00
_g=0.90
_l=0.70

for _a in .2 .5 .7 .9
do

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"


#------------------------
#    by learning time
#------------------------
_trials="--TRIALS=1"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=2048 --TIME_STEPS=32768 --TEST_INTERVAL=512


#---------------------------------------------------
#    repeat by learning time for multiple trials
#---------------------------------------------------
_trials="--TRIALS=16"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=2048 --TIME_STEPS=32768 --TEST_INTERVAL=512


done