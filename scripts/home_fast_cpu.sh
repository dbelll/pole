#!/bin/bash
#
# CPU - only timing and testing
#
#

_location="./bin/darwin/release"

_test="--TEST_REPS=8192"
#_restart="--RESTART_INTERVAL=2048"
_sharing="--SHARING_INTERVAL=16"
_wgtinfo="--INIT_SHARING_WGT=0.50 --INIT_THETA_MIN=-1.0 --INIT_THETA_MAX=0.1"


#--------------
#     CPU
#--------------

_trials="--TRIALS=1"

_a=0.05
_e=0.00
_g=0.90
_l=0.70

_run="--RUN_ON_CPU=1 --RUN_ON_GPU=0"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

#$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=524288 --TEST_INTERVAL=524288

#_trials="--TRIALS=16"

for _a in .05
do

_run="--RUN_ON_CPU=1 --RUN_ON_GPU=1"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=2 --TEST_INTERVAL=1

done
