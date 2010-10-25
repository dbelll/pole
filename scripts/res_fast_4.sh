#!/bin/bash
#
# Baseline runs for res computer
#
#

_trials="--TRIALS=256"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"

_test="--TEST_REPS=8192"

_restart="--RESTART_INTERVAL=2048"
_sharing="--SHARING_INTERVAL=2048"

_wgtinfo="--INIT_SHARING_WGT=0.50 --INIT_THETA_MIN=-0.1 --INIT_THETA_MAX=0.1"

_a=0.01
_e=0.00
_g=0.90
_l=0.70

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

#$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=262144 --TEST_INTERVAL=16384

_a=0.05
_e=0.02
_g=0.90
_l=0.80

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

#$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=65536 --TEST_INTERVAL=4096


_a=0.10
_e=0.02
_g=0.90
_l=0.90

for _l in 0.85 .90 .95
do
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"

_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=16384 --TEST_INTERVAL=1024

done
_a=0.10
_e=0.02
_g=0.90
_l=0.90
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"

_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

#$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=4096 --TEST_INTERVAL=256


_a=0.90
_e=0.02
_g=0.90
_l=0.70
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"

_sharing="--SHARING_INTERVAL=1024"

_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
#$_location/pole $_common --AGENT_GROUP_SIZE=256 --TIME_STEPS=1024 --TEST_INTERVAL=64

