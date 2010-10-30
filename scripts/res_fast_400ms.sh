#!/bin/bash
#
# Baseline runs for res computer
#
#

_trials="--TRIALS=$1"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"
_test="--TEST_REPS=8192"
_restart="--RESTART_INTERVAL=2048"
_wgtinfo="--INIT_SHARING_WGT=0.50 --INIT_THETA_MIN=-0.1 --INIT_THETA_MAX=0.1"


#----------------------
# agent group size = 1
#----------------------
_a=0.05
_e=0.00
_g=0.90
_l=0.70

_sharing="--SHARING_INTERVAL=2048"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=4096 --TEST_INTERVAL=128


#----------------------
# agent group size = 4
#----------------------
_a=0.05
_e=0.00
_g=0.90
_l=0.80

_sharing="--SHARING_INTERVAL=2048"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=4096 --TEST_INTERVAL=128


#----------------------
# agent group size = 16
#----------------------
_a=0.10
_e=0.00
_g=0.90
_l=0.90

_sharing="--SHARING_INTERVAL=2048"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=4096 --TEST_INTERVAL=128


#-----------------------
# agent group size = 64
#-----------------------
_a=0.20
_e=0.00
_g=0.90
_l=0.90

_sharing="--SHARING_INTERVAL=1024"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=4096 --TEST_INTERVAL=128


#------------------------
# agent group size = 256
#------------------------
_a=0.50
_e=0.00
_g=0.90
_l=0.90

_sharing="--SHARING_INTERVAL=256"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=256 --TIME_STEPS=4096 --TEST_INTERVAL=128


#------------------------
# agent group size = 512
#------------------------
_a=0.90
_e=0.00
_g=0.90
_l=0.90

_sharing="--SHARING_INTERVAL=128"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
$_location/pole $_common --AGENT_GROUP_SIZE=512 --TIME_STEPS=4096 --TEST_INTERVAL=128


#-------------------------
# agent group size = 1024
#-------------------------
_a=0.90
_e=0.00
_g=0.90
_l=0.90

_sharing="--SHARING_INTERVAL=128"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
$_location/pole $_common --AGENT_GROUP_SIZE=1024 --TIME_STEPS=4096 --TEST_INTERVAL=128


#-------------------------
# agent group size = 2048
#-------------------------
_a=0.90
_e=0.00
_g=0.90
_l=0.90

_sharing="--SHARING_INTERVAL=128"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
$_location/pole $_common --AGENT_GROUP_SIZE=2048 --TIME_STEPS=4096 --TEST_INTERVAL=128


#-------------------------
# agent group size = 4096
#-------------------------
if (test $1 -le 512)
then
_a=0.90
_e=0.00
_g=0.90
_l=0.90

_sharing="--SHARING_INTERVAL=128"

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
$_location/pole $_common --AGENT_GROUP_SIZE=4096 --TIME_STEPS=4096 --TEST_INTERVAL=128
fi

