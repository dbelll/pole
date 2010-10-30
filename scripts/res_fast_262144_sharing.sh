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


for _s in 8192 4096 2048 1024 512 256 128 64 32
do
_sharing="--SHARING_INTERVAL=$_s"

#----------------------
# agent group size = 1
#----------------------
_a=0.05
_e=0.00
_g=0.90
_l=0.70

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=262144 --TEST_INTERVAL=8192


#----------------------
# agent group size = 4
#----------------------
_a=0.05
_e=0.00
_g=0.90
_l=0.80

_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=65536 --TEST_INTERVAL=2048


#----------------------
# agent group size = 16
#----------------------
if (test $_s -le 16384)
then
_a=0.10
_e=0.00
_g=0.90
_l=0.90
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=16384 --TEST_INTERVAL=512
fi

#-----------------------
# agent group size = 64
#-----------------------
if (test $_s -le 4096)
then
_a=0.20
_e=0.00
_g=0.90
_l=0.90
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=4096 --TEST_INTERVAL=128
fi

#------------------------
# agent group size = 256
#------------------------
if (test $_s -le 1024)
then
_a=0.50
_e=0.00
_g=0.90
_l=0.90
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"

$_location/pole $_common --AGENT_GROUP_SIZE=256 --TIME_STEPS=1024 --TEST_INTERVAL=32
fi


#------------------------
# agent group size = 512
#------------------------
if (test $_s -le 512)
then
_a=0.90
_e=0.00
_g=0.90
_l=0.90
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
$_location/pole $_common --AGENT_GROUP_SIZE=512 --TIME_STEPS=512 --TEST_INTERVAL=16
fi


#-------------------------
# agent group size = 1024
#-------------------------
if (test $_s -le 256)
then
_a=0.90
_e=0.00
_g=0.90
_l=0.90
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
$_location/pole $_common --AGENT_GROUP_SIZE=1024 --TIME_STEPS=256 --TEST_INTERVAL=8
fi


#-------------------------
# agent group size = 2048
#-------------------------
if (test $_s -le 128)
then
_a=0.90
_e=0.00
_g=0.90
_l=0.90
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
$_location/pole $_common --AGENT_GROUP_SIZE=2048 --TIME_STEPS=128 --TEST_INTERVAL=4
fi


#-------------------------
# agent group size = 4096
#-------------------------
if (test $_s -le 64)
then
_a=0.90
_e=0.00
_g=0.90
_l=0.90
_parms="--ALPHA=$_a --EPSILON=$_e --GAMMA=$_g --LAMBDA=$_l"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms $_wgtinfo"
$_location/pole $_common --AGENT_GROUP_SIZE=4096 --TIME_STEPS=64 --TEST_INTERVAL=2
fi

done
