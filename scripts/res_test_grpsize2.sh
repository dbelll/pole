#!/bin/bash
# Testing script for res computer
#
#  Test restart interavl
#

_trials="--TRIALS=1024"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"
_test="--TEST_REPS=16384"

_restart="--RESTART_INTERVAL=1024"

_a="--ALPHA"
_e="--EPSILON"
_g="--GAMMA"
_l="--LAMBDA"

_parms="$_a=.50 $_e=.00 $_l=.70"


_sharing="--SHARING_INTERVAL=64"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=524288 --TEST_INTERVAL=8192
$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=8 --TIME_STEPS=65536 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=512
$_location/pole $_common --AGENT_GROUP_SIZE=32 --TIME_STEPS=16384 --TEST_INTERVAL=256
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=8192 --TEST_INTERVAL=128
$_location/pole $_common --AGENT_GROUP_SIZE=128 --TIME_STEPS=4096 --TEST_INTERVAL=64
$_location/pole $_common --AGENT_GROUP_SIZE=256 --TIME_STEPS=2048 --TEST_INTERVAL=32


_sharing="--SHARING_INTERVAL=128"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=8 --TIME_STEPS=65536 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=512
$_location/pole $_common --AGENT_GROUP_SIZE=32 --TIME_STEPS=16384 --TEST_INTERVAL=256
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=8192 --TEST_INTERVAL=128
$_location/pole $_common --AGENT_GROUP_SIZE=128 --TIME_STEPS=4096 --TEST_INTERVAL=64
$_location/pole $_common --AGENT_GROUP_SIZE=256 --TIME_STEPS=2048 --TEST_INTERVAL=32


