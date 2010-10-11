#!/bin/bash
# Testing script for res computer
#
#  Test restart interavl
#
_trials="--TRIALS=2048"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"
_restart="--RESTART_INTERVAL=1024"
_test="--TEST_REPS=16384"


_a="--ALPHA"
_e="--EPSILON"
_g="--GAMMA"
_l="--LAMBDA"

_parms="$_a=.50 $_e=.00 $_l=.70"


_sharing="--SHARING_INTERVAL=1024"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=524288 --TEST_INTERVAL=8192
$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=8 --TIME_STEPS=65536 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=32 --TIME_STEPS=16384 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=8192 --TEST_INTERVAL=1024


_sharing="--SHARING_INTERVAL=2048"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=8 --TIME_STEPS=65536 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=32 --TIME_STEPS=16384 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=8192 --TEST_INTERVAL=1024


_sharing="--SHARING_INTERVAL=4096"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=8 --TIME_STEPS=65536 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=32 --TIME_STEPS=16384 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=8192 --TEST_INTERVAL=1024


_sharing="--SHARING_INTERVAL=8192"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=8 --TIME_STEPS=65536 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=32 --TIME_STEPS=16384 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=8192 --TEST_INTERVAL=1024


_sharing="--SHARING_INTERVAL=16384"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=131072 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=8 --TIME_STEPS=65536 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=32768 --TEST_INTERVAL=1024
$_location/pole $_common --AGENT_GROUP_SIZE=32 --TIME_STEPS=16384 --TEST_INTERVAL=1024





