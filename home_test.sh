#!/bin/bash
# Testing script for home computer
#
# Baseline tests
#

_trials="--TRIALS=1024"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/darwin/release"
_test="--TEST_REPS=8192"

_restart="--RESTART_INTERVAL=1024"

_a="--ALPHA"
_e="--EPSILON"
_g="--GAMMA"
_l="--LAMBDA"

_parms="$_a=.50 $_e=.00 $_l=.70"


_sharing="--SHARING_INTERVAL=256"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=1 --TIME_STEPS=262144 --TEST_INTERVAL=8192
$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=65536 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=16384 --TEST_INTERVAL=512
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=4096 --TEST_INTERVAL=128


_sharing="--SHARING_INTERVAL=1024"
_common="$_trials $_run $_grpsize $_restart $_test $_sharing $_parms"

$_location/pole $_common --AGENT_GROUP_SIZE=4 --TIME_STEPS=65536 --TEST_INTERVAL=2048
$_location/pole $_common --AGENT_GROUP_SIZE=16 --TIME_STEPS=16384 --TEST_INTERVAL=512
$_location/pole $_common --AGENT_GROUP_SIZE=64 --TIME_STEPS=4096 --TEST_INTERVAL=128





