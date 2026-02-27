#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using DrWatson
DrWatson.@quickactivate
using WRCircuit
