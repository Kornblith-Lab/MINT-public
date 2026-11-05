#!/usr/bin/env scons
"""SCons entry point. Each experiment's sconscript builds its own graph under
`exp_<name>/_output/`. Add experiments by appending to EXPERIMENTS.
"""
import os
import SCons.Script as sc

sc.AddOption("--output", type="string", help="output folder name", default="_output")
sc.AddOption("--slurm", action="store_true", help="run via SLURM srun", default=False)

env = sc.Environment(
    ENV=os.environ,
    output=sc.GetOption("output"),
    use_slurm=sc.GetOption("slurm"),
)
sc.Export("env")
env.SConsignFile()

EXPERIMENTS = [
    "exp_demo",
]
for flag in EXPERIMENTS:
    sc.SConscript(flag + "/sconscript", exports=["flag"])
