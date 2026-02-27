#!/usr/bin/env python3
"""Convert CondaPkg.toml to conda environment.yml"""
import sys, tomllib

with open(sys.argv[1] if len(sys.argv) > 1 else "CondaPkg.toml", "rb") as f:
    cfg = tomllib.load(f)

def fmt(name, ver):
    return f"{name}{ver}" if ver else name

deps = [fmt(k, v) for k, v in cfg.get("deps", {}).items()]
pips = [f"{k} @ {v[1:]}" if v.startswith("@") else fmt(k, v)
        for k, v in cfg.get("pip", {}).get("deps", {}).items()]

print("name: env\nchannels:\n  - conda-forge\n  - nvidia\ndependencies:")
for d in deps:
    print(f"  - {d}")
if pips:
    print("  - pip\n  - pip:")
    for p in pips:
        print(f"    - {p}")
