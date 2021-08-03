#!/usr/bin/env bash

for tract in 3078 3085 3261 3268 3447 3454 3637 3825 3832 4023 4030 4225 4232; do
  python process_tract.py --tract=${tract}
done
