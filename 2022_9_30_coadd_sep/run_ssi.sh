#!/usr/bin/env bash

for tract in 3637; do
  python process_tract.py --tract=${tract}
done
