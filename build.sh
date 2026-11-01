#!/usr/bin/env bash
# Build the engine (needs g++ >= 11 and an AVX2+FMA CPU; -march=native picks the host features, e.g. znver3).
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p bin
g++ -O3 -march=native -mtune=native -std=c++20 -fno-semantic-interposition -DNDEBUG -fopenmp engine/gemma4_engine.cpp -I engine -o bin/gemma4_engine
echo "built bin/gemma4_engine"
