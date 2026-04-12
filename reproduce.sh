#!/bin/bash
set -e
echo "Reproducing all results..."

echo "=== Part 1: EDA ==="
cd part1 && python3 eda.py && cd ..

echo "=== Part 1: Interpretability ==="
cd part1 && python3 interpretability.py && cd ..

echo "=== Part 2: Compression ==="
cd part2 && python3 compress.py && cd ..

echo "=== Part 2: Benchmark table ==="
cd part2 && python3 benchmark.py && cd ..

echo "=== Part 3: Stress tests ==="
cd part3 && python3 stress_test.py && cd ..

echo "Done. All figures saved to part*/figures/"
