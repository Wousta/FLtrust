#!/bin/bash
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

rm -rf build
mkdir build
cd build
export PATH="/home/bustaman/.local/bin:$PATH"
conan install .. --output-folder=. --build=missing
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
