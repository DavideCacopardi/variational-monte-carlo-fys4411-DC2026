#!/bin/bash

# Create build-directory (ignores if it already exists)
mkdir -p build_release

# Move into the build-directory
cd build_release

# Run CMake to create a Release Makefile
cmake -DCMAKE_BUILD_TYPE=Release ..

# No need to generate docs twice
# make doc > /dev/null

# Make the Makefile using eight threads
make -j8

# Move and rename the executable to the top-directory
mv vmc ../vmc_release

echo "Release build complete. Run with: ./vmc_release"