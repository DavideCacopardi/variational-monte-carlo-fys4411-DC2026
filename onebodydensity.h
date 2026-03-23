#pragma once

#include <vector>
#include <fstream>
#include "WaveFunctions/wavefunction.h"
#include "particle.h"
#include "Math/random.h"

std::vector<std::pair<double, double>> computeOnebodyDensity(
    WaveFunction& waveFunction,
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    const std::vector<std::vector<double>>& rGrid,
    double L,
    unsigned int nSteps,
    int seed = 0,
    std::ofstream* out = nullptr
);