#pragma once

#include <vector>
#include <fstream>
#include "WaveFunctions/wavefunction.h"
#include "particle.h"
#include "mcengine.h"
#include "Math/random.h"

std::vector<std::pair<double, double>> computeOnebodyDensity(
    MCEngine& engine,
    const std::vector<double>& params,
    unsigned int numberOfMetropolisSteps,
    double rMax,
    unsigned int nBins,
    std::ofstream* densitiesOut = nullptr);