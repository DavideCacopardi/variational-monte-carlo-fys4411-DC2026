#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "onebodydensity.h"
#include "common.h"
#include "Math/random.h"

using namespace CommonUtils;

std::vector<std::pair<double, double>> computeOnebodyDensity(
    WaveFunction& waveFunction,
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    const std::vector<std::vector<double>>& rGrid,
    double L,
    unsigned int nSteps,
    int seed,
    std::ofstream* out) {
    // seed
    Random rng((seed == 0)
        ? std::chrono::system_clock::now().time_since_epoch().count()
        : seed);

    // build particles — particle 0 will be fixed, others sampled uniformly
    std::vector<std::vector<double>> positions(numberOfParticles,
        std::vector<double>(numberOfDimensions, 0.0));

    // build Particle objects
    std::vector<std::unique_ptr<Particle>> particles;
    for (unsigned int p = 0; p < numberOfParticles; p++)
        particles.push_back(std::make_unique<Particle>(positions[p]));

    // calc normalization
    double totd = 0;
    double volume = pow(2.0 * L, numberOfParticles * numberOfDimensions);
    for (unsigned int step = 0; step < nSteps; step++) {
        // sample particles 0..N-1 uniformly in [-L, L]
        for (unsigned int p = 0; p < numberOfParticles; p++)
        for (unsigned int d = 0; d < numberOfDimensions; d++)
        particles[p]->setPosition((rng.nextDouble() * 2 - 1) * L, d);
        
        totd += exp(2.0 * waveFunction.evaluateLn(particles));
        // totd += sq(waveFunction.evaluate(particles));
    }
    totd *= volume / (double) nSteps; 
    
    unsigned long int gridSize = rGrid.size();
    std::vector<std::pair<double, double>> density(gridSize);
    std::vector<double> densSamples(nSteps);
    
    volume = pow(2.0 * L, (numberOfParticles - 1) * numberOfDimensions);
    for (unsigned int r_idx = 0; r_idx < gridSize; r_idx++) {
        std::cout << "\rComputing one-body density for r_"
        << r_idx + 1 << " out of " << gridSize << std::flush;
        // fix particle 0 at r
        for (unsigned int d = 0; d < numberOfDimensions; d++)
        particles[0]->setPosition(rGrid[r_idx][d], d);
        
        for (unsigned int step = 0; step < nSteps; step++) {
            // sample particles 1..N-1 uniformly in [-L, L]
            for (unsigned int p = 1; p < numberOfParticles; p++)
            for (unsigned int d = 0; d < numberOfDimensions; d++)
            particles[p]->setPosition((rng.nextDouble() * 2 - 1) * L, d);
            
            densSamples[step] = exp(2.0 * waveFunction.evaluateLn(particles)) / totd * numberOfParticles * volume;
            // densSamples[step] = sq(waveFunction.evaluate(particles)) / totd * numberOfParticles * volume;
        }
        
        density[r_idx] = mean_err(densSamples);

        if (out) {
            *out << std::scientific << std::setprecision(9) << density[r_idx].first
                << ", " << density[r_idx].second << std::endl;
        }
    }

    std::cout << std::endl;
    return density;
}