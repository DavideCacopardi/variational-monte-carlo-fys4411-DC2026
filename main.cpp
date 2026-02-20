#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>

#include "system.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Math/random.h"
#include "particle.h"
#include "sampler.h"

using namespace std;

vector<vector<double>> readParameters(ifstream& ins, unsigned int numberOfParameters) {
    vector<vector<double>> params(1);
    int idx = 0;
    while (!ins.eof()) {
        params[idx].resize(numberOfParameters);
        for (unsigned int i = 0; i < numberOfParameters; i++) {
            ins >> params[idx][i];
        }
        idx++;
        params.resize(idx + 1);
    }
    return params;
}

int main() {
    // Seed for the random number generator
    int seed = 2023;

    unsigned int numberOfDimensions = 3;
    unsigned int numberOfParticles = 1;
    unsigned int numberOfParameters = 2;
    unsigned int numberOfMetropolisSteps = (unsigned int) 1e6;
    unsigned int numberOfEquilibrationSteps = (unsigned int) 1e5;
    double omega = 1.0; // Oscillator frequency.
    double stepLength = 0.1; // Metropolis step length.
    // double alpha = 0.5; // Variational parameter.
    // double beta = 1; // Variational parameter.

    ifstream infile("iofiles/input.csv");
    vector<vector<double>> parameters = readParameters(infile, numberOfParameters);
    infile.close();
    
    ofstream outputFile("./iofiles/output.csv");
    for (unsigned int i = 0; i < parameters.size(); i++) {
        // The random engine can also be built without a seed
        auto rng = std::make_unique<Random>(seed);
        // Initialize particles
        auto particles = setupRandomUniformInitialState(stepLength, numberOfDimensions, numberOfParticles, *rng);
        // Construct a unique pointer to a new System
        // assert(numberOfDimensions == 3);
        auto system = std::make_unique<System>(
            // Construct unique_ptr to Hamiltonian
            std::make_unique<HarmonicOscillator>(omega),
            // Construct unique_ptr to wave function
            // std::make_unique<SimpleGaussian>(parameters[i][0]),
            std::make_unique<EllipticGaussian>(parameters[i][0], parameters[i][1]),
            // Construct unique_ptr to solver, and move rng
            std::make_unique<Metropolis>(std::move(rng)),
            // Move the vector of particles to system
            std::move(particles));

        // Run steps to equilibrate particles
        auto acceptedEquilibrationSteps = system->runEquilibrationSteps(
            stepLength,
            numberOfEquilibrationSteps);

        // Run the Metropolis algorithm
        auto sampler = system->runMetropolisSteps(
            stepLength,
            numberOfMetropolisSteps);

        // Output information from the simulation
        sampler->printOutputToTerminal(*system);
        sampler->printOutputToFile(*system, outputFile);
    }
    outputFile.close();

    return 0;
}
