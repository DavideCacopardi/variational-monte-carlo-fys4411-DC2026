#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>
#include <ctime>
#include <cstring>

#include "system.h"
#include "common.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/metropolishastings.h"
#include "Math/random.h"
#include "particle.h"
#include "sampler.h"

using namespace std;

vector<vector<double>> readParameters(ifstream& ins, unsigned int numberOfParameters) {
    vector<vector<double>> params(1);
    
    int idx = 0;
    double temp;
    while (ins >> temp) {
        params.resize(idx + 1);
        params[idx].resize(numberOfParameters);
        params[idx][0] = temp;
        for (unsigned int i = 1; i < numberOfParameters; i++) {
            ins >> params[idx][i];
        }
        idx++;
    }
    return params;
}

int main(int argc, char* argv[]) {
    bool analytical_ifAvailable = true;
    if (argc == 2) {
        if (strcmp(argv[1], "true") == 0) {
            analytical_ifAvailable = true;
        }
        else if (strcmp(argv[1], "false") == 0) {
            analytical_ifAvailable = false;
        }
        else {
            throw(invalid_argument("Invalid boolean value specified."));
        }
    }

    // Seed for the random number generator
    int seed = 2023;

    unsigned int numberOfDimensions = 3;
    unsigned int numberOfParticles = 1;
    unsigned int numberOfParameters = 2;
    unsigned int numberOfMetropolisSteps = (unsigned int) 1e6;
    unsigned int numberOfEquilibrationSteps = (unsigned int) 1e5;
    double omega = 1.0; // Oscillator frequency.

    double stepLength = 0.1; // brute force Metropolis step length.
    double timeStep = 0.05; // Metropolis Hastings timestep.
    double stepParameter = timeStep;
    // double alpha = 0.5; // Variational parameter.
    // double beta = 1; // Variational parameter.
    
    
    ifstream infile("iofiles/input.csv");
    vector<vector<double>> parameters = readParameters(infile, numberOfParameters);
    infile.close();
    ofstream outputFile;
    
    outputFile.open("./iofiles/output.csv");
    for (unsigned int i = 0; i < parameters.size(); i++) {
        // The random engine can also be built without a seed
        auto rng = std::make_unique<Random>(seed);
        // Initialize particles
        auto particles = setupRandomUniformInitialState(stepLength, numberOfDimensions, numberOfParticles, *rng);

        // Construct a unique pointer to wave function
        // auto waveFun = std::make_unique<SimpleGaussian>(parameters[i][0]),
        assert(numberOfDimensions == 3);
        auto waveFun = std::make_unique<EllipticGaussian>(parameters[i][0], parameters[i][1]);

        // Construct a unique pointer to a new System
        auto system = std::make_unique<System>(
            // Construct unique_ptr to Hamiltonian
            std::make_unique<HarmonicOscillator>(omega),
            std::move(waveFun),
            // Construct unique_ptr to solver, and move rng
            // std::make_unique<Metropolis>(std::move(rng), analytical_ifAvailable),
            std::make_unique<MetropolisHastings>(std::move(rng), analytical_ifAvailable),
            // Move the vector of particles to system
            std::move(particles));
        
        // Run steps to equilibrate particles
        auto acceptedEquilibrationSteps = system->runEquilibrationSteps(
            stepParameter,
            numberOfEquilibrationSteps);

        // Run the Metropolis algorithm
        auto sampler = system->runMetropolisSteps(
            stepParameter,
            numberOfMetropolisSteps);

        // Output information from the simulation
        sampler->printOutputToTerminal(*system);
        sampler->printOutputToFile(*system, outputFile);
    }
    outputFile.close();

    return 0;
}
