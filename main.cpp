#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>
#include <armadillo>
#include <nlopt.hpp>

#include "system.h"
#include "common.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "WaveFunctions/repulsiveellipticgaussian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "Hamiltonians/repulsiveho.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/metropolishastings.h"
#include "Math/random.h"
#include "Math/blocker.h"
#include "particle.h"
#include "sampler.h"
#include "VMCOptimizer.h"

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

/*
int old_main(int argc, char* argv[]) {
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
*/

int main(int argc, char* argv[]) {
    unsigned int numberOfDimensions = 3;
    unsigned int numberOfParticles = 10;
    unsigned int numberOfMetropolisSteps = (unsigned int)1e5;
    unsigned int numberOfEquilibrationSteps = (unsigned int)1e5;
    double omega = 1.0;
    double omega_z = 1;
    double repulsive_a_factor = 0;
    double timeStep = 0.1;
    double BFGS_tol = 1e-5;
    int seed = 0;
    // if seed==0, then every Random instance is generated with a random seed,
    // else every Random instance is generated with this specified seed
    chrono::high_resolution_clock::time_point watch_start, watch_end;
    chrono::duration<double> elapsedTime;

    // --- Optimization ---
    VMCOptimizer optimizer(
        numberOfDimensions,
        numberOfParticles,
        // Hamiltonian factory — captures omega
        // [omega]() { return std::make_unique<HarmonicOscillator>(omega); },
        [omega, omega_z, repulsive_a_factor]() {
            return make_unique<RepulsiveHO>(omega, omega_z, repulsive_a_factor);
        },
        // WaveFunction factory — receives params from BFGS
        // [](const std::vector<double>& p) { return make_unique<EllipticGaussian>(p[0], p[1]); },
        [omega, omega_z, repulsive_a_factor](const std::vector<double>& p) {
            return make_unique<RepEllipticGaussian>(p[0], p[1], repulsive_a_factor / sqrt(omega));
        },
        numberOfMetropolisSteps,
        numberOfEquilibrationSteps,
        timeStep,
        BFGS_tol,
        "./iofiles/details_results.csv",
        "./iofiles/log.csv",
        seed
    );
    vector<double> initialParams = { 0.75, 1.2 }; // initial alpha, beta
    watch_start = std::chrono::high_resolution_clock::now();
    vector<double> optimalParams = optimizer.optimize(initialParams);
    // print to terminal and to .dat
    cout << "Optimal parameters: " << setprecision(9);
    ofstream outParams("./iofiles/params.dat");
    outParams << scientific << setprecision(9);
    for (unsigned int i = 0; i < optimalParams.size(); i++) {
        cout << optimalParams[i] << ", \t";
        outParams << optimalParams[i] << endl;
    }
    cout << endl; outParams.close();
    watch_end = chrono::high_resolution_clock::now();
    elapsedTime = watch_end - watch_start;
    cout << "\nVMC Optimization done (in " << elapsedTime.count() << " s).\n\n";

    // --- Statistical analysis ---
    // Final MC with 2^19 metropolis steps
    watch_start = chrono::high_resolution_clock::now();
    fstream statfile("./iofiles/finalMCenergies.dat",
        ios::out | ios::in | ios::trunc);
    // you could choose params from somewhere else - in this case, reuse optimalParams
    auto result_finalMC = optimizer.finalMC(optimalParams, 19, &statfile);
    cout << scientific << setprecision(9) << "FinalMC energy: " << result_finalMC.first
        << " +- " << result_finalMC.second << endl << defaultfloat;
    watch_end = chrono::high_resolution_clock::now();
    elapsedTime = watch_end - watch_start;
    cout << "FinalMC done (in " << elapsedTime.count() << " s).\n\n";

    // Read printed energies
    vector<double> finalEnergies;
    string line;
    while (getline(statfile, line)) {
        finalEnergies.push_back(strtod(line.c_str(), nullptr));
    }
    statfile.close();

    // This vector must have a size which is a power of 2.
    watch_start = chrono::high_resolution_clock::now();
    Blocker block(finalEnergies);
    block.printResults("./iofiles/blocking_results.csv");
    cout << scientific << setprecision(9) << "Blocking energy: " << block.mean
        << " +- " << block.stdErr << endl << defaultfloat;
    watch_end = chrono::high_resolution_clock::now();
    elapsedTime = watch_end - watch_start;
    cout << "Blocking analysis done (in " << elapsedTime.count() << " s).\n\n";

    cout << "Exiting.\n";
    return 0;
}