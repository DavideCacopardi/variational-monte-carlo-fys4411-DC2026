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

std::vector<double> readVector(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<double> vec;
    double temp;
    while (file >> temp)
        vec.push_back(temp);
    file.close();
    return vec;
}

int main(int argc, char* argv[]) {
    // --- Parameters ---
    unsigned int numberOfDimensions = 3;
    unsigned int numberOfParticles = 3;
    unsigned int numberOfMetropolisSteps = (unsigned int)1e5;
    unsigned int numberOfEquilibrationSteps = (unsigned int)1e5;
    unsigned int finalMClog2steps = 19;
    double omega = 1.0;
    double omega_z = 1.0;
    double repulsive_a_factor = 0.0;
    double timeStep = 0.1;
    double BFGS_tol = 1e-2;
    int seed = 0;

    vector<double> initialParams = { 0.75, 1.2 };

    chrono::high_resolution_clock::time_point watch_start, watch_end;
    chrono::duration<double> elapsedTime;

    // --- Build MC engine ---
    MCEngine engine(
        numberOfDimensions,
        numberOfParticles,
        numberOfEquilibrationSteps,
        timeStep,
        [omega, omega_z, repulsive_a_factor]() {
            return make_unique<RepulsiveHO>(omega, omega_z, repulsive_a_factor);
        },
        [omega, repulsive_a_factor](const std::vector<double>& p) {
            return make_unique<RepEllipticGaussian>(p[0], p[1], repulsive_a_factor / sqrt(omega));
        },
        seed
    );

    // --- Optimization ---
    ofstream logfile("./iofiles/log.csv");
    ofstream outfile("./iofiles/details_results.csv");
    ofstream paramsfile("./iofiles/params.dat");
    VMCOptimizer optimizer(engine, numberOfMetropolisSteps, BFGS_tol, &logfile, &outfile, &paramsfile);

    watch_start = chrono::high_resolution_clock::now();
    vector<double> optimalParams = optimizer.optimize(initialParams);
    watch_end = chrono::high_resolution_clock::now();
    elapsedTime = watch_end - watch_start;

    cout << "Optimal parameters: " << setprecision(9);
    for (unsigned int i = 0; i < optimalParams.size(); i++) {
        cout << optimalParams[i] << ", \t";
    }
    cout << "\nVMC Optimization done (in " << elapsedTime.count() << " s).\n\n";
    logfile.close(); outfile.close(); paramsfile.close();

    // --- Final MC ---
    watch_start = chrono::high_resolution_clock::now();
    vector<double> params = readVector("./iofiles/params.dat");
    ofstream energiesfile("./iofiles/finalMCenergies.dat");
    unique_ptr<Sampler> sampler = engine.run(params, (unsigned int)pow(2, finalMClog2steps), &energiesfile);
    cout << scientific << setprecision(9) << "FinalMC energy: " << sampler->getEnergy()
    << " +- " << sampler->getError() << endl << defaultfloat;
    watch_end = chrono::high_resolution_clock::now();
    elapsedTime = watch_end - watch_start;
    cout << "FinalMC done (in " << elapsedTime.count() << " s).\n\n";
    energiesfile.close();

    // --- Blocking ---
    watch_start = chrono::high_resolution_clock::now();
    vector<double> finalEnergies = readVector("./iofiles/finalMCenergies.dat");
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