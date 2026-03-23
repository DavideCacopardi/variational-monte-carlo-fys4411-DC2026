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
#include "onebodydensity.h"
#include "Samplers/energysampler.h"
#include "Samplers/densitysampler.h"
#include "VMCOptimizer.h"

using namespace std;
using namespace CommonUtils;

using HamiltonianFactory = function<unique_ptr<class Hamiltonian>()>;
using WaveFunctionFactory = function<unique_ptr<class WaveFunction>(const vector<double>&)>;
using SolverFactory = function<unique_ptr<MonteCarlo>(unique_ptr<Random>, bool)>;

int main(int argc, char* argv[]) {
    // --- Parameters ---
    string hamiltonianType = "RepulsiveHO";
    string waveFunctionType = "RepEllipticGaussian";
    string solverType = "MetropolisHastings";
    unsigned int numberOfDimensions = 3;
    unsigned int numberOfParticles = 3;
    unsigned int numberOfMetropolisSteps = 1e5;
    unsigned int numberOfEquilibrationSteps = 1e5;
    unsigned int finalMClog2steps = 20;
    unsigned int onebodyDensitySteps = 5e6;
    double omega = 1.0;
    double omega_z = 1.0;
    double repulsive_a_factor = 0.33;
    double timeStep = 0.05;
    double onebodyDensity_rMax = 3.5;
    unsigned int onebodyDensity_nBins = 50;
    double BFGS_tol = 1e-4;
    int seed = 0;
    vector<double> initialParams = { 0.75 , 2 };

    chrono::high_resolution_clock::time_point watch_start, watch_end;
    chrono::duration<double> elapsedTime;

    // --- Toggles based on argc, argv ---
    vector<bool> toggles(3, false);
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            int temp = atoi(argv[i]);
            if (0 < temp && temp <= 3) {
                toggles[temp - 1] = true;
            }
        }
    }
    else {
        toggles.assign(3, true);
    }

    // --- Global Log file setup ---
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm* now_tm = localtime(&now_time);
    ostringstream filenameStream;
    filenameStream << "./logs/run_" << put_time(now_tm, "%Y%m%d_%H%M%S") << ".log";
    ofstream globalLog(filenameStream.str());
    if (!globalLog.is_open()) {
        cerr << "Error: unable to generate log file " << filenameStream.str() << endl;
        return 1;
    }
    {
        globalLog << "=========================================\n";
        globalLog << "               VMC RUN LOG               \n";
        globalLog << "=========================================\n";
        globalLog << "Date and time    : " << put_time(now_tm, "%Y-%m-%d %H:%M:%S") << "\n";
        globalLog << "-----------------------------------------\n";
        globalLog << "Hamiltonian      : " << hamiltonianType << "\n";
        globalLog << "WaveFunction     : " << waveFunctionType << "\n";
        globalLog << "Solver           : " << solverType << "\n";
        globalLog << "-----------------------------------------\n";
        globalLog << "dimensions (D)   : " << numberOfDimensions << "\n";
        globalLog << "particles (N)    : " << numberOfParticles << "\n";
        globalLog << "omega            : " << omega << "\n";
        globalLog << "omega_z          : " << omega_z << "\n";
        globalLog << "rep_a_factor     : " << repulsive_a_factor << "\n";
        globalLog << "time Step        : " << timeStep << "\n";
        globalLog << "Equilibr. Steps  : " << numberOfEquilibrationSteps << "\n";
        globalLog << "BFGS MC Steps    : " << numberOfMetropolisSteps << "\n";
        globalLog << "Final MC Steps   : 2^" << finalMClog2steps << "\n";
        globalLog << "1bodyDens. Steps : " << onebodyDensitySteps << "\n";
        globalLog << "1bodyDens. rMax  : " << onebodyDensity_rMax << "\n";
        globalLog << "1bodyDens. nBins : " << onebodyDensity_nBins << "\n";
        globalLog << "BFGS_tol         : " << BFGS_tol << "\n";
        globalLog << "Seed             : " << seed << "\n";
        globalLog << "=========================================\n\n";
        globalLog << flush;
    }

    // --- Setup Factories ---
    HamiltonianFactory hFac = [=]() -> unique_ptr<Hamiltonian> {
        if (hamiltonianType == "HarmonicOscillator")
            return make_unique<HarmonicOscillator>(omega);
        else // default to Repulsive
            return make_unique<RepulsiveHO>(omega, omega_z, repulsive_a_factor);
        };
    WaveFunctionFactory wfFac = [=](const vector<double>& p) -> unique_ptr<WaveFunction> {
        if (waveFunctionType == "SimpleGaussian")
            return make_unique<SimpleGaussian>(p[0]);
        else if (waveFunctionType == "EllipticGaussian")
            return make_unique<EllipticGaussian>(p[0], p[1]);
        else // default to Repulsive
            return make_unique<RepEllipticGaussian>(p[0], p[1], repulsive_a_factor / sqrt(omega));
        };
    SolverFactory solverFac = [=](unique_ptr<Random> rng, bool preferAnalytic) -> unique_ptr<MonteCarlo> {
        if (solverType == "Metropolis") {
            return make_unique<Metropolis>(move(rng), preferAnalytic);
        }
        else // default to Metropolis-Hastings
            return make_unique<MetropolisHastings>(move(rng), preferAnalytic);
        };

    // --- Build MC engine ---
    MCEngine engine(
        numberOfDimensions,
        numberOfParticles,
        numberOfEquilibrationSteps,
        timeStep,
        hFac,
        wfFac,
        solverFac,
        seed
    );

    if (toggles[0]) {
        // --- 1: Optimization ---
        ofstream logfile("./iofiles/log.csv");
        ofstream outfile("./iofiles/details_results.csv");
        ofstream paramsfile("./iofiles/params.dat");
        VMCOptimizer optimizer(engine, numberOfMetropolisSteps, BFGS_tol, &logfile, &outfile, &paramsfile);

        watch_start = chrono::high_resolution_clock::now();
        vector<double> optimalParams = optimizer.optimize(initialParams);
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;

        cout << "Optimal parameters: " << setprecision(9);
        globalLog << "Optimal parameters: " << setprecision(9);
        for (unsigned int i = 0; i < optimalParams.size(); i++) {
            cout << optimalParams[i] << ", \t";
            globalLog << optimalParams[i] << ", \t";
        }
        cout << "\nVMC Optimization done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "\nVMC Optimization done (in " << elapsedTime.count() << " s).\n\n";
        logfile.close(); outfile.close(); paramsfile.close();
    }

    if (toggles[1]) {
        // --- 2a: Final MC ---
        watch_start = chrono::high_resolution_clock::now();
        vector<double> params = readVector("./iofiles/params.dat");
        ofstream energiesfile("./iofiles/finalMCenergies.dat");
        unique_ptr<EnergySampler> sampler =
            engine.run(params, (unsigned int)pow(2, finalMClog2steps), &energiesfile);
        cout << scientific << setprecision(9) << "FinalMC energy: " << sampler->getEnergy()
            << " +- " << sampler->getError() << endl << defaultfloat;
        globalLog << scientific << setprecision(9) << "FinalMC energy: " << sampler->getEnergy()
            << " +- " << sampler->getError() << endl << defaultfloat;
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "FinalMC done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "FinalMC done (in " << elapsedTime.count() << " s).\n\n";
        energiesfile.close();

        // --- 2b: Blocking ---
        watch_start = chrono::high_resolution_clock::now();
        vector<double> finalEnergies = readVector("./iofiles/finalMCenergies.dat");
        Blocker block(finalEnergies);
        block.printResults("./iofiles/blocking_results.csv");
        cout << scientific << setprecision(9) << "Blocking energy: " << block.mean
            << " +- " << block.stdErr << endl << defaultfloat;
        globalLog << scientific << setprecision(9) << "Blocking energy: " << block.mean
            << " +- " << block.stdErr << endl << defaultfloat;
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "Blocking analysis done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "Blocking analysis done (in " << elapsedTime.count() << " s).\n\n";
    }

    if (toggles[2]) {
        // --- 3: One-body density ---
        watch_start = chrono::high_resolution_clock::now();
        vector<double> params = readVector("./iofiles/params.dat");
        vector<vector<double>> rGrid = readMatrix("./iofiles/rGrid.csv");
        ofstream densityfile("./iofiles/onebodydensity.dat");
        vector<pair<double, double>> density = computeOnebodyDensity(
            engine, params, onebodyDensitySteps, onebodyDensity_rMax,
            onebodyDensity_nBins, &densityfile);
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "One-body density done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "One-body density done (in " << elapsedTime.count() << " s).\n\n";
        densityfile.close();
    }
    globalLog << "=========================================\n";
    globalLog.close();

    cout << "Exiting. (Log saved in: " << filenameStream.str() << ")\n";
    return 0;
}