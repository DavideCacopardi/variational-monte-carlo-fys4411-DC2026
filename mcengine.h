#pragma once

#include <memory>
#include <vector>
#include <fstream>
#include <functional>
#include <chrono>

class MCEngine {
public:
    using HamiltonianFactory  = std::function<std::unique_ptr<class Hamiltonian>()>;
    using WaveFunctionFactory = std::function<std::unique_ptr<class WaveFunction>(const std::vector<double>&)>;
    using SolverFactory = std::function<std::unique_ptr<class MonteCarlo>(std::unique_ptr<class Random>, bool)>;

    MCEngine(
        unsigned int numberOfDimensions,
        unsigned int numberOfParticles,
        unsigned int numberOfEquilibrationSteps,
        double timeStep,
        HamiltonianFactory hamiltonianFactory,
        WaveFunctionFactory waveFunctionFactory,
        SolverFactory solverFactory,
        int seed = 0
    );

    std::unique_ptr<class EnergySampler> run(
        const std::vector<double>& params,
        unsigned int numberOfMetropolisSteps,
        std::ofstream* energiesOut = nullptr
    );
    
    std::unique_ptr<class DensitySampler> runOnebodyDensity(
        const std::vector<double>& params,
        unsigned int numberOfMetropolisSteps,
        double rMax, 
        unsigned int nBins);

    double getRepulsiveFactor() const;
    std::unique_ptr<class WaveFunction> makeWaveFunction(const std::vector<double>& params) const;

private:
    unsigned int m_numberOfDimensions;
    unsigned int m_numberOfParticles;
    unsigned int m_numberOfEquilibrationSteps;
    double m_timeStep;
    HamiltonianFactory m_hamiltonianFactory;
    WaveFunctionFactory m_waveFunctionFactory;
    SolverFactory m_solverFactory;
    int m_seed;
    double m_rep_a;
};