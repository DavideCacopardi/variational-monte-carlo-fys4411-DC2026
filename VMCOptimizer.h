#pragma once
#include <vector>
#include <functional>
#include <fstream>

class VMCOptimizer {
public:
    using HamiltonianFactory = std::function<std::unique_ptr<class Hamiltonian>()>;
    using WaveFunctionFactory = std::function<std::unique_ptr<class WaveFunction>(const std::vector<double>&)>;
    
    VMCOptimizer(unsigned int numberOfDimensions,
        unsigned int numberOfParticles,
        HamiltonianFactory hamiltonianFactory,
        WaveFunctionFactory waveFunctionFactory,
        unsigned int numberOfMetropolisSteps,
        unsigned int numberOfEquilibrationSteps,
        double timeStep,
        double BFGS_tol,
        int seed, const std::string& outfile_name,
        const std::string& logfile_name);
    ~VMCOptimizer();

    // Run BFGS optimization starting from initial parameters
    std::vector<double> optimize(std::vector<double> initialParams);

private:
    // Runs VMC for given params and returns energy (+ fills grad)
    double computeMC(const std::vector<double>& params, std::vector<double>& grad);

    // Static wrapper required by NLopt C-style callback
    static double nloptObjective(const std::vector<double>& params,
        std::vector<double>& grad,
        void* data) {
        return static_cast<VMCOptimizer*>(data)->computeMC(params, grad);
    }

    // MC settings
    private:
    unsigned int m_numberOfDimensions;
    unsigned int m_numberOfParticles;
    HamiltonianFactory m_hamiltonianFactory;
    WaveFunctionFactory m_waveFunctionFactory;
    unsigned int m_numberOfMetropolisSteps;
    unsigned int m_numberOfEquilibrationSteps;
    double m_timeStep;
    double m_BFGS_tol;
    int m_seed;
    std::ofstream m_outfile;
    std::ofstream m_logfile;
};