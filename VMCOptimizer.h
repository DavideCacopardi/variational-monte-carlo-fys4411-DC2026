#pragma once
#include <vector>
#include <fstream>
#include <string>

#include "mcengine.h"

class VMCOptimizer {
public:
    VMCOptimizer(
        MCEngine& engine,
        unsigned int numberOfMetropolisSteps,
        double BFGS_tol,
        std::ofstream* logfile = nullptr,
        std::ofstream* outfile = nullptr,
        std::ofstream* paramsfile = nullptr
    );

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

    MCEngine& m_engine;
    unsigned int m_numberOfMetropolisSteps;
    double m_BFGS_tol;
    std::ofstream* m_logfile;
    std::ofstream* m_outfile;
    std::ofstream* m_paramsfile;
    unsigned int m_mcCount = 0;
};