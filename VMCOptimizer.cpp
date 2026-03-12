#include <memory>
#include <iostream>
#include <iomanip>
#include <nlopt.hpp>

#include "VMCOptimizer.h"
#include "system.h"
#include "sampler.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "harmonicoscillator.h"
#include "Solvers/metropolishastings.h"

VMCOptimizer::VMCOptimizer(unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    HamiltonianFactory hamiltonianFactory,
    WaveFunctionFactory waveFunctionFactory,
    unsigned int numberOfMetropolisSteps,
    unsigned int numberOfEquilibrationSteps,
    double timeStep,
    double BFGS_tol,
    int seed,
    const std::string& outfile_name,
    const std::string& logfile_name
) : m_numberOfDimensions(numberOfDimensions)
, m_numberOfParticles(numberOfParticles)
, m_hamiltonianFactory(std::move(hamiltonianFactory))
, m_waveFunctionFactory(std::move(waveFunctionFactory))
, m_numberOfMetropolisSteps(numberOfMetropolisSteps)
, m_numberOfEquilibrationSteps(numberOfEquilibrationSteps)
, m_timeStep(timeStep)
, m_BFGS_tol(BFGS_tol)
, m_seed(seed)
, m_outfile(outfile_name)
, m_logfile(logfile_name) {}

VMCOptimizer::~VMCOptimizer() {
    m_outfile.close();
    m_logfile.close();
}

double VMCOptimizer::computeMC(const std::vector<double>& params, std::vector<double>& grad) {
    static unsigned int count = 1;
    std::cout << "\rComputing MC #" << count++ << std::flush;

    // rebuild system with current params
    auto rng = std::make_unique<Random>(m_seed);      // !!! check this
    auto particles = setupRandomUniformInitialState(m_timeStep, m_numberOfDimensions, m_numberOfParticles, *rng);
    auto waveFun = m_waveFunctionFactory(params);
    auto system = std::make_unique<System>(
        m_hamiltonianFactory(),
        std::move(waveFun),
        std::make_unique<MetropolisHastings>(std::move(rng), true),
        std::move(particles));

    system->runEquilibrationSteps(m_timeStep, m_numberOfEquilibrationSteps);
    auto sampler = system->runMetropolisSteps(m_timeStep, m_numberOfMetropolisSteps);

    // finite difference gradient if grad is requested
    if (!grad.empty()) {
        for (unsigned int i = 0; i < params.size(); i++) {
            grad[i] = 2 * sampler->getCovariance(i);
        }
    }

    // finite difference gradient if grad is requested
    // if (!grad.empty()) {
    //     double h = 1e-4;
    //     for (unsigned int i = 0; i < params.size(); i++) {
    //         std::vector<double> paramsPlus = params;
    //         std::vector<double> paramsMinus = params;
    //         paramsPlus[i] += h;
    //         paramsMinus[i] -= h;
    //         std::vector<double> dummyGrad; // empty = no grad needed
    //         double ePlus  = computeMC(paramsPlus,  dummyGrad);
    //         double eMinus = computeMC(paramsMinus, dummyGrad);
    //         grad[i] = (ePlus - eMinus) / (2 * h);
    //     }
    // }

    // sampler->printOutputToTerminal(*system);
    sampler->logOutput(*system, m_logfile);

    return sampler->getEnergy();
}

std::vector<double> VMCOptimizer::optimize(std::vector<double> initialParams) {
    // print log header
    m_logfile << "#";
    unsigned int width = 17;
    for (unsigned int i = 0; i < initialParams.size(); i++) {
        std::string temp = "p[" + std::to_string(i) + "],";
        m_logfile << std::setw(width - (i == 0)) << temp;
    }
    m_logfile << std::setw(width) << "energy," << std::setw(width) << "variance,"
        << std::setw(width) << "error," << std::setw(width) << "elapsed time,"
        << std::setw(width) << "acceptance ratio" << std::endl;

    nlopt::opt lib_optimizer(nlopt::LD_LBFGS, initialParams.size());
    {
        auto tempWaveFunction = m_waveFunctionFactory(initialParams);
        auto lb = tempWaveFunction->lowerBounds();
        auto ub = tempWaveFunction->upperBounds();
        if (!lb.empty()) lib_optimizer.set_lower_bounds(lb);
        if (!ub.empty()) lib_optimizer.set_upper_bounds(ub);
    }
    lib_optimizer.set_min_objective(nloptObjective, this);
    lib_optimizer.set_xtol_rel(m_BFGS_tol);

    double minEnergy;
    lib_optimizer.optimize(initialParams, minEnergy);

    // print details and results
    {
        m_outfile << "#  -- System info -- "
            << "\n# Number of particles  : " << m_numberOfParticles
            << "\n# Number of dimensions : " << m_numberOfDimensions
            << "\n# Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps)
            << "\n# Time step used : " << m_timeStep
            << "\n# BFGS tolerance used : " << m_BFGS_tol
            << "\n\n# Optimal energy: " << minEnergy
            << "\n# Optimal parameters: " << std::setprecision(8);
        for (unsigned int i = 0; i < initialParams.size(); i++) {
            m_outfile << initialParams[i] << ", \t";
        }
    }

    return initialParams; // NLopt overwrites this with the optimal params
}

