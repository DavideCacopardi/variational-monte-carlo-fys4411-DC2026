#include <memory>
#include <iostream>
#include <iomanip>
#include <tuple>
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
    const std::string& outfile_name,
    const std::string& logfile_name,
    int seed
) : m_numberOfDimensions(numberOfDimensions)
, m_numberOfParticles(numberOfParticles)
, m_hamiltonianFactory(std::move(hamiltonianFactory))
, m_waveFunctionFactory(std::move(waveFunctionFactory))
, m_numberOfMetropolisSteps(numberOfMetropolisSteps)
, m_numberOfEquilibrationSteps(numberOfEquilibrationSteps)
, m_timeStep(timeStep)
, m_BFGS_tol(BFGS_tol)
, m_outfile(outfile_name)
, m_logfile(logfile_name)
, m_seed(seed) {
    m_rep_a = m_hamiltonianFactory()->getRepulsiveFactor();
}

VMCOptimizer::~VMCOptimizer() {
    m_outfile.close();
    m_logfile.close();
}

double VMCOptimizer::computeMC(const std::vector<double>& params, std::vector<double>& grad) {
    static unsigned int count = 1;
    std::cout << "\rComputing MC #" << count++ << std::flush;

    // rebuild system with current params
    auto rng = std::make_unique<Random>(
        m_seed ? m_seed : std::chrono::system_clock::now().time_since_epoch().count());
    auto particles = setupRandomUniformInitialState(m_numberOfDimensions, m_numberOfParticles, *rng, m_rep_a);
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

    sampler->logOutput(*system, m_logfile);

    return sampler->getEnergy();
}

std::pair<double, double> VMCOptimizer::finalMC(const std::vector<double>& params, unsigned int log2steps, std::fstream* energiesOut) {
    // build system with current params
    unsigned int numberOfMetropolisSteps = pow(2, log2steps);
    auto rng = std::make_unique<Random>(
        m_seed ? m_seed : std::chrono::system_clock::now().time_since_epoch().count());
    auto particles = setupRandomUniformInitialState(m_numberOfDimensions, m_numberOfParticles, *rng, m_rep_a);
    auto waveFun = m_waveFunctionFactory(params);
    auto system = std::make_unique<System>(
        m_hamiltonianFactory(),
        std::move(waveFun),
        std::make_unique<MetropolisHastings>(std::move(rng), true),
        std::move(particles));

    system->runEquilibrationSteps(m_timeStep, m_numberOfEquilibrationSteps);
    *energiesOut << std::scientific << std::setprecision(9);
    auto sampler = system->runMetropolisSteps(m_timeStep, numberOfMetropolisSteps, energiesOut);
    energiesOut->seekg(0);

    // sampler->printOutputToTerminal(*system);

    return std::make_pair(sampler->getEnergy(), sampler->getError());
}

std::vector<double> VMCOptimizer::optimize(std::vector<double> initialParams) {
    // print log header
    m_logfile << "#";
    const unsigned int width = 17;
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
    lib_optimizer.set_maxeval(400);         // max number of evaluations
    lib_optimizer.set_maxtime(3600.0);

    double minEnergy;
    try {
        lib_optimizer.optimize(initialParams, minEnergy);
    }
    catch (const std::runtime_error& e) {
        std::cout << "\nNLopt failed: " << e.what() << std::endl;
        std::cout << "Last energy: " << minEnergy << std::endl;
        std::cout << "Last params: ";
        for (auto p : initialParams) std::cout << p << " ";

        m_outfile << "#  -- ERROR -- ";
        m_outfile << "\n# NLopt failed: " << e.what();
        m_outfile << "\n# Last energy: " << minEnergy;
        m_outfile << "\n# Last params: ";
        for (auto p : initialParams) m_outfile << p << " ";
        m_outfile << std::endl << std::endl;
        // throw;
    }

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
        m_outfile << std::endl;
        std::cout << std::endl;
    }

    return initialParams; // NLopt overwrites this with the optimal params
}

