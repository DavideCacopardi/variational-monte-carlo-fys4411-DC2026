#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include "mcengine.h"
#include "system.h"
#include "sampler.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "Solvers/metropolishastings.h"
#include "Hamiltonians/hamiltonian.h"

MCEngine::MCEngine(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    unsigned int numberOfEquilibrationSteps,
    double timeStep,
    HamiltonianFactory hamiltonianFactory,
    WaveFunctionFactory waveFunctionFactory,
    int seed
) : m_numberOfDimensions(numberOfDimensions)
, m_numberOfParticles(numberOfParticles)
, m_numberOfEquilibrationSteps(numberOfEquilibrationSteps)
, m_timeStep(timeStep)
, m_hamiltonianFactory(std::move(hamiltonianFactory))
, m_waveFunctionFactory(std::move(waveFunctionFactory))
, m_seed(seed) {
    m_rep_a = m_hamiltonianFactory()->getRepulsiveFactor();
}

std::unique_ptr<Sampler> MCEngine::run(
    const std::vector<double>& params,
    unsigned int numberOfMetropolisSteps,
    std::ofstream* energiesOut) {
    auto rng = std::make_unique<Random>(m_seed == 0
        ? std::chrono::system_clock::now().time_since_epoch().count()
        : m_seed);
    auto particles = setupRandomUniformInitialState(
        m_numberOfDimensions, m_numberOfParticles, *rng, m_rep_a);
    auto system = std::make_unique<System>(
        m_hamiltonianFactory(),
        m_waveFunctionFactory(params),
        std::make_unique<MetropolisHastings>(std::move(rng), true),
        std::move(particles));

    system->runEquilibrationSteps(m_timeStep, m_numberOfEquilibrationSteps);
    return system->runMetropolisSteps(m_timeStep, numberOfMetropolisSteps, energiesOut);
}

double MCEngine::getRepulsiveFactor() const {
    return m_rep_a;
}

std::unique_ptr<WaveFunction> MCEngine::makeWaveFunction(const std::vector<double>& params) const {
    return m_waveFunctionFactory(params);
}