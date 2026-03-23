#include <iostream>
#include <memory>
#include <cassert>

#include "system.h"
#include "Samplers/energysampler.h"
#include "Samplers/densitysampler.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Solvers/montecarlo.h"


System::System(
    std::unique_ptr<class Hamiltonian> hamiltonian,
    std::unique_ptr<class WaveFunction> waveFunction,
    std::unique_ptr<class MonteCarlo> solver,
    std::vector<std::unique_ptr<class Particle>> particles) {
    m_numberOfParticles = particles.size();;
    m_numberOfDimensions = particles[0]->getNumberOfDimensions();
    m_hamiltonian = std::move(hamiltonian);
    m_waveFunction = std::move(waveFunction);
    m_solver = std::move(solver);
    m_particles = std::move(particles);
    if (m_solver->hasAnalyticalOption()) {
        m_hamiltonian->set_analytic_ifAvailable(m_solver->get_preferAnalytic());
    }
}


unsigned int System::runEquilibrationSteps(double stepParameter,
    unsigned int numberOfEquilibrationSteps) {
    unsigned int acceptedSteps = 0;

    for (unsigned int i = 0; i < numberOfEquilibrationSteps; i++) {
        acceptedSteps += m_solver->step(stepParameter, *m_waveFunction, m_particles);
    }

    return acceptedSteps;
}

std::unique_ptr<class EnergySampler> System::runMetropolisSteps(double stepParameter,
    unsigned int numberOfMetropolisSteps, std::ofstream* energiesOut) {
    auto sampler = std::make_unique<EnergySampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_waveFunction->getNumberOfParameters(),
        stepParameter,
        numberOfMetropolisSteps);

    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        /* Call solver method to do a single Monte-Carlo step.
         */
        bool acceptedStep = m_solver->step(stepParameter, *m_waveFunction, m_particles);

        // Sample energy
        sampler->sample(acceptedStep, this, energiesOut);
    }

    sampler->computeAverages();

    return sampler;
}

std::unique_ptr<class DensitySampler> System::runMetropolisStepsOnebodyDensity(double stepParameter,
    unsigned int numberOfMetropolisSteps, double rMax, unsigned int nBins) {
    auto sampler = std::make_unique<DensitySampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_waveFunction->getNumberOfParameters(),
        stepParameter,
        numberOfMetropolisSteps,
        rMax,
        nBins);

    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        /* Call solver method to do a single Monte-Carlo step.
         */
        bool acceptedStep = m_solver->step(stepParameter, *m_waveFunction, m_particles);

        // sample 1
        sampler->sample(acceptedStep, this);
    }

    sampler->computeAverages();

    return sampler;
}

double System::computeLocalEnergy() {
    return m_hamiltonian->computeLocalEnergy(
        *m_waveFunction, m_particles, m_solver->getCache());
}

double System::computeParamDerivativeLn(unsigned int param_idx) {
    // Helper function
    return m_waveFunction->computeParamDerivativeLn(m_particles, param_idx);
}

const std::vector<double>& System::getWaveFunctionParameters() {
    // Helper function
    return m_waveFunction->getParameters();
}
