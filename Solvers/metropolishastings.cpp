#include <memory>
#include <vector>

#include "common.h"
#include "metropolishastings.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/wavefunctioncache.h"
#include "particle.h"
#include "Math/random.h"

using namespace CommonUtils;

MetropolisHastings::MetropolisHastings(std::unique_ptr<class Random> rng, bool preferAnalytic)
    : MonteCarlo(std::move(rng)), m_preferAnalytic(preferAnalytic) {}


/*
bool MetropolisHastings::step(double timeStep, class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles) {
    // Perform the actual Metropolis-Hastings step

    unsigned int particle_idx = m_rng->nextInt(0, particles.size() - 1);
    double wfold = waveFunction.evaluate(particles);
    std::vector<double> qforceold = waveFunction.computeQuantumForce(particles, particle_idx);
    unsigned int numberOfDimensions = particles[particle_idx]->getNumberOfDimensions();
    std::vector<double> displacement(numberOfDimensions);
    // update position
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        displacement[i] = m_rng->nextGaussian(0.0, 1.0) * sqrt(timeStep) + qforceold[i] * timeStep * m_D;
        particles[particle_idx]->adjustPosition(displacement[i], i);
    }
    double wfnew = waveFunction.evaluate(particles);
    std::vector<double> qforcenew = waveFunction.computeQuantumForce(particles, particle_idx);

    // evaluate GreensFunction
    double GreensFunction = 0;
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        GreensFunction += 0.5 * (qforceold[i] + qforcenew[i]) *
            (m_D * timeStep * 0.5 * (qforceold[i] - qforcenew[i]) - displacement[i]);
    }
    GreensFunction = exp(GreensFunction);

    // accept or reject
    bool accepted = m_rng->nextDouble() <= GreensFunction * sq(wfnew / wfold);
    if (!accepted) {
        for (unsigned int i = 0; i < numberOfDimensions; i++) {
            particles[particle_idx]->adjustPosition(-displacement[i], i);
        }
    }

    return accepted;
}
*/

bool MetropolisHastings::step(double timeStep, class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<Particle>>& particles) {
    // Perform the actual Metropolis-Hastings step

    if (!m_cache)
        m_cache = std::make_unique<WaveFunctionCache>(waveFunction, particles);

    unsigned int particle_idx = m_rng->nextInt(0, particles.size() - 1);
    std::vector<double> qforceold = waveFunction.computeQuantumForce(particles, particle_idx);
    unsigned int numberOfDimensions = particles[particle_idx]->getNumberOfDimensions();
    std::vector<double> displacement(numberOfDimensions);

    // update position
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        displacement[i] = m_rng->nextGaussian(0.0, 1.0) * sqrt(timeStep) + qforceold[i] * timeStep * m_D;
        particles[particle_idx]->adjustPosition(displacement[i], i);
    }
    double lnRatio = m_cache->computeLnRatio(particles, particle_idx);
    std::vector<double> qforcenew = waveFunction.computeQuantumForce(particles, particle_idx);

    // evaluate GreensFunction
    double GreensFunction = 0;
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        GreensFunction += 0.5 * (qforceold[i] + qforcenew[i]) *
            (m_D * timeStep * 0.5 * (qforceold[i] - qforcenew[i]) - displacement[i]);
    }
    GreensFunction = exp(GreensFunction);

    // accept or reject
    bool accepted = m_rng->nextDouble() <= GreensFunction * exp(2.0 * lnRatio);
    if (accepted) {
        m_cache->acceptMove(particle_idx);
    }
    else {
        for (unsigned int i = 0; i < numberOfDimensions; i++) {
            particles[particle_idx]->adjustPosition(-displacement[i], i);
        }
    }

    return accepted;
}
