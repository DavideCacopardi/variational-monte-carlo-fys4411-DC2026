#include <memory>
#include <vector>

#include "metropolis.h"
#include "WaveFunctions/wavefunction.h"
#include "particle.h"
#include "Math/random.h"

// maybe move elsewhere
#define SQ(x) ((x) * (x))

Metropolis::Metropolis(std::unique_ptr<class Random> rng)
    : MonteCarlo(std::move(rng))
{
}


bool Metropolis::step(
        double stepLength,
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles)
{
    /* Perform the actual Metropolis step: Choose a particle at random and
     * change its position by a random amount, and check if the step is
     * accepted by the Metropolis test (compare the wave function evaluated at
     * this new position with the one at the old position).
     */
    unsigned int particle_idx = m_rng->nextInt(0, particles.size() - 1);
    
    double wfold = waveFunction.evaluate(particles);
    unsigned int numberOfDimensions = particles[particle_idx]->getNumberOfDimensions();
    std::vector<double> displacement(numberOfDimensions);
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        displacement[i] = (m_rng->nextDouble() - .5) * stepLength;
        particles[particle_idx]->adjustPosition(displacement[i], i);
    }
    double wfnew = waveFunction.evaluate(particles);

    bool accepted = m_rng->nextDouble() <= SQ(wfnew) / SQ(wfold);
    if (!accepted) {
        for (unsigned int i = 0; i < numberOfDimensions; i++) {
            particles[particle_idx]->adjustPosition(-displacement[i], i);
        }
    }

    return accepted;
}
