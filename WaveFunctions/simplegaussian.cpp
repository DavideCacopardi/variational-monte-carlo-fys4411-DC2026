#include <memory>
#include <cmath>
#include <cassert>

#include "simplegaussian.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

// maybe move elsewhere
#define SQ(x) ((x) * (x))

SimpleGaussian::SimpleGaussian(double alpha)
{
    assert(alpha >= 0);
    m_numberOfParameters = 1;
    m_parameters.reserve(1);
    m_parameters.push_back(alpha);
}

double SimpleGaussian::evaluate(std::vector<std::unique_ptr<class Particle>>& particles) {
    /* You need to implement a Gaussian wave function here. The positions of
     * the particles are accessible through the particle[i]->getPosition()
     * function.
     */
    long double sum = 0;

    // sum all coordinates squared
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = 0; j < particles[i]->getNumberOfDimensions(); j++) {
            sum += SQ(particles[i]->getPosition()[j]);
        }
    }

    // assumes the first parameter is the alpha value
    return exp(-m_parameters[0] * sum);
}

double SimpleGaussian::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) {
    /* All wave functions need to implement this function, so you need to
     * find the double derivative analytically. Note that by double derivative,
     * we actually mean the sum of the Laplacians with respect to the
     * coordinates of each particle.
     *
     * This quantity is needed to compute the (local) energy (consider the
     * Schrödinger equation to see how the two are related).
     */
    
    double alpha = m_parameters[0];
    unsigned int numberOfDimensions = particles[0]->getNumberOfDimensions();

    double sum_over_particles = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        double rad_sq = 0;
        for (unsigned int j = 0; j < numberOfDimensions; j++) {
            rad_sq += SQ(particles[i]->getPosition()[j]);
        }
        
        double phi_i = exp(-alpha * rad_sq);
        double lapl_term = -2 * alpha * (numberOfDimensions - 2 * alpha * rad_sq) * phi_i;

        double prod_term = evaluate(particles) / phi_i;

        sum_over_particles += lapl_term * prod_term;
    }
    return sum_over_particles;
}
