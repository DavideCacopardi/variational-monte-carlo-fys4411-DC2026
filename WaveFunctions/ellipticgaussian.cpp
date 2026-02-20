#include <memory>
#include <cmath>
#include <cassert>

#include "ellipticgaussian.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

// maybe move elsewhere
#define SQ(x) ((x) * (x))

EllipticGaussian::EllipticGaussian(double alpha, double beta)
{
    assert(alpha >= 0 && beta >= 0);
    m_numberOfParameters = 2;
    m_parameters.reserve(2);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
}

double EllipticGaussian::evaluate(std::vector<std::unique_ptr<class Particle>>& particles) {
    /* You need to implement a Gaussian wave function here. The positions of
     * the particles are accessible through the particle[i]->getPosition()
     * function.
     */
    long double sum = 0;

    // sum all coordinates squared
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = 0; j < particles[i]->getNumberOfDimensions(); j++) {
            sum += SQ(particles[i]->getPosition()[j]);
            if (j == 2)
                sum += m_parameters[1] * SQ(particles[i]->getPosition()[j]);
            else
                sum += SQ(particles[i]->getPosition()[j]);
        }
    }

    // assumes the first parameter is the alpha value
    return exp(-m_parameters[0] * sum);
}

double EllipticGaussian::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) {
    /* All wave functions need to implement this function, so you need to
     * find the double derivative analytically. Note that by double derivative,
     * we actually mean the sum of the Laplacians with respect to the
     * coordinates of each particle.
     *
     * This quantity is needed to compute the (local) energy (consider the
     * Schrödinger equation to see how the two are related).
     */
    
    double alpha = m_parameters[0];
    double beta = m_parameters[1];
    unsigned int numberOfDimensions = particles[0]->getNumberOfDimensions();

    double sum_over_particles = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        double rad_sq = 0;
        for (unsigned int j = 0; j < numberOfDimensions; j++) {
            if (j == 2)
                rad_sq += beta * SQ(particles[i]->getPosition()[j]);
            else
                rad_sq += SQ(particles[i]->getPosition()[j]);        
        }
        
        double phi_i = exp(-alpha * rad_sq);
        double lapl_term = -2 * alpha * (2 + beta - 2 * alpha * rad_sq) * phi_i;

        double prod_term = evaluate(particles) / phi_i;

        sum_over_particles += lapl_term * prod_term;
    }
    return sum_over_particles;
}
