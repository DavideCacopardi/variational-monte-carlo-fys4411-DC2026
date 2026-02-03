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
    for (int i = 0; i < particles.size(); i++) {
        for (int j = 0; j < particles[i]->getNumberOfDimensions(); j++) {
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
    long double laplacian = 0;
    
    for (int i = 0; i < particles.size(); i++) {
        long double laplacian_ith_term = 0;

        // compute laplacian of phi(r_i)
        double norm_i_sq = 0;   // norm of i-th particle's position squared
        for (int j = 0; j < particles[i]->getNumberOfDimensions(); j++) {
            norm_i_sq += SQ(particles[i]->getPosition()[j]);
        }
        // assumes the first parameter is the alpha value
        laplacian_ith_term = -2 * m_parameters[0] * (1 - 2 * m_parameters[0] * norm_i_sq) * exp(-m_parameters[0] * norm_i_sq);

        // compute productory of phi_j s.t. j != j
        for (int j = 0; j < particles.size(); j++) {
            if (j == i) continue;

            laplacian_ith_term *= evaluate(particles);
        }

        // no other non-zero term appears in the simple case
        laplacian += laplacian_ith_term;
    }

    return laplacian;
}
