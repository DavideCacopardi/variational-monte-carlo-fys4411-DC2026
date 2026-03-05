#include <memory>
#include <cmath>
#include <cassert>

#include "../common.h"
#include "ellipticgaussian.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

using namespace CommonUtils;

EllipticGaussian::EllipticGaussian(double alpha, double beta) {
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
        for (unsigned int j = 0; j < m_NDIM; j++) {
            if (j == 2)
                sum += m_parameters[1] * sq(particles[i]->getPosition()[j]);
            else
                sum += sq(particles[i]->getPosition()[j]);
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

    double sum_over_particles = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        double rad_sq = 0;       // x² + y² + βz²   (for the linear alpha term)
        double rad_sq2 = 0;      // x² + y² + β²z²  (for the quadratic alpha term)

        for (unsigned int j = 0; j < m_NDIM; j++) {
            if (j == 2) {
                rad_sq += beta * sq(particles[i]->getPosition()[j]);
                rad_sq2 += sq(beta * particles[i]->getPosition()[j]);
            }
            else {
                rad_sq += sq(particles[i]->getPosition()[j]);
                rad_sq2 += sq(particles[i]->getPosition()[j]);
            }
        }

        double phi_i = exp(-alpha * rad_sq);
        double lapl_term = (-2 * alpha * (2 + beta) + 4 * sq(alpha) * rad_sq2) * phi_i;

        double prod_term = evaluate(particles) / phi_i;

        sum_over_particles += lapl_term * prod_term;
    }
    return sum_over_particles;
}

std::vector<double> EllipticGaussian::computeQuantumForce(std::vector<std::unique_ptr<class Particle>>& particles, unsigned int particle_idx) {
    double alpha = m_parameters[0];
    double beta = m_parameters[1];
    std::vector<double> qForce = std::vector<double>(m_NDIM);

    double prod = 1;
    for (int i = 0; i < particles.size(); i++) {
        if (i == particle_idx) continue;
        
        double rad_sq = 0;      // x² + y² + βz² 
        for (unsigned int j = 0; j < m_NDIM; j++) {
            if (j == 2) {
                rad_sq += beta * sq(particles[i]->getPosition()[j]);
            }
            else {
                rad_sq += sq(particles[i]->getPosition()[j]);
            }
        }

        prod *= exp(-alpha * rad_sq);
    }

    for (unsigned int i = 0; i < m_NDIM; i++) {
        double rad_sq = 0;      // x² + y² + βz² 
        for (unsigned int j = 0; j < m_NDIM; j++) {
            if (j == 2) {
                rad_sq += beta * sq(particles[particle_idx]->getPosition()[j]);
            }
            else {
                rad_sq += sq(particles[particle_idx]->getPosition()[j]);
            }
        }

        double deriv = -2 * alpha * particles[particle_idx]->getPosition()[i] * exp(-alpha * rad_sq);
        if (i == 2)
            deriv *= beta;
        
        qForce[i] = prod * deriv;
    }

    return qForce;
}