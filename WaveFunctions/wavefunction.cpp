#include <memory>
#include <cmath>
#include <cassert>
#include <iostream>

#include "common.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

using namespace CommonUtils;

double WaveFunction::computeNumericalDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) {
    /* Numerical double derivative
     * 
     */

    double sum = 0.0;
    double wfCurrent = evaluate(particles);
    unsigned int numberOfDimensions = particles[0]->getNumberOfDimensions();

    for (auto& particle : particles) {

        for (unsigned int d = 0; d < numberOfDimensions; d++) {
            double h = 1e-4 * std::max(1.0, std::abs(particle->getPosition()[d]));
            double h2 = h * h;

            particle->adjustPosition(h, d);
            double wfPlus = evaluate(particles);

            particle->adjustPosition(-2.0 * h, d);
            double wfMinus = evaluate(particles);

            particle->adjustPosition(h, d);

            sum += (wfPlus - 2.0 * wfCurrent + wfMinus) / h2;
        }
    }

    return sum;
}
