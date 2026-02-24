#pragma once

#include <memory>

#include "wavefunction.h"

class EllipticGaussian : public WaveFunction {
    /* assumes 3D case */
public:
    EllipticGaussian(double alpha, double beta);
    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles);
    bool hasAnalyticalDerivative() override { return true; }
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles);
};
