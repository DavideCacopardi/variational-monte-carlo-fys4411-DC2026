#pragma once

#include <memory>

#include "wavefunction.h"

class RepEllipticGaussian : public WaveFunction {
    /* assumes 3D case */
public:
    RepEllipticGaussian(double alpha, double beta, double rep_a);
    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles);
    double evaluateLn(std::vector<std::unique_ptr<class Particle>>& particles);
    double evaluateLn_noInteraction(std::vector<std::unique_ptr<class Particle>>& particles);
    double computeParticleLn(std::vector<std::unique_ptr<Particle>>& particles,
        unsigned int particle_idx);

    bool hasAnalyticalDerivative() override { return false; }
    std::vector<double> lowerBounds() const override { return { 1e-3, 0.1 }; }
    std::vector<double> upperBounds() const override { return { 1.0, 5.0 }; }

private:
    const unsigned int m_NDIM = 3;
    double m_rep_a;
};
