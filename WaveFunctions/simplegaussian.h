#pragma once

#include <memory>

#include "wavefunction.h"

class SimpleGaussian : public WaveFunction {
public:
    SimpleGaussian(double alpha);
    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles);
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles);
    double computeParticleLn(std::vector<std::unique_ptr<class Particle>>& particles,
        unsigned int particle_idx);
    bool hasAnalyticalDerivative() override { return true; }
};
 