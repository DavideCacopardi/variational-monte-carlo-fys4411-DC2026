#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

class RepulsiveHO : public Hamiltonian {
public:
    RepulsiveHO(double omega);
    RepulsiveHO(double omega, double omega_z);
    RepulsiveHO(double omega, double omega_z, double repulsive_a_factor);
    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles,
        class WaveFunctionCache& cache
    );
    double getRepulsiveFactor() const override { return m_rep_a; }

private:
    double m_omega;
    double m_omega_z;
    double m_rep_a;
};

