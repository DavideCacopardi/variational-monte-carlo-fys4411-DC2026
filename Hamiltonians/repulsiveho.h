#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

/**
 * @brief Represents an interacting Bose gas in an elliptical harmonic trap.
 * * This Hamiltonian introduces a hard-core repulsive potential between particles
 * (the Jastrow factor physics) and allows for asymmetrical trapping frequencies
 * (omega_z != omega_perp)
 */
class RepulsiveHO : public Hamiltonian {
public:
    /**
     * @brief Constructs a spherical trap with default hard-core repulsion.
     */
    RepulsiveHO(double omega);

    /**
     * @brief Constructs an elliptical trap with default hard-core repulsion.
     */
    RepulsiveHO(double omega, double omega_z);

    /**
     * @brief Constructs an elliptical trap specifying the hard-core radius.
     * @param omega Trap frequency in the xy-plane.
     * @param omega_z Trap frequency along the z-axis.
     * @param repulsive_a_factor The interaction diameter 'a' in standard units.
     */
    RepulsiveHO(double omega, double omega_z, double repulsive_a_factor);

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles,
        class WaveFunctionCache& cache
    ) override;

    double getRepulsiveFactor() const override { return m_rep_a; }

private:
    double m_omega;
    double m_omega_z;
    double m_rep_a;     ///< Scaled hard-core diameter
};

