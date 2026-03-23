#pragma once

#include <memory>
#include <vector>

/**
 * @brief Represents the complete quantum system.
 * * This class acts as a bridge between the Hamiltonian, WaveFunction, 
 * particles, and the Monte Carlo solver. It coordinates the step-by-step
 * sampling process.
 */
class System {
public:
    /**
     * @brief Constructs the system by assembling all physical and numerical components.
     * @param hamiltonian Unique pointer to the Hamiltonian object.
     * @param waveFunction Unique pointer to the WaveFunction object.
     * @param solver Unique pointer to the Monte Carlo engine (e.g., Metropolis).
     * @param particles Vector of unique pointers to the system's particles.
     */
    System(
        std::unique_ptr<class Hamiltonian> hamiltonian,
        std::unique_ptr<class WaveFunction> waveFunction,
        std::unique_ptr<class MonteCarlo> solver,
        std::vector<std::unique_ptr<class Particle>> particles);

    /**
     * @brief Executes equilibration (thermalization) steps.
     * * Moves the particles towards the state of highest probability 
     * without collecting any statistics.
     * @param stepParameter Step length for the solver (\f$\Delta t\f$).
     * @param numberOfEquilibrationSteps Number of steps to take and discard.
     * @return Number of accepted moves during equilibration.
     */
    unsigned int runEquilibrationSteps(
        double stepParameter,
        unsigned int numberOfEquilibrationSteps);

    /**
     * @brief Executes the Metropolis simulation to sample the system's energy.
     * @param stepParameter Step length (\f$\Delta t\f$).
     * @param numberOfMetropolisSteps Total number of steps to execute.
     * @param energiesOut Optional pointer to a file to log individual local energies.
     * @return An EnergySampler containing final statistics (mean energy, variance).
     */
    std::unique_ptr<class EnergySampler> runMetropolisSteps(
        double stepParameter,
        unsigned int numberOfMetropolisSteps,
        std::ofstream* energiesOut = nullptr);

    /**
     * @brief Executes a Metropolis simulation dedicated to density sampling.
     * @param stepParameter Step length (\f$\Delta t\f$).
     * @param numberOfMetropolisSteps Total number of steps to execute.
     * @param rMax Maximum spatial radius covered by the histogram.
     * @param nBins Number of bins for the radial histogram.
     * @return A DensitySampler containing the radial density histogram.
     */
    std::unique_ptr<class DensitySampler> runMetropolisStepsOnebodyDensity(
        double stepParameter, unsigned int numberOfMetropolisSteps,
        double rMax, unsigned int nBins);

    /**
     * @brief Evaluates the local energy \f$E_L = \frac{1}{\Psi} \hat{H} \Psi\f$ for the current configuration.
     * @return The local energy in natural units.
     */
    double computeLocalEnergy();

    /**
     * @brief Calculates the logarithmic derivative of the wave function with respect to a parameter.
     * * Essential for computing the energy gradient during variational optimization.
     * @param param_idx The index of the variational parameter (e.g., 0 for alpha, 1 for beta).
     * @return Value of the logarithmic derivative evaluated at the current position.
     */
    double computeParamDerivativeLn(unsigned int param_idx);

    /**
     * @brief Gets the current variational parameters.
     * @return Vector containing the wave function's parameters.
     */
    const std::vector<double>& getWaveFunctionParameters();

    /**
     * @brief Retrieves the particles currently in the system.
     * @return Constant reference to the vector of particle pointers.
     */
    const std::vector<std::unique_ptr<class Particle>>& getParticles() const { return m_particles; }

private:
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;

    std::unique_ptr<class Hamiltonian> m_hamiltonian;
    std::unique_ptr<class WaveFunction> m_waveFunction;
    std::unique_ptr<class MonteCarlo> m_solver;
    std::vector<std::unique_ptr<class Particle>> m_particles;
};

