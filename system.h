#pragma once

#include <memory>
#include <vector>


class System {
public:
    System(
        std::unique_ptr<class Hamiltonian> hamiltonian,
        std::unique_ptr<class WaveFunction> waveFunction,
        std::unique_ptr<class MonteCarlo> solver,
        std::vector<std::unique_ptr<class Particle>> particles);

    unsigned int runEquilibrationSteps(
        double stepParameter,
        unsigned int numberOfEquilibrationSteps);

    std::unique_ptr<class EnergySampler> runMetropolisSteps(
        double stepParameter,
        unsigned int numberOfMetropolisSteps,
        std::ofstream* energiesOut = nullptr);

    std::unique_ptr<class DensitySampler> runMetropolisStepsOnebodyDensity(
        double stepParameter, unsigned int numberOfMetropolisSteps,
        double rMax, unsigned int nBins);

    double computeLocalEnergy();
    double computeParamDerivativeLn(unsigned int param_idx);
    const std::vector<double>& getWaveFunctionParameters();

    const std::vector<std::unique_ptr<class Particle>>& getParticles() const { return m_particles; }

private:
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;

    std::unique_ptr<class Hamiltonian> m_hamiltonian;
    std::unique_ptr<class WaveFunction> m_waveFunction;
    std::unique_ptr<class MonteCarlo> m_solver;
    std::vector<std::unique_ptr<class Particle>> m_particles;
};

