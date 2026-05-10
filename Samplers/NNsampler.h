#pragma once
#include <memory>
#include <vector>

#include "WaveFunctions/nn_envelope.h"
#include "WaveFunctions/wavefunction.h"

class NNsampler {
public:
    NNsampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        unsigned int numberOfMetropolisSteps,
        WaveFunction& wf_train
    );

    void sample_train(bool acceptedStep, class System* system);
    void sample_pretrain(bool acceptedStep, class System* system);
    void computeAverages();
    void printOutputToTerminal();

    double getEnergy() const;
    double getAcceptanceRatio() const;

    double get_K() const { return m_K; };
    std::vector<double> get_dEdW() const;
    std::vector<double> get_dKdW() const;

private:
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfParameters = 0;
    unsigned int m_numberOfMetropolisSteps = 0;
    WaveFunction& m_wf_train;
    bool m_storeEnergyHistory = false;
    
    unsigned int m_stepNumber = 0;
    unsigned int m_numberOfAcceptedSteps = 0;
    double m_acceptanceRatio = 0;
    double m_energy = 0.0;
    double m_cumulativeEnergy = 0.0;
    double m_B = 0;
    double m_B2 = 0;
    double m_cumulativeB = 0;
    double m_cumulativeB2 = 0;
    double m_K = 0;

    const double c_eps = 1e-12; // against numerical errors

    std::vector<double> m_energyHistory;

    // <O>
    std::vector<double> m_OW;
    std::vector<double> m_cumulativeOW;
    // <E O>
    std::vector<double> m_cumulativeEOW;
    std::vector<double> m_EOW;
    // <B O>
    std::vector<double> m_cumulativeBOW;
    std::vector<double> m_BOW;
    // <B2 O>
    std::vector<double> m_cumulativeB2OW;
    std::vector<double> m_B2OW;
};