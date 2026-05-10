#include <memory>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "system.h"
#include "NNsampler.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/nn_envelope.h"
#include "WaveFunctions/wavefunction.h"

NNsampler::NNsampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    unsigned int numberOfMetropolisSteps,
    WaveFunction& wf_train
) : m_numberOfParticles(numberOfParticles),
m_numberOfDimensions(numberOfDimensions),
m_numberOfParameters(numberOfParameters),
m_numberOfMetropolisSteps(numberOfMetropolisSteps),
m_wf_train(wf_train) {

    if (m_storeEnergyHistory) {
        m_energyHistory.reserve(numberOfMetropolisSteps);
    }

    m_cumulativeOW.assign(m_numberOfParameters, 0);
    m_OW.assign(m_numberOfParameters, 0);
    m_cumulativeAOW.assign(m_numberOfParameters, 0);
    m_AOW.assign(m_numberOfParameters, 0);
    m_cumulativeEOW.assign(m_numberOfParameters, 0);
    m_EOW.assign(m_numberOfParameters, 0);
}

void NNsampler::sample_train(bool acceptedStep, System* system) {
    auto* ptr = dynamic_cast<NN_envelope*>(&system->getWaveFunction());
    if (ptr == nullptr) {
        throw std::logic_error("NNsampler requires a NN_envelope wave function.");
    }

    const double localEnergy = system->computeLocalEnergy();
    if (m_storeEnergyHistory) {
        m_energyHistory.push_back(localEnergy);
    }
    auto OW = system->getWaveFunction().computeLogParDer(system->getParticles());

    m_cumulativeEnergy += localEnergy;

    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        m_cumulativeOW[i] += OW[i];
        m_cumulativeEOW[i] += localEnergy * OW[i];
    }

    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
}

void NNsampler::sample_pretrain(bool acceptedStep, System* system) {
    auto* ptr = dynamic_cast<NN_envelope*>(&system->getWaveFunction());
    if (ptr == nullptr) {
        throw std::logic_error("NNsampler requires a NN_envelope wave function.");
    }

    auto OW = system->getWaveFunction().computeLogParDer(system->getParticles());

    auto& particles = system->getParticles();
    double psi = system->getWaveFunction().evaluate(particles);
    double psi_train = m_wf_train.evaluate(particles);

    double A = psi_train / (psi + c_eps);
    // A = std::max(1e-10, std::min(A, 1e10));
    m_cumulativeA += A;
    m_cumulativeA2 += A * A;
    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        m_cumulativeOW[i] += OW[i];
        m_cumulativeAOW[i] += A * OW[i];
    }

    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
}

void NNsampler::computeAverages() {
    // const double M = static_cast<double>(m_stepNumber);
    m_energy = m_cumulativeEnergy / (double) m_stepNumber;
    m_A = m_cumulativeA / (double) m_stepNumber;
    m_A2 = m_cumulativeA2 / (double)m_stepNumber;
    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        m_OW[i] = m_cumulativeOW[i] / (double) m_stepNumber;
        m_AOW[i] = m_cumulativeAOW[i] / (double) m_stepNumber;
        m_EOW[i] = m_cumulativeEOW[i] / (double) m_stepNumber;
    }

    m_K = m_A * m_A / (m_A2 + c_eps);
}

void NNsampler::printOutputToTerminal() {
    std::cout << std::endl;
    std::cout << "  -- NN Sampler results -- " << std::endl;
    std::cout << " Energy : " << m_energy << std::endl;
    std::cout << " K : " << m_K << std::endl;
    // cout << " dEdW : " << m_energy << endl;
    std::cout << " Acceptance ratio : "
        << (double) m_numberOfAcceptedSteps / (double) m_numberOfMetropolisSteps
        << std::endl;
    std::cout << std::endl;
}

double NNsampler::getEnergy() const { return m_energy; }

double NNsampler::getAcceptanceRatio() const {
    return (double) m_numberOfAcceptedSteps / (double) m_numberOfMetropolisSteps;
}

std::vector<double> NNsampler::get_dEdW() const {
    std::vector<double> dEdW(m_numberOfParameters);
    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        dEdW[i] = 2 * (m_EOW[i] - m_energy * m_OW[i]);
    }
    return dEdW;
}

std::vector<double> NNsampler::get_dKdW() const {
    std::vector<double> dKdW(m_numberOfParameters);
    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        dKdW[i] = 2 * m_K * (m_AOW[i] / (m_A + c_eps) - m_OW[i]);
    }
    return dKdW;
}