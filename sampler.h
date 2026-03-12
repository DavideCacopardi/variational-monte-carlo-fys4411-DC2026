#pragma once
#include <memory>
#include <vector>
#include <chrono>

class Sampler {
public:
    Sampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps);


    void sample(bool acceptedStep, class System* system);
    void printOutputToTerminal(class System& system);
    void printOutputToFile(class System& system, std::ofstream& outs);
    void logOutput(System& system, std::ofstream& outs);
    void computeAverages();
    double getEnergy() { return m_energy; }
    double getCovariance(unsigned int param_idx) { return m_covariance[param_idx]; }
    void startStopwatch();
    double stopStopwatch();
    double partialStopwatch();

private:
    unsigned int m_stepNumber = 0;
    unsigned int m_numberOfMetropolisSteps = 0;
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfParameters = 0;
    unsigned int m_numberOfAcceptedSteps = 0;
    double m_energy = 0;
    double m_energySQ = 0;
    double m_variance = 0;
    std::vector<double> m_covariance;
    std::vector<double> m_opO;
    double m_error = 0;
    double m_cumulativeEnergy = 0;
    double m_cumulativeEnergySQ = 0;
    double m_stepLength = 0;
    std::chrono::high_resolution_clock::time_point m_watch_start;
    std::chrono::high_resolution_clock::time_point m_watch_end;
    std::chrono::duration<double> m_elapsedTime;
};
