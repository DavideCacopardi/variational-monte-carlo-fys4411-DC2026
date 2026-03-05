#pragma once
#include <memory>
#include <ctime>

class Sampler {
public:
    Sampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        double stepLength,
        unsigned int numberOfMetropolisSteps);


    void sample(bool acceptedStep, class System* system);
    void printOutputToTerminal(class System& system);
    void printOutputToFile(class System& system, std::ofstream& outs);
    void computeAverages();
    double getEnergy() { return m_energy; }
    void startStopwatch();
    double stopStopwatch();
    double partialStopwatch();

private:
    unsigned int m_stepNumber = 0;
    unsigned int m_numberOfMetropolisSteps = 0;
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfAcceptedSteps = 0;
    double m_energy = 0;
    double m_energySQ = 0;
    double m_variance = 0;
    double m_error = 0;
    double m_cumulativeEnergy = 0;
    double m_cumulativeEnergySQ = 0;
    double m_stepLength = 0;
    std::time_t m_watch_start = 0;
    std::time_t m_watch_end = 0;
};
