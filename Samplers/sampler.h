#pragma once
#include <memory>
#include <vector>
#include <chrono>

class Sampler {
public:
    Sampler(unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps);
    virtual ~Sampler() = default;

    virtual void sample(bool acceptedStep, class System* system, std::ofstream* outfile = nullptr) = 0;
    virtual void computeAverages() = 0;

protected:
    unsigned int m_stepNumber = 0;
    unsigned int m_numberOfMetropolisSteps = 0;
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfParameters = 0;
    unsigned int m_numberOfAcceptedSteps = 0;
    double m_stepLength = 0;
    std::chrono::high_resolution_clock::time_point m_watch_start;
    std::chrono::high_resolution_clock::time_point m_watch_end;
    std::chrono::duration<double> m_elapsedTime;
};
