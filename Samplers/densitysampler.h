#pragma once
#include <memory>
#include <vector>
#include <chrono>

#include "sampler.h"

class DensitySampler : Sampler {
public:
    DensitySampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        double rMax,
        unsigned int nBins);


    void sample(bool acceptedStep, class System* system, std::ofstream* outfile = nullptr) override;
    void computeAverages();
    const std::vector<double>& getDensityProfile() const { return m_densityProfile; }
    const std::vector<double>& getRadialGrid() const { return m_rGrid; }
    const std::vector<double>& getDensityError() const { return m_densityError; }

private:
    double m_rMax;
    unsigned int m_nBins;
    double m_dr;
    
    std::vector<double> m_histogram;
    std::vector<double> m_densityProfile;
    std::vector<double> m_densityError;
    std::vector<double> m_rGrid;
};
