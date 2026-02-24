#pragma once

#include <memory>

#include "montecarlo.h"


class Metropolis : public MonteCarlo {
public:
    Metropolis(std::unique_ptr<class Random> rng, bool preferAnalytic = true);
    bool step(
            double stepLength,
            class WaveFunction& waveFunction,
            std::vector<std::unique_ptr<class Particle>>& particles);
    bool get_preferAnalytic() override { return m_preferAnalytic; };
    bool hasAnalyticalOption() override { return true; }
private:
    bool m_preferAnalytic = true;
};
