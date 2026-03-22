#pragma once

#include <memory>

#include "montecarlo.h"
#include "../WaveFunctions/wavefunctioncache.h"

class MetropolisHastings : public MonteCarlo {
public:
    MetropolisHastings(std::unique_ptr<class Random> rng, bool preferAnalytic = true);
    bool step(double timeStep, class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);
    bool get_preferAnalytic() override { return m_preferAnalytic; };
    bool hasAnalyticalOption() override { return true; }
private:
    bool m_preferAnalytic = true;
    const double m_D = 0.5;
};
