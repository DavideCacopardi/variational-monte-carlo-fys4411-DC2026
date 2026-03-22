#pragma once

#include <vector>
#include <memory>
#include "wavefunction.h"

class WaveFunctionCache {
public:
    WaveFunctionCache(WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);

    double computeLnRatio(std::vector<std::unique_ptr<class Particle>>& particles,
        unsigned int particle_idx);

    void acceptMove(unsigned int particle_idx);

    double getTotalLn() const { return m_totalLn; }
    double getParticleLn(unsigned int i) const { return m_particleLn[i]; }

    double computeNumericalLaplacian(
        std::vector<std::unique_ptr<Particle>>& particles,
        WaveFunction& waveFunction);

private:
    WaveFunction& m_wf;
    std::vector<double> m_particleLn;
    double m_totalLn = 0.0;
    double m_pendingLn = 0.0;  // last computeLnRatio
};