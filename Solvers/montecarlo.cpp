#include "montecarlo.h"
#include "Math/random.h"


MonteCarlo::MonteCarlo(std::unique_ptr<class Random> rng, bool preferAnalytic) {
    m_rng = std::move(rng);
}
