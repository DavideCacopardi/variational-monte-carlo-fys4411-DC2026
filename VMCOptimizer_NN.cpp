#include <iostream>
#include <iomanip>
#include <nlopt.hpp>

#include "VMCOptimizer_NN.h"
#include "system.h"
#include "Samplers/energysampler.h"
#include "Samplers/NNsampler.h"
#include "Hamiltonians/repulsiveho.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "WaveFunctions/repulsiveellipticgaussian.h"
#include "WaveFunctions/nn_envelope.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "Solvers/metropolishastings.h"

VMCOptimizer_NN::VMCOptimizer_NN(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    unsigned int numberOfEquilibrationSteps,
    double timeStep,
    std::unique_ptr<RepulsiveHO> hamiltonian,
    SolverFactory solverFactory,
    int seed,
    int Nhid,
    int optEquil,
    int nSamples,
    int nPretrainSteps,
    int nEnergySteps,
    double strengthRate,
    double lr,
    double Adam_tol,
    std::ofstream* logfile,
    std::ofstream* outfile,
    std::ofstream* paramsfile
) : m_numberOfDimensions(numberOfDimensions),
m_numberOfParticles(numberOfParticles),
m_numberOfEquilibrationSteps(numberOfEquilibrationSteps),
m_timeStep(timeStep),
m_hamiltonian(std::move(hamiltonian)),
m_solverFactory(solverFactory),
m_seed(seed),
m_Nhid(Nhid),
m_optEquil(optEquil),
m_nSamples(nSamples),
m_nPretrainSteps(nPretrainSteps),
m_nEnergySteps(nEnergySteps),
m_strengthRate(strengthRate),
m_lr(lr),
m_Adam_tol(Adam_tol),
m_logfile(logfile),
m_outfile(outfile),
m_paramsfile(paramsfile) {}

std::vector<double> VMCOptimizer_NN::optimize(std::unique_ptr<WaveFunction> wf_train) {
    // print log header
    // if (m_logfile) {
    //     *m_logfile << "#";
    //     const unsigned int width = 17;
    //     *m_logfile << std::setw(width) << "energy," << std::setw(width) << "variance,"
    //         << std::setw(width) << "error," << std::setw(width) << "elapsed time,"
    //         << std::setw(width) << "acceptance ratio" << std::endl;
    // }

    auto rng = std::make_unique<Random>(m_seed == 0
        ? std::chrono::system_clock::now().time_since_epoch().count()
        : m_seed);

    auto wf_nn = std::make_unique<NN_envelope>(
        m_numberOfParticles, m_numberOfDimensions, m_numberOfParticles * m_numberOfDimensions, m_Nhid
    );
    NeuralNetwork* nnptr = &(wf_nn->net());
    torch::optim::Adam optimizer(
        nnptr->parameters(),
        torch::optim::AdamOptions(m_lr).betas({ 0.9, 0.999 }).eps(1e-6)
    );
    m_hamiltonian->set_hardcore_strength(0);
    auto system = std::make_unique<System>(
        std::move(m_hamiltonian),
        std::move(wf_train)
    );
    std::cout << "\rDEBUG: Created NN system\n";

    // -------- PRE-TRAINING --------
    std::cout << "Running Adam optimization without interactions...\n";

    system->setParticles(
        setupRandomUniformInitialState(m_numberOfDimensions, m_numberOfParticles, *rng)
    );
    system->setSolver(m_solverFactory(std::move(rng)));
    system->runEquilibrationSteps(m_timeStep, m_optEquil);
    // save positions
    std::vector<std::vector<double>> saved_pos = std::vector<std::vector<double>>(m_numberOfParticles);
    for (unsigned i = 0; i < m_numberOfParticles; i++)
        for (unsigned j = 0; j < m_numberOfDimensions; j++)
            saved_pos[i].push_back(system->getParticles()[i]->getPosition()[j]);
    auto tempsampler = system->runMetropolisSteps(m_timeStep, m_nSamples * 10);
    std::cout << "DEBUG: tempsampler energy = " << tempsampler->getEnergy() << std::endl;

    wf_train = std::move(system->setWaveFunction(std::move(wf_nn)));

    for (int step = 0; step < m_nPretrainSteps; step++) {
        // if (step % 5 == 0) {
            // system->runEquilibrationSteps(m_timeStep, m_optEquil);
            // auto tempsampler = system->runMetropolisSteps_NN(m_timeStep, m_nSamples, *wf_train);
            // std::cout << "DEBUG: step " << step << " tempsampler energy = " << tempsampler->getEnergy() << std::endl;
            // // reset positions
            // for (unsigned i = 0; i < m_numberOfParticles; i++)
            //     for (unsigned j = 0; j < m_numberOfDimensions; j++)
            //         system->getParticles()[i]->setPosition(saved_pos[i][j], j);
        // }

        // system->runEquilibrationSteps(m_timeStep, 50);
        std::unique_ptr<NNsampler> sampler =
            system->runMetropolisSteps_NN_pretrain(m_timeStep, m_nSamples, *wf_train);
        std::vector<double> dKdW = sampler->get_dKdW();
        // Adam minimizes, but we want to maximize K
        for (double& g : dKdW) {
            g = -g;
        }

        optimizer.zero_grad();
        nnptr->setGrads(dKdW);
        torch::nn::utils::clip_grad_norm_(nnptr->parameters(), 10);   // for stability
        optimizer.step();

        if (step % 1 == 0) {
            std::cout << "  step " << step
                << "  K = " << std::scientific << std::setprecision(9) << sampler->get_K()
                << "\t  acc_ratio = " << std::setprecision(9) << sampler->getAcceptanceRatio() << "\n";
        }
    }
    std::cout << std::endl;

    // -------- TRAINING --------
    std::cout << "Running Adam optimization with interactions...\n";

    system->runEquilibrationSteps(m_timeStep, m_optEquil);
    for (int step = 0; step < m_nEnergySteps; step++) {
        system->getHamiltonian().set_hardcore_strength(step * m_strengthRate);
        system->runEquilibrationSteps(m_timeStep, 50);
        auto sampler = system->runMetropolisSteps_NN(m_timeStep, m_nSamples, *wf_train);
        auto dEdW = sampler->get_dEdW();

        optimizer.zero_grad();
        nnptr->setGrads(dEdW);
        torch::nn::utils::clip_grad_norm_(nnptr->parameters(), 10);
        optimizer.step();

        if (step % 1 == 0) {
            std::cout << "  step " << step
                << "  E = " << sampler->getEnergy()
                << "\t  acc_ratio = " << std::setprecision(9) << sampler->getAcceptanceRatio() << "\n";
        }
    }

    return nnptr->getParams();
}