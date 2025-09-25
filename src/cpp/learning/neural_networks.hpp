#pragma once

#include "../common/agent.hpp"
#include <vector>
#include <random>
#include <functional>
#include <memory>

namespace iapv {
namespace learning {

// Simple neural network implementation
class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers);
    
    std::vector<float> feedForward(const std::vector<float>& inputs);
    void backpropagate(const std::vector<float>& inputs, const std::vector<float>& targets, float learningRate = 0.01f);
    
    void mutate(float mutationRate = 0.1f, float mutationStrength = 0.5f);
    NeuralNetwork crossover(const NeuralNetwork& other) const;
    
    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    std::vector<std::vector<std::vector<float>>> weights_;
    std::vector<std::vector<float>> biases_;
    std::vector<int> layers_;
    
    float sigmoid(float x) const;
    float sigmoidDerivative(float x) const;
    float randomWeight() const;
};

// Q-Learning implementation
class QLearning {
public:
    QLearning(int numStates, int numActions, float learningRate = 0.1f, float discountFactor = 0.9f, float explorationRate = 0.1f);
    
    int selectAction(int state);
    void updateQ(int state, int action, float reward, int nextState);
    float getQValue(int state, int action) const;
    
    void setExplorationRate(float rate) { explorationRate_ = rate; }
    float getExplorationRate() const { return explorationRate_; }

private:
    std::vector<std::vector<float>> qTable_;
    int numStates_;
    int numActions_;
    float learningRate_;
    float discountFactor_;
    float explorationRate_;
    
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;
};

// Experience replay buffer for deep Q-learning
struct Experience {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> nextState;
    bool terminal;
};

class ExperienceReplay {
public:
    ExperienceReplay(size_t capacity) : capacity_(capacity) {}
    
    void add(const Experience& experience);
    std::vector<Experience> sample(size_t batchSize);
    size_t size() const { return experiences_.size(); }

private:
    std::vector<Experience> experiences_;
    size_t capacity_;
    size_t currentIndex_ = 0;
    std::mt19937 rng_;
};

// Genetic Algorithm for evolving neural networks
class GeneticAlgorithm {
public:
    struct Individual {
        NeuralNetwork network;
        float fitness;
        
        Individual(const std::vector<int>& layers) : network(layers), fitness(0.0f) {}
    };
    
    GeneticAlgorithm(const std::vector<int>& networkLayers, int populationSize);
    
    void evaluateFitness(std::function<float(const NeuralNetwork&)> fitnessFunction);
    void evolve();
    
    const Individual& getBestIndividual() const;
    const std::vector<Individual>& getPopulation() const { return population_; }

private:
    std::vector<Individual> population_;
    std::vector<int> networkLayers_;
    int populationSize_;
    std::mt19937 rng_;
    
    std::vector<Individual> selection();
    Individual crossover(const Individual& parent1, const Individual& parent2);
    void mutate(Individual& individual);
};

// Adaptive learning agent
class AdaptiveLearningAgent : public common::Agent {
public:
    AdaptiveLearningAgent(const std::string& id, const std::vector<int>& networkLayers);
    
    void update(float deltaTime) override;
    
    // Learning interface
    void addExperience(const std::vector<float>& state, int action, float reward, const std::vector<float>& nextState, bool terminal);
    int selectAction(const std::vector<float>& state);
    void trainOnBatch(size_t batchSize = 32);
    
    // State representation
    virtual std::vector<float> getCurrentState() = 0;
    virtual int getNumActions() const = 0;
    virtual float calculateReward() = 0;

protected:
    std::unique_ptr<NeuralNetwork> network_;
    std::unique_ptr<ExperienceReplay> experienceReplay_;
    std::vector<float> lastState_;
    int lastAction_;
    float explorationRate_;
    
private:
    float timeSinceLastTrain_ = 0.0f;
    const float trainingInterval_ = 1.0f; // Train every second
};

// Imitation Learning
class ImitationLearner {
public:
    ImitationLearner(const std::vector<int>& networkLayers);
    
    void addDemonstration(const std::vector<float>& state, int action);
    void train(int epochs = 100, float learningRate = 0.01f);
    int predict(const std::vector<float>& state);
    
private:
    std::unique_ptr<NeuralNetwork> network_;
    std::vector<std::pair<std::vector<float>, int>> demonstrations_;
};

} // namespace learning
} // namespace iapv