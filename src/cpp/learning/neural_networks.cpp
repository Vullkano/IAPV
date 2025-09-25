#include "neural_networks.hpp"
#include <algorithm>
#include <fstream>
#include <cmath>

namespace iapv {
namespace learning {

// NeuralNetwork implementation
NeuralNetwork::NeuralNetwork(const std::vector<int>& layers) : layers_(layers) {
    // Initialize weights and biases
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        weights_.emplace_back();
        biases_.emplace_back();
        
        weights_[i].resize(layers[i]);
        biases_[i].resize(layers[i + 1], 0.0f);
        
        for (int j = 0; j < layers[i]; ++j) {
            weights_[i][j].resize(layers[i + 1]);
            for (int k = 0; k < layers[i + 1]; ++k) {
                weights_[i][j][k] = randomWeight();
            }
        }
    }
}

std::vector<float> NeuralNetwork::feedForward(const std::vector<float>& inputs) {
    std::vector<float> activations = inputs;
    
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        std::vector<float> newActivations(layers_[layer + 1], 0.0f);
        
        for (int j = 0; j < layers_[layer + 1]; ++j) {
            float sum = biases_[layer][j];
            for (int i = 0; i < layers_[layer]; ++i) {
                sum += activations[i] * weights_[layer][i][j];
            }
            newActivations[j] = sigmoid(sum);
        }
        
        activations = newActivations;
    }
    
    return activations;
}

void NeuralNetwork::backpropagate(const std::vector<float>& inputs, const std::vector<float>& targets, float learningRate) {
    // Forward pass to get all activations
    std::vector<std::vector<float>> activations;
    activations.push_back(inputs);
    
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        std::vector<float> newActivations(layers_[layer + 1], 0.0f);
        
        for (int j = 0; j < layers_[layer + 1]; ++j) {
            float sum = biases_[layer][j];
            for (int i = 0; i < layers_[layer]; ++i) {
                sum += activations[layer][i] * weights_[layer][i][j];
            }
            newActivations[j] = sigmoid(sum);
        }
        
        activations.push_back(newActivations);
    }
    
    // Backward pass
    std::vector<std::vector<float>> deltas(weights_.size());
    
    // Output layer
    int outputLayer = weights_.size() - 1;
    deltas[outputLayer].resize(layers_.back());
    for (int i = 0; i < layers_.back(); ++i) {
        float output = activations.back()[i];
        float error = targets[i] - output;
        deltas[outputLayer][i] = error * sigmoidDerivative(output);
    }
    
    // Hidden layers
    for (int layer = outputLayer - 1; layer >= 0; --layer) {
        deltas[layer].resize(layers_[layer + 1]);
        for (int i = 0; i < layers_[layer + 1]; ++i) {
            float error = 0.0f;
            for (int j = 0; j < layers_[layer + 2]; ++j) {
                error += deltas[layer + 1][j] * weights_[layer + 1][i][j];
            }
            deltas[layer][i] = error * sigmoidDerivative(activations[layer + 1][i]);
        }
    }
    
    // Update weights and biases
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        for (int i = 0; i < layers_[layer]; ++i) {
            for (int j = 0; j < layers_[layer + 1]; ++j) {
                weights_[layer][i][j] += learningRate * activations[layer][i] * deltas[layer][j];
            }
        }
        for (int j = 0; j < layers_[layer + 1]; ++j) {
            biases_[layer][j] += learningRate * deltas[layer][j];
        }
    }
}

void NeuralNetwork::mutate(float mutationRate, float mutationStrength) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, mutationStrength);
    
    for (auto& layerWeights : weights_) {
        for (auto& nodeWeights : layerWeights) {
            for (auto& weight : nodeWeights) {
                if (dis(gen) < mutationRate) {
                    weight += normal(gen);
                }
            }
        }
    }
    
    for (auto& layerBiases : biases_) {
        for (auto& bias : layerBiases) {
            if (dis(gen) < mutationRate) {
                bias += normal(gen);
            }
        }
    }
}

float NeuralNetwork::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

float NeuralNetwork::sigmoidDerivative(float x) const {
    return x * (1.0f - x);
}

float NeuralNetwork::randomWeight() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    return dis(gen);
}

// QLearning implementation
QLearning::QLearning(int numStates, int numActions, float learningRate, float discountFactor, float explorationRate)
    : numStates_(numStates), numActions_(numActions), learningRate_(learningRate),
      discountFactor_(discountFactor), explorationRate_(explorationRate), rng_(std::random_device{}()), dist_(0.0f, 1.0f) {
    
    qTable_.resize(numStates);
    for (auto& row : qTable_) {
        row.resize(numActions, 0.0f);
    }
}

int QLearning::selectAction(int state) {
    if (dist_(rng_) < explorationRate_) {
        // Explore: random action
        std::uniform_int_distribution<int> actionDist(0, numActions_ - 1);
        return actionDist(rng_);
    } else {
        // Exploit: best known action
        int bestAction = 0;
        float bestValue = qTable_[state][0];
        for (int action = 1; action < numActions_; ++action) {
            if (qTable_[state][action] > bestValue) {
                bestValue = qTable_[state][action];
                bestAction = action;
            }
        }
        return bestAction;
    }
}

void QLearning::updateQ(int state, int action, float reward, int nextState) {
    float maxNextQ = *std::max_element(qTable_[nextState].begin(), qTable_[nextState].end());
    float target = reward + discountFactor_ * maxNextQ;
    qTable_[state][action] += learningRate_ * (target - qTable_[state][action]);
}

float QLearning::getQValue(int state, int action) const {
    return qTable_[state][action];
}

// ExperienceReplay implementation
void ExperienceReplay::add(const Experience& experience) {
    if (experiences_.size() < capacity_) {
        experiences_.push_back(experience);
    } else {
        experiences_[currentIndex_] = experience;
        currentIndex_ = (currentIndex_ + 1) % capacity_;
    }
}

std::vector<Experience> ExperienceReplay::sample(size_t batchSize) {
    std::vector<Experience> batch;
    std::uniform_int_distribution<size_t> dist(0, experiences_.size() - 1);
    
    for (size_t i = 0; i < batchSize && i < experiences_.size(); ++i) {
        batch.push_back(experiences_[dist(rng_)]);
    }
    
    return batch;
}

// AdaptiveLearningAgent implementation
AdaptiveLearningAgent::AdaptiveLearningAgent(const std::string& id, const std::vector<int>& networkLayers)
    : Agent(id), explorationRate_(0.1f) {
    network_ = std::make_unique<NeuralNetwork>(networkLayers);
    experienceReplay_ = std::make_unique<ExperienceReplay>(10000);
}

void AdaptiveLearningAgent::update(float deltaTime) {
    timeSinceLastTrain_ += deltaTime;
    
    if (timeSinceLastTrain_ >= trainingInterval_) {
        trainOnBatch();
        timeSinceLastTrain_ = 0.0f;
    }
    
    // Get current state and select action
    std::vector<float> currentState = getCurrentState();
    int action = selectAction(currentState);
    
    // Calculate reward based on previous action
    if (!lastState_.empty()) {
        float reward = calculateReward();
        bool terminal = false; // Derived classes can override this
        addExperience(lastState_, lastAction_, reward, currentState, terminal);
    }
    
    lastState_ = currentState;
    lastAction_ = action;
}

void AdaptiveLearningAgent::addExperience(const std::vector<float>& state, int action, float reward, 
                                        const std::vector<float>& nextState, bool terminal) {
    Experience exp = {state, action, reward, nextState, terminal};
    experienceReplay_->add(exp);
}

int AdaptiveLearningAgent::selectAction(const std::vector<float>& state) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    if (dis(gen) < explorationRate_) {
        // Explore: random action
        std::uniform_int_distribution<int> actionDist(0, getNumActions() - 1);
        return actionDist(gen);
    } else {
        // Exploit: use neural network
        std::vector<float> outputs = network_->feedForward(state);
        return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
    }
}

void AdaptiveLearningAgent::trainOnBatch(size_t batchSize) {
    if (experienceReplay_->size() < batchSize) return;
    
    std::vector<Experience> batch = experienceReplay_->sample(batchSize);
    
    for (const auto& exp : batch) {
        std::vector<float> targetOutputs = network_->feedForward(exp.state);
        
        if (exp.terminal) {
            targetOutputs[exp.action] = exp.reward;
        } else {
            std::vector<float> nextOutputs = network_->feedForward(exp.nextState);
            float maxNextQ = *std::max_element(nextOutputs.begin(), nextOutputs.end());
            targetOutputs[exp.action] = exp.reward + 0.99f * maxNextQ; // Discount factor
        }
        
        network_->backpropagate(exp.state, targetOutputs);
    }
}

} // namespace learning
} // namespace iapv