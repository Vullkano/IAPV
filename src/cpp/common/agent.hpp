#pragma once

#include "math_utils.hpp"
#include <string>
#include <map>
#include <any>

namespace iapv {
namespace common {

// Base class for all virtual characters/agents
class Agent {
public:
    Agent(const std::string& id, const Vector3D& position = Vector3D())
        : id_(id), position_(position), velocity_(0, 0, 0), health_(100.0f), energy_(100.0f) {}

    virtual ~Agent() = default;

    // Basic properties
    const std::string& getId() const { return id_; }
    const Vector3D& getPosition() const { return position_; }
    const Vector3D& getVelocity() const { return velocity_; }
    float getHealth() const { return health_; }
    float getEnergy() const { return energy_; }

    void setPosition(const Vector3D& position) { position_ = position; }
    void setVelocity(const Vector3D& velocity) { velocity_ = velocity; }
    void setHealth(float health) { health_ = std::max(0.0f, std::min(100.0f, health)); }
    void setEnergy(float energy) { energy_ = std::max(0.0f, std::min(100.0f, energy)); }

    // Update method - called each frame
    virtual void update(float deltaTime) = 0;

    // Agent memory system for storing arbitrary data
    template<typename T>
    void setMemory(const std::string& key, const T& value) {
        memory_[key] = value;
    }

    template<typename T>
    T getMemory(const std::string& key, const T& defaultValue = T{}) const {
        auto it = memory_.find(key);
        if (it != memory_.end()) {
            try {
                return std::any_cast<T>(it->second);
            } catch (const std::bad_any_cast&) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    bool hasMemory(const std::string& key) const {
        return memory_.find(key) != memory_.end();
    }

protected:
    std::string id_;
    Vector3D position_;
    Vector3D velocity_;
    float health_;
    float energy_;
    std::map<std::string, std::any> memory_;
};

// Environment class for managing world state
class Environment {
public:
    Environment() = default;
    virtual ~Environment() = default;

    virtual void addAgent(std::shared_ptr<Agent> agent) {
        agents_[agent->getId()] = agent;
    }

    virtual void removeAgent(const std::string& id) {
        agents_.erase(id);
    }

    virtual std::shared_ptr<Agent> getAgent(const std::string& id) {
        auto it = agents_.find(id);
        return (it != agents_.end()) ? it->second : nullptr;
    }

    virtual void update(float deltaTime) {
        for (auto& [id, agent] : agents_) {
            agent->update(deltaTime);
        }
    }

    const std::map<std::string, std::shared_ptr<Agent>>& getAgents() const {
        return agents_;
    }

protected:
    std::map<std::string, std::shared_ptr<Agent>> agents_;
};

} // namespace common
} // namespace iapv