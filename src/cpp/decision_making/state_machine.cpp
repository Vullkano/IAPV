#include "state_machine.hpp"
#include "../common/math_utils.hpp"
#include <algorithm>

namespace iapv {
namespace decision_making {

// FiniteStateMachine implementation
void FiniteStateMachine::addState(const std::string& name, std::unique_ptr<State> state) {
    states_[name] = std::move(state);
}

void FiniteStateMachine::setState(const std::string& name) {
    auto it = states_.find(name);
    if (it != states_.end()) {
        if (currentState_) {
            currentState_->exit(agent_);
        }
        
        currentState_ = it->second.get();
        currentStateName_ = name;
        currentState_->enter(agent_);
    }
}

void FiniteStateMachine::update(float deltaTime) {
    if (currentState_) {
        currentState_->update(agent_, deltaTime);
    }
}

// UtilitySystem implementation
void UtilitySystem::executeHighestUtilityAction(common::Agent* agent) {
    if (actions_.empty()) return;
    
    float highestUtility = -1.0f;
    UtilityAction* bestAction = nullptr;
    
    for (auto& action : actions_) {
        float utility = action.utilityFunction(agent);
        if (utility > highestUtility) {
            highestUtility = utility;
            bestAction = &action;
        }
    }
    
    if (bestAction) {
        bestAction->action(agent);
    }
}

std::string UtilitySystem::getHighestUtilityActionName(common::Agent* agent) {
    if (actions_.empty()) return "";
    
    float highestUtility = -1.0f;
    std::string bestActionName = "";
    
    for (const auto& action : actions_) {
        float utility = action.utilityFunction(agent);
        if (utility > highestUtility) {
            highestUtility = utility;
            bestActionName = action.name;
        }
    }
    
    return bestActionName;
}

// IdleState implementation
void IdleState::update(common::Agent* agent, float deltaTime) {
    // Simple idle behavior - maybe look around occasionally
    static float idleTimer = 0.0f;
    idleTimer += deltaTime;
    
    if (idleTimer > 3.0f) {
        // Reset energy slightly while idle
        float currentEnergy = agent->getEnergy();
        agent->setEnergy(std::min(100.0f, currentEnergy + 5.0f * deltaTime));
        idleTimer = 0.0f;
    }
}

// PatrolState implementation
void PatrolState::enter(common::Agent* agent) {
    if (!waypoints_.empty()) {
        agent->setMemory("patrol_target", waypoints_[currentWaypoint_]);
    }
}

void PatrolState::update(common::Agent* agent, float deltaTime) {
    if (waypoints_.empty()) return;
    
    common::Vector3D currentPos = agent->getPosition();
    common::Vector3D target = waypoints_[currentWaypoint_];
    
    // Calculate direction to target
    common::Vector3D direction = target - currentPos;
    float distance = direction.magnitude();
    
    if (distance < 1.0f) {
        // Reached waypoint, move to next
        currentWaypoint_ = (currentWaypoint_ + 1) % waypoints_.size();
        target = waypoints_[currentWaypoint_];
        agent->setMemory("patrol_target", target);
    } else {
        // Move towards target
        direction = direction.normalized();
        float speed = 5.0f; // units per second
        common::Vector3D velocity = direction * speed;
        agent->setVelocity(velocity);
        
        common::Vector3D newPos = currentPos + velocity * deltaTime;
        agent->setPosition(newPos);
    }
}

// ChaseState implementation
void ChaseState::update(common::Agent* agent, float deltaTime) {
    // This would typically access the environment to find the target
    // For now, we'll use a simplified version
    common::Vector3D targetPos = agent->getMemory<common::Vector3D>("chase_target", common::Vector3D(0, 0, 0));
    common::Vector3D currentPos = agent->getPosition();
    
    common::Vector3D direction = targetPos - currentPos;
    float distance = direction.magnitude();
    
    if (distance > 0.1f) {
        direction = direction.normalized();
        float speed = 8.0f; // Faster than patrol
        common::Vector3D velocity = direction * speed;
        agent->setVelocity(velocity);
        
        common::Vector3D newPos = currentPos + velocity * deltaTime;
        agent->setPosition(newPos);
    }
    
    // Consume energy while chasing
    float currentEnergy = agent->getEnergy();
    agent->setEnergy(std::max(0.0f, currentEnergy - 10.0f * deltaTime));
}

// FleeState implementation
void FleeState::update(common::Agent* agent, float deltaTime) {
    common::Vector3D threatPos = agent->getMemory<common::Vector3D>("threat_position", common::Vector3D(0, 0, 0));
    common::Vector3D currentPos = agent->getPosition();
    
    // Move away from threat
    common::Vector3D direction = currentPos - threatPos;
    if (direction.magnitude() > 0.1f) {
        direction = direction.normalized();
        float speed = 10.0f; // Fastest movement
        common::Vector3D velocity = direction * speed;
        agent->setVelocity(velocity);
        
        common::Vector3D newPos = currentPos + velocity * deltaTime;
        agent->setPosition(newPos);
    }
    
    // Consume energy while fleeing
    float currentEnergy = agent->getEnergy();
    agent->setEnergy(std::max(0.0f, currentEnergy - 15.0f * deltaTime));
}

} // namespace decision_making
} // namespace iapv