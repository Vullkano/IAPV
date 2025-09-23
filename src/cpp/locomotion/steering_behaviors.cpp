#include "steering_behaviors.hpp"
#include "../common/math_utils.hpp"
#include <algorithm>
#include <cmath>

namespace iapv {
namespace locomotion {

// SeekBehavior implementation
common::Vector3D SeekBehavior::calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) {
    common::Vector3D desired = target_ - agent->getPosition();
    desired = desired.normalized() * 10.0f; // Max speed
    
    return desired - agent->getVelocity();
}

// FleeBehavior implementation
common::Vector3D FleeBehavior::calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) {
    common::Vector3D desired = agent->getPosition() - threat_;
    desired = desired.normalized() * 10.0f; // Max speed
    
    return desired - agent->getVelocity();
}

// WanderBehavior implementation
common::Vector3D WanderBehavior::calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) {
    // Add random jitter to wander target
    wanderTarget_ = wanderTarget_ + common::Vector3D(
        common::Random::range(-wanderJitter_, wanderJitter_),
        0,
        common::Random::range(-wanderJitter_, wanderJitter_)
    );
    
    // Normalize to stay on circle
    wanderTarget_ = wanderTarget_.normalized() * wanderRadius_;
    
    // Calculate wander circle center
    common::Vector3D velocity = agent->getVelocity();
    if (velocity.magnitude() > 0.1f) {
        velocity = velocity.normalized();
    } else {
        velocity = common::Vector3D(0, 0, 1); // Default forward
    }
    
    common::Vector3D circleCenter = agent->getPosition() + velocity * wanderDistance_;
    common::Vector3D target = circleCenter + wanderTarget_;
    
    // Seek toward the target
    common::Vector3D desired = target - agent->getPosition();
    desired = desired.normalized() * 5.0f; // Moderate speed for wandering
    
    return desired - agent->getVelocity();
}

// SeparationBehavior implementation
common::Vector3D SeparationBehavior::calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) {
    common::Vector3D steer(0, 0, 0);
    int count = 0;
    
    for (const auto* neighbor : neighbors) {
        if (neighbor == agent) continue;
        
        float distance = (agent->getPosition() - neighbor->getPosition()).magnitude();
        if (distance > 0 && distance < separationRadius_) {
            common::Vector3D diff = agent->getPosition() - neighbor->getPosition();
            diff = diff.normalized();
            diff = diff * (1.0f / distance); // Weight by distance
            steer = steer + diff;
            count++;
        }
    }
    
    if (count > 0) {
        steer = steer * (1.0f / count);
        steer = steer.normalized() * 10.0f; // Max speed
        steer = steer - agent->getVelocity();
    }
    
    return steer;
}

// AlignmentBehavior implementation
common::Vector3D AlignmentBehavior::calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) {
    common::Vector3D sum(0, 0, 0);
    int count = 0;
    
    for (const auto* neighbor : neighbors) {
        if (neighbor == agent) continue;
        
        float distance = (agent->getPosition() - neighbor->getPosition()).magnitude();
        if (distance > 0 && distance < alignmentRadius_) {
            sum = sum + neighbor->getVelocity();
            count++;
        }
    }
    
    if (count > 0) {
        sum = sum * (1.0f / count);
        sum = sum.normalized() * 10.0f; // Max speed
        common::Vector3D steer = sum - agent->getVelocity();
        return steer;
    }
    
    return common::Vector3D(0, 0, 0);
}

// CohesionBehavior implementation
common::Vector3D CohesionBehavior::calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) {
    common::Vector3D sum(0, 0, 0);
    int count = 0;
    
    for (const auto* neighbor : neighbors) {
        if (neighbor == agent) continue;
        
        float distance = (agent->getPosition() - neighbor->getPosition()).magnitude();
        if (distance > 0 && distance < cohesionRadius_) {
            sum = sum + neighbor->getPosition();
            count++;
        }
    }
    
    if (count > 0) {
        sum = sum * (1.0f / count);
        // Seek toward center of mass
        common::Vector3D desired = sum - agent->getPosition();
        desired = desired.normalized() * 10.0f; // Max speed
        return desired - agent->getVelocity();
    }
    
    return common::Vector3D(0, 0, 0);
}



// AvoidanceBehavior implementation
common::Vector3D AvoidanceBehavior::calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) {
    common::Vector3D steer(0, 0, 0);
    
    // Avoid other agents
    for (const auto* neighbor : neighbors) {
        if (neighbor == agent) continue;
        
        float distance = (agent->getPosition() - neighbor->getPosition()).magnitude();
        if (distance > 0 && distance < avoidanceRadius_) {
            common::Vector3D diff = agent->getPosition() - neighbor->getPosition();
            diff = diff.normalized();
            diff = diff * (avoidanceRadius_ / distance); // Stronger avoidance when closer
            steer = steer + diff;
        }
    }
    
    // Avoid obstacles
    for (const auto& obstacle : obstacles_) {
        float distance = (agent->getPosition() - obstacle.position).magnitude();
        float totalRadius = obstacle.radius + 1.0f; // Agent radius
        
        if (distance < totalRadius + avoidanceRadius_) {
            common::Vector3D diff = agent->getPosition() - obstacle.position;
            diff = diff.normalized();
            diff = diff * ((totalRadius + avoidanceRadius_) / distance);
            steer = steer + diff;
        }
    }
    
    if (steer.magnitude() > 0) {
        steer = steer.normalized() * 10.0f; // Max speed
        steer = steer - agent->getVelocity();
    }
    
    return steer;
}

// SteeringController implementation
void SteeringController::update(float deltaTime, const std::vector<common::Agent*>& neighbors) {
    common::Vector3D totalForce(0, 0, 0);
    
    for (const auto& behavior : behaviors_) {
        if (behavior->enabled) {
            common::Vector3D force = behavior->calculate(agent_, neighbors);
            totalForce = totalForce + force * behavior->weight;
        }
    }
    
    // Limit the steering force
    totalForce = truncate(totalForce, maxForce_);
    
    // Apply force to velocity
    common::Vector3D newVelocity = agent_->getVelocity() + totalForce * deltaTime;
    newVelocity = truncate(newVelocity, maxSpeed_);
    
    agent_->setVelocity(newVelocity);
    
    // Update position
    common::Vector3D newPosition = agent_->getPosition() + newVelocity * deltaTime;
    agent_->setPosition(newPosition);
}

common::Vector3D SteeringController::truncate(const common::Vector3D& vector, float maxLength) {
    float magnitude = vector.magnitude();
    if (magnitude > maxLength) {
        return vector.normalized() * maxLength;
    }
    return vector;
}

// MovementAnimator implementation
void MovementAnimator::update(common::Agent* agent, float deltaTime) {
    animationTime_ += deltaTime;
    
    AnimationType newAnimation = determineAnimation(agent->getVelocity());
    
    if (newAnimation != currentAnimation_) {
        currentAnimation_ = newAnimation;
        animationTime_ = 0.0f;
        
        // Store animation state in agent memory for external systems to use
        agent->setMemory("animation_type", static_cast<int>(currentAnimation_));
        agent->setMemory("animation_time", animationTime_);
    }
    
    agent->setMemory("animation_time", animationTime_);
}

MovementAnimator::AnimationType MovementAnimator::determineAnimation(const common::Vector3D& velocity) {
    float speed = velocity.magnitude();
    
    if (speed < 0.1f) {
        return AnimationType::Idle;
    } else if (speed < 5.0f) {
        return AnimationType::Walk;
    } else {
        return AnimationType::Run;
    }
}

} // namespace locomotion
} // namespace iapv