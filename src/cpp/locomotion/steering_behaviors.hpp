#pragma once

#include "../common/agent.hpp"
#include "../common/math_utils.hpp"
#include <vector>
#include <memory>

namespace iapv {
namespace locomotion {

// Base class for steering behaviors
class SteeringBehavior {
public:
    virtual ~SteeringBehavior() = default;
    virtual common::Vector3D calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) = 0;
    
    float weight = 1.0f;
    bool enabled = true;
};

// Individual steering behaviors
class SeekBehavior : public SteeringBehavior {
public:
    SeekBehavior(const common::Vector3D& target) : target_(target) {}
    
    common::Vector3D calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) override;
    
    void setTarget(const common::Vector3D& target) { target_ = target; }
    const common::Vector3D& getTarget() const { return target_; }

private:
    common::Vector3D target_;
};

class FleeBehavior : public SteeringBehavior {
public:
    FleeBehavior(const common::Vector3D& threat) : threat_(threat) {}
    
    common::Vector3D calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) override;
    
    void setThreat(const common::Vector3D& threat) { threat_ = threat; }

private:
    common::Vector3D threat_;
};

class WanderBehavior : public SteeringBehavior {
public:
    WanderBehavior(float radius = 2.0f, float distance = 5.0f, float jitter = 1.0f)
        : wanderRadius_(radius), wanderDistance_(distance), wanderJitter_(jitter) {
        wanderTarget_ = common::Vector3D(0, 0, 1);
    }
    
    common::Vector3D calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) override;

private:
    float wanderRadius_;
    float wanderDistance_;
    float wanderJitter_;
    common::Vector3D wanderTarget_;
};

class SeparationBehavior : public SteeringBehavior {
public:
    SeparationBehavior(float separationRadius = 3.0f) : separationRadius_(separationRadius) {}
    
    common::Vector3D calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) override;

private:
    float separationRadius_;
};

class AlignmentBehavior : public SteeringBehavior {
public:
    AlignmentBehavior(float alignmentRadius = 5.0f) : alignmentRadius_(alignmentRadius) {}
    
    common::Vector3D calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) override;

private:
    float alignmentRadius_;
};

class CohesionBehavior : public SteeringBehavior {
public:
    CohesionBehavior(float cohesionRadius = 8.0f) : cohesionRadius_(cohesionRadius) {}
    
    common::Vector3D calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) override;

private:
    float cohesionRadius_;
};

class AvoidanceBehavior : public SteeringBehavior {
public:
    AvoidanceBehavior(float avoidanceRadius = 4.0f) : avoidanceRadius_(avoidanceRadius) {}
    
    common::Vector3D calculate(common::Agent* agent, const std::vector<common::Agent*>& neighbors) override;
    
    void addObstacle(const common::Vector3D& position, float radius) {
        obstacles_.push_back({position, radius});
    }

private:
    float avoidanceRadius_;
    struct Obstacle {
        common::Vector3D position;
        float radius;
    };
    std::vector<Obstacle> obstacles_;
};

// Steering controller that combines multiple behaviors
class SteeringController {
public:
    SteeringController(common::Agent* agent, float maxSpeed = 10.0f, float maxForce = 5.0f)
        : agent_(agent), maxSpeed_(maxSpeed), maxForce_(maxForce) {}
    
    void addBehavior(std::unique_ptr<SteeringBehavior> behavior) {
        behaviors_.push_back(std::move(behavior));
    }
    
    void update(float deltaTime, const std::vector<common::Agent*>& neighbors);
    
    void setMaxSpeed(float maxSpeed) { maxSpeed_ = maxSpeed; }
    void setMaxForce(float maxForce) { maxForce_ = maxForce; }

private:
    common::Agent* agent_;
    float maxSpeed_;
    float maxForce_;
    std::vector<std::unique_ptr<SteeringBehavior>> behaviors_;
    
    common::Vector3D truncate(const common::Vector3D& vector, float maxLength);
};

// Animation and movement utilities
class MovementAnimator {
public:
    enum class AnimationType {
        Walk,
        Run,
        Idle,
        Turn
    };
    
    MovementAnimator() : currentAnimation_(AnimationType::Idle) {}
    
    void update(common::Agent* agent, float deltaTime);
    AnimationType getCurrentAnimation() const { return currentAnimation_; }

private:
    AnimationType currentAnimation_;
    float animationTime_ = 0.0f;
    
    AnimationType determineAnimation(const common::Vector3D& velocity);
};

} // namespace locomotion
} // namespace iapv