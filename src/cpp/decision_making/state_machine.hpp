#pragma once

#include "../common/agent.hpp"
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace iapv {
namespace decision_making {

// Finite State Machine implementation
class State {
public:
    virtual ~State() = default;
    virtual void enter(common::Agent* agent) {}
    virtual void update(common::Agent* agent, float deltaTime) = 0;
    virtual void exit(common::Agent* agent) {}
    virtual std::string getName() const = 0;
};

class FiniteStateMachine {
public:
    FiniteStateMachine(common::Agent* agent) : agent_(agent), currentState_(nullptr) {}
    
    void addState(const std::string& name, std::unique_ptr<State> state);
    void setState(const std::string& name);
    void update(float deltaTime);
    
    State* getCurrentState() const { return currentState_; }
    const std::string& getCurrentStateName() const { return currentStateName_; }

private:
    common::Agent* agent_;
    State* currentState_;
    std::string currentStateName_;
    std::unordered_map<std::string, std::unique_ptr<State>> states_;
};

// Decision Tree implementation
struct DecisionNode {
    virtual ~DecisionNode() = default;
    virtual std::string decide(common::Agent* agent) = 0;
};

struct ConditionNode : public DecisionNode {
    std::function<bool(common::Agent*)> condition;
    std::unique_ptr<DecisionNode> trueNode;
    std::unique_ptr<DecisionNode> falseNode;
    
    ConditionNode(std::function<bool(common::Agent*)> cond) : condition(cond) {}
    
    std::string decide(common::Agent* agent) override {
        if (condition(agent)) {
            return trueNode ? trueNode->decide(agent) : "";
        } else {
            return falseNode ? falseNode->decide(agent) : "";
        }
    }
};

struct ActionNode : public DecisionNode {
    std::string action;
    
    ActionNode(const std::string& act) : action(act) {}
    
    std::string decide(common::Agent* agent) override {
        return action;
    }
};

class DecisionTree {
public:
    DecisionTree() : root_(nullptr) {}
    
    void setRoot(std::unique_ptr<DecisionNode> root) {
        root_ = std::move(root);
    }
    
    std::string decide(common::Agent* agent) {
        return root_ ? root_->decide(agent) : "";
    }

private:
    std::unique_ptr<DecisionNode> root_;
};

// Utility-based decision making
struct UtilityAction {
    std::string name;
    std::function<float(common::Agent*)> utilityFunction;
    std::function<void(common::Agent*)> action;
    
    UtilityAction(const std::string& n, 
                  std::function<float(common::Agent*)> utility,
                  std::function<void(common::Agent*)> act)
        : name(n), utilityFunction(utility), action(act) {}
};

class UtilitySystem {
public:
    void addAction(const UtilityAction& action) {
        actions_.push_back(action);
    }
    
    void executeHighestUtilityAction(common::Agent* agent);
    std::string getHighestUtilityActionName(common::Agent* agent);

private:
    std::vector<UtilityAction> actions_;
};

// Common states for virtual characters
class IdleState : public State {
public:
    void update(common::Agent* agent, float deltaTime) override;
    std::string getName() const override { return "Idle"; }
};

class PatrolState : public State {
public:
    PatrolState(const std::vector<common::Vector3D>& waypoints) 
        : waypoints_(waypoints), currentWaypoint_(0) {}
    
    void enter(common::Agent* agent) override;
    void update(common::Agent* agent, float deltaTime) override;
    std::string getName() const override { return "Patrol"; }

private:
    std::vector<common::Vector3D> waypoints_;
    size_t currentWaypoint_;
};

class ChaseState : public State {
public:
    ChaseState(const std::string& targetId) : targetId_(targetId) {}
    
    void update(common::Agent* agent, float deltaTime) override;
    std::string getName() const override { return "Chase"; }

private:
    std::string targetId_;
};

class FleeState : public State {
public:
    FleeState(const std::string& threatId) : threatId_(threatId) {}
    
    void update(common::Agent* agent, float deltaTime) override;
    std::string getName() const override { return "Flee"; }

private:
    std::string threatId_;
};

} // namespace decision_making
} // namespace iapv