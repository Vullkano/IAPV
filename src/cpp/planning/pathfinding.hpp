#pragma once

#include "../common/math_utils.hpp"
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace iapv {
namespace planning {

using common::Vector2D;

// Node for A* pathfinding
struct Node {
    Vector2D position;
    float gCost;  // Distance from start
    float hCost;  // Heuristic distance to goal
    float fCost() const { return gCost + hCost; }
    Node* parent;
    
    Node(const Vector2D& pos) : position(pos), gCost(0), hCost(0), parent(nullptr) {}
};

// Simple grid-based world representation
class GridWorld {
public:
    GridWorld(int width, int height, float cellSize = 1.0f);
    
    bool isWalkable(int x, int y) const;
    void setWalkable(int x, int y, bool walkable);
    
    Vector2D gridToWorld(int x, int y) const;
    std::pair<int, int> worldToGrid(const Vector2D& position) const;
    
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    float getCellSize() const { return cellSize_; }

private:
    int width_, height_;
    float cellSize_;
    std::vector<std::vector<bool>> walkable_;
};

// A* pathfinder implementation
class AStarPathfinder {
public:
    AStarPathfinder(const GridWorld& world) : world_(world) {}
    
    std::vector<Vector2D> findPath(const Vector2D& start, const Vector2D& goal);
    
private:
    const GridWorld& world_;
    
    float heuristic(const Vector2D& a, const Vector2D& b) const;
    std::vector<std::pair<int, int>> getNeighbors(int x, int y) const;
};

// Behavior Tree nodes
enum class BehaviorStatus {
    Success,
    Failure,
    Running
};

class BehaviorNode {
public:
    virtual ~BehaviorNode() = default;
    virtual BehaviorStatus execute() = 0;
    virtual void reset() {}
};

// Composite nodes
class SequenceNode : public BehaviorNode {
public:
    void addChild(std::unique_ptr<BehaviorNode> child) {
        children_.push_back(std::move(child));
    }
    
    BehaviorStatus execute() override;
    void reset() override;

private:
    std::vector<std::unique_ptr<BehaviorNode>> children_;
    size_t currentChild_ = 0;
};

class SelectorNode : public BehaviorNode {
public:
    void addChild(std::unique_ptr<BehaviorNode> child) {
        children_.push_back(std::move(child));
    }
    
    BehaviorStatus execute() override;
    void reset() override;

private:
    std::vector<std::unique_ptr<BehaviorNode>> children_;
    size_t currentChild_ = 0;
};

// Action node
class ActionNode : public BehaviorNode {
public:
    ActionNode(std::function<BehaviorStatus()> action) : action_(action) {}
    
    BehaviorStatus execute() override {
        return action_();
    }

private:
    std::function<BehaviorStatus()> action_;
};

// Condition node
class ConditionNode : public BehaviorNode {
public:
    ConditionNode(std::function<bool()> condition) : condition_(condition) {}
    
    BehaviorStatus execute() override {
        return condition_() ? BehaviorStatus::Success : BehaviorStatus::Failure;
    }

private:
    std::function<bool()> condition_;
};

} // namespace planning
} // namespace iapv