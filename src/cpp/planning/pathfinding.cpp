#include "pathfinding.hpp"
#include <algorithm>
#include <cmath>

namespace iapv {
namespace planning {

using common::Vector2D;

// GridWorld implementation
GridWorld::GridWorld(int width, int height, float cellSize)
    : width_(width), height_(height), cellSize_(cellSize) {
    walkable_.resize(height);
    for (int y = 0; y < height; ++y) {
        walkable_[y].resize(width, true); // All cells walkable by default
    }
}

bool GridWorld::isWalkable(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        return false;
    }
    return walkable_[y][x];
}

void GridWorld::setWalkable(int x, int y, bool walkable) {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
        walkable_[y][x] = walkable;
    }
}

Vector2D GridWorld::gridToWorld(int x, int y) const {
    return Vector2D(x * cellSize_, y * cellSize_);
}

std::pair<int, int> GridWorld::worldToGrid(const Vector2D& position) const {
    return {static_cast<int>(position.x / cellSize_), 
            static_cast<int>(position.y / cellSize_)};
}

// AStarPathfinder implementation
std::vector<Vector2D> AStarPathfinder::findPath(const Vector2D& start, const Vector2D& goal) {
    auto [startX, startY] = world_.worldToGrid(start);
    auto [goalX, goalY] = world_.worldToGrid(goal);
    
    if (!world_.isWalkable(startX, startY) || !world_.isWalkable(goalX, goalY)) {
        return {}; // No path possible
    }
    
    auto compare = [](Node* a, Node* b) { return a->fCost() > b->fCost(); };
    std::priority_queue<Node*, std::vector<Node*>, decltype(compare)> openSet(compare);
    std::unordered_set<Node*> closedSet;
    std::unordered_map<int, std::unique_ptr<Node>> allNodes;
    
    auto getNodeKey = [](int x, int y) { return y * 10000 + x; };
    
    // Create start node
    auto startNode = std::make_unique<Node>(world_.gridToWorld(startX, startY));
    startNode->gCost = 0;
    startNode->hCost = heuristic(startNode->position, world_.gridToWorld(goalX, goalY));
    
    Node* startPtr = startNode.get();
    allNodes[getNodeKey(startX, startY)] = std::move(startNode);
    openSet.push(startPtr);
    
    while (!openSet.empty()) {
        Node* current = openSet.top();
        openSet.pop();
        
        if (closedSet.find(current) != closedSet.end()) {
            continue;
        }
        
        closedSet.insert(current);
        
        auto [currentX, currentY] = world_.worldToGrid(current->position);
        
        // Check if we reached the goal
        if (currentX == goalX && currentY == goalY) {
            // Reconstruct path
            std::vector<Vector2D> path;
            Node* node = current;
            while (node != nullptr) {
                path.push_back(node->position);
                node = node->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }
        
        // Check neighbors
        for (auto [nx, ny] : getNeighbors(currentX, currentY)) {
            if (!world_.isWalkable(nx, ny)) continue;
            
            int key = getNodeKey(nx, ny);
            Node* neighbor = nullptr;
            
            if (allNodes.find(key) == allNodes.end()) {
                auto newNode = std::make_unique<Node>(world_.gridToWorld(nx, ny));
                neighbor = newNode.get();
                allNodes[key] = std::move(newNode);
            } else {
                neighbor = allNodes[key].get();
            }
            
            if (closedSet.find(neighbor) != closedSet.end()) continue;
            
            float tentativeGCost = current->gCost + 
                (current->position - neighbor->position).magnitude();
                
            if (neighbor->parent == nullptr || tentativeGCost < neighbor->gCost) {
                neighbor->parent = current;
                neighbor->gCost = tentativeGCost;
                neighbor->hCost = heuristic(neighbor->position, world_.gridToWorld(goalX, goalY));
                openSet.push(neighbor);
            }
        }
    }
    
    return {}; // No path found
}

float AStarPathfinder::heuristic(const Vector2D& a, const Vector2D& b) const {
    return (a - b).magnitude(); // Euclidean distance
}

std::vector<std::pair<int, int>> AStarPathfinder::getNeighbors(int x, int y) const {
    std::vector<std::pair<int, int>> neighbors;
    
    // 8-directional movement
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) continue;
            neighbors.emplace_back(x + dx, y + dy);
        }
    }
    
    return neighbors;
}

// Behavior Tree implementations
BehaviorStatus SequenceNode::execute() {
    while (currentChild_ < children_.size()) {
        BehaviorStatus status = children_[currentChild_]->execute();
        
        if (status == BehaviorStatus::Failure) {
            reset();
            return BehaviorStatus::Failure;
        } else if (status == BehaviorStatus::Running) {
            return BehaviorStatus::Running;
        }
        
        // Success, move to next child
        currentChild_++;
    }
    
    // All children succeeded
    reset();
    return BehaviorStatus::Success;
}

void SequenceNode::reset() {
    currentChild_ = 0;
    for (auto& child : children_) {
        child->reset();
    }
}

BehaviorStatus SelectorNode::execute() {
    while (currentChild_ < children_.size()) {
        BehaviorStatus status = children_[currentChild_]->execute();
        
        if (status == BehaviorStatus::Success) {
            reset();
            return BehaviorStatus::Success;
        } else if (status == BehaviorStatus::Running) {
            return BehaviorStatus::Running;
        }
        
        // Failure, try next child
        currentChild_++;
    }
    
    // All children failed
    reset();
    return BehaviorStatus::Failure;
}

void SelectorNode::reset() {
    currentChild_ = 0;
    for (auto& child : children_) {
        child->reset();
    }
}

} // namespace planning
} // namespace iapv