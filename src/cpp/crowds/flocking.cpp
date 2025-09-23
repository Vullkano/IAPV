#include "flocking.hpp"
#include "../common/math_utils.hpp"
#include <algorithm>
#include <cmath>
#include <map>

namespace iapv {
namespace crowds {

// Boid implementation
Boid::Boid(const std::string& id, const common::Vector3D& position) : Agent(id, position) {
}

void Boid::update(float deltaTime) {
    // Calculate flocking forces
    common::Vector3D separation = calculateSeparation() * separationWeight_;
    common::Vector3D alignment = calculateAlignment() * alignmentWeight_;
    common::Vector3D cohesion = calculateCohesion() * cohesionWeight_;
    
    // Combine forces
    common::Vector3D totalForce = separation + alignment + cohesion;
    
    // Apply force to velocity
    common::Vector3D acceleration = totalForce;
    common::Vector3D newVelocity = getVelocity() + acceleration * deltaTime;
    
    // Limit speed
    float maxSpeed = 8.0f;
    if (newVelocity.magnitude() > maxSpeed) {
        newVelocity = newVelocity.normalized() * maxSpeed;
    }
    
    setVelocity(newVelocity);
    
    // Update position
    common::Vector3D newPosition = getPosition() + newVelocity * deltaTime;
    setPosition(newPosition);
}

common::Vector3D Boid::calculateSeparation() {
    common::Vector3D steer(0, 0, 0);
    int count = 0;
    
    for (const auto* neighbor : neighbors_) {
        float distance = (getPosition() - neighbor->getPosition()).magnitude();
        if (distance > 0 && distance < separationRadius_) {
            common::Vector3D diff = getPosition() - neighbor->getPosition();
            diff = diff.normalized();
            diff = diff * (1.0f / distance); // Weight by distance
            steer = steer + diff;
            count++;
        }
    }
    
    if (count > 0) {
        steer = steer * (1.0f / count);
        steer = steer.normalized() * 10.0f; // Max force
    }
    
    return steer;
}

common::Vector3D Boid::calculateAlignment() {
    common::Vector3D average(0, 0, 0);
    int count = 0;
    
    for (const auto* neighbor : neighbors_) {
        float distance = (getPosition() - neighbor->getPosition()).magnitude();
        if (distance > 0 && distance < alignmentRadius_) {
            average = average + neighbor->getVelocity();
            count++;
        }
    }
    
    if (count > 0) {
        average = average * (1.0f / count);
        average = average.normalized() * 10.0f; // Max speed
        common::Vector3D steer = average - getVelocity();
        return steer;
    }
    
    return common::Vector3D(0, 0, 0);
}

common::Vector3D Boid::calculateCohesion() {
    common::Vector3D center(0, 0, 0);
    int count = 0;
    
    for (const auto* neighbor : neighbors_) {
        float distance = (getPosition() - neighbor->getPosition()).magnitude();
        if (distance > 0 && distance < cohesionRadius_) {
            center = center + neighbor->getPosition();
            count++;
        }
    }
    
    if (count > 0) {
        center = center * (1.0f / count);
        common::Vector3D desired = center - getPosition();
        desired = desired.normalized() * 10.0f; // Max speed
        common::Vector3D steer = desired - getVelocity();
        return steer;
    }
    
    return common::Vector3D(0, 0, 0);
}

// CrowdSimulation implementation
void CrowdSimulation::addBoid(std::unique_ptr<Boid> boid) {
    boids_.push_back(std::move(boid));
}

void CrowdSimulation::update(float deltaTime) {
    // Update neighbor relationships
    for (auto& boid : boids_) {
        std::vector<Boid*> neighbors = findNeighbors(boid.get());
        boid->setNeighbors(neighbors);
    }
    
    // Update all boids
    for (auto& boid : boids_) {
        boid->update(deltaTime);
        applyBoundaryForces(boid.get());
    }
}

void CrowdSimulation::setBoundary(const common::Vector3D& min, const common::Vector3D& max) {
    boundaryMin_ = min;
    boundaryMax_ = max;
}

std::vector<Boid*> CrowdSimulation::findNeighbors(const Boid* boid) {
    std::vector<Boid*> neighbors;
    
    for (auto& otherBoid : boids_) {
        if (otherBoid.get() != boid) {
            float distance = (boid->getPosition() - otherBoid->getPosition()).magnitude();
            if (distance <= neighborRadius_) {
                neighbors.push_back(otherBoid.get());
            }
        }
    }
    
    return neighbors;
}

void CrowdSimulation::applyBoundaryForces(Boid* boid) {
    common::Vector3D position = boid->getPosition();
    common::Vector3D velocity = boid->getVelocity();
    common::Vector3D boundaryForce(0, 0, 0);
    
    float margin = 5.0f;
    float forceStrength = 20.0f;
    
    // Check boundaries and apply repulsion forces
    if (position.x < boundaryMin_.x + margin) {
        boundaryForce.x += forceStrength;
    }
    if (position.x > boundaryMax_.x - margin) {
        boundaryForce.x -= forceStrength;
    }
    if (position.y < boundaryMin_.y + margin) {
        boundaryForce.y += forceStrength;
    }
    if (position.y > boundaryMax_.y - margin) {
        boundaryForce.y -= forceStrength;
    }
    if (position.z < boundaryMin_.z + margin) {
        boundaryForce.z += forceStrength;
    }
    if (position.z > boundaryMax_.z - margin) {
        boundaryForce.z -= forceStrength;
    }
    
    if (boundaryForce.magnitude() > 0) {
        common::Vector3D newVelocity = velocity + boundaryForce * 0.016f; // Assuming ~60fps
        boid->setVelocity(newVelocity);
    }
}

// CrowdAnalyzer implementation
CrowdAnalyzer::DensityData CrowdAnalyzer::analyzeDensity(const std::vector<std::unique_ptr<Boid>>& boids, float cellSize) {
    DensityData data;
    
    if (boids.empty()) {
        data.averageDensity = 0;
        data.maxDensity = 0;
        return data;
    }
    
    // Find bounds
    common::Vector3D minPos = boids[0]->getPosition();
    common::Vector3D maxPos = boids[0]->getPosition();
    
    for (const auto& boid : boids) {
        const auto& pos = boid->getPosition();
        minPos.x = std::min(minPos.x, pos.x);
        minPos.y = std::min(minPos.y, pos.y);
        minPos.z = std::min(minPos.z, pos.z);
        maxPos.x = std::max(maxPos.x, pos.x);
        maxPos.y = std::max(maxPos.y, pos.y);
        maxPos.z = std::max(maxPos.z, pos.z);
    }
    
    // Create density grid
    int gridWidth = static_cast<int>((maxPos.x - minPos.x) / cellSize) + 1;
    int gridHeight = static_cast<int>((maxPos.y - minPos.y) / cellSize) + 1;
    
    std::vector<std::vector<int>> densityGrid(gridHeight, std::vector<int>(gridWidth, 0));
    
    // Count boids in each cell
    for (const auto& boid : boids) {
        const auto& pos = boid->getPosition();
        int gridX = static_cast<int>((pos.x - minPos.x) / cellSize);
        int gridY = static_cast<int>((pos.y - minPos.y) / cellSize);
        
        if (gridX >= 0 && gridX < gridWidth && gridY >= 0 && gridY < gridHeight) {
            densityGrid[gridY][gridX]++;
        }
    }
    
    // Calculate statistics
    float totalDensity = 0;
    data.maxDensity = 0;
    
    for (int y = 0; y < gridHeight; ++y) {
        for (int x = 0; x < gridWidth; ++x) {
            float density = densityGrid[y][x] / (cellSize * cellSize);
            totalDensity += density;
            data.maxDensity = std::max(data.maxDensity, density);
            
            // Mark hotspots
            if (density > data.maxDensity * 0.8f) {
                common::Vector3D hotspot(minPos.x + x * cellSize, minPos.y + y * cellSize, 0);
                data.hotspots.push_back(hotspot);
            }
        }
    }
    
    data.averageDensity = totalDensity / (gridWidth * gridHeight);
    
    return data;
}

float CrowdAnalyzer::calculateLocalDensity(const common::Vector3D& position, const std::vector<std::unique_ptr<Boid>>& boids, float radius) {
    int count = 0;
    for (const auto& boid : boids) {
        float distance = (position - boid->getPosition()).magnitude();
        if (distance <= radius) {
            count++;
        }
    }
    
    float area = M_PI * radius * radius;
    return count / area;
}

// EmergentBehaviorDetector implementation
EmergentBehaviorDetector::BehaviorPattern EmergentBehaviorDetector::detectPattern(const std::vector<std::unique_ptr<Boid>>& boids) {
    if (boids.size() < 3) return BehaviorPattern::Unknown;
    
    float alignmentLevel = calculateAlignmentLevel(boids);
    float cohesionLevel = calculateCohesionLevel(boids);
    float velocityVariance = calculateVelocityVariance(boids);
    
    // Pattern recognition heuristics
    if (alignmentLevel > 0.8f && cohesionLevel > 0.7f && velocityVariance < 0.3f) {
        return BehaviorPattern::Flocking;
    } else if (cohesionLevel > 0.8f && velocityVariance > 0.6f) {
        return BehaviorPattern::Swarming;
    } else if (alignmentLevel < 0.3f && velocityVariance > 0.7f) {
        return BehaviorPattern::Milling;
    } else if (cohesionLevel < 0.4f) {
        return BehaviorPattern::Splitting;
    } else {
        return BehaviorPattern::Schooling;
    }
}

float EmergentBehaviorDetector::calculateAlignmentLevel(const std::vector<std::unique_ptr<Boid>>& boids) {
    if (boids.empty()) return 0.0f;
    
    common::Vector3D avgVelocity(0, 0, 0);
    for (const auto& boid : boids) {
        avgVelocity = avgVelocity + boid->getVelocity();
    }
    avgVelocity = avgVelocity * (1.0f / boids.size());
    
    float totalAlignment = 0.0f;
    for (const auto& boid : boids) {
        common::Vector3D vel = boid->getVelocity();
        if (vel.magnitude() > 0 && avgVelocity.magnitude() > 0) {
            float dot = vel.normalized().dot(avgVelocity.normalized());
            totalAlignment += (dot + 1.0f) / 2.0f; // Normalize to 0-1
        }
    }
    
    return totalAlignment / boids.size();
}

float EmergentBehaviorDetector::calculateCohesionLevel(const std::vector<std::unique_ptr<Boid>>& boids) {
    if (boids.empty()) return 0.0f;
    
    // Calculate center of mass
    common::Vector3D center(0, 0, 0);
    for (const auto& boid : boids) {
        center = center + boid->getPosition();
    }
    center = center * (1.0f / boids.size());
    
    // Calculate average distance from center
    float avgDistance = 0.0f;
    for (const auto& boid : boids) {
        avgDistance += (boid->getPosition() - center).magnitude();
    }
    avgDistance /= boids.size();
    
    // Normalize cohesion (closer = higher cohesion)
    return 1.0f / (1.0f + avgDistance * 0.1f);
}

float EmergentBehaviorDetector::calculateVelocityVariance(const std::vector<std::unique_ptr<Boid>>& boids) {
    if (boids.empty()) return 0.0f;
    
    float avgSpeed = 0.0f;
    for (const auto& boid : boids) {
        avgSpeed += boid->getVelocity().magnitude();
    }
    avgSpeed /= boids.size();
    
    float variance = 0.0f;
    for (const auto& boid : boids) {
        float speed = boid->getVelocity().magnitude();
        variance += (speed - avgSpeed) * (speed - avgSpeed);
    }
    variance /= boids.size();
    
    return std::sqrt(variance) / (avgSpeed + 0.1f); // Normalize
}

} // namespace crowds
} // namespace iapv