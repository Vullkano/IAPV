#pragma once

#include "../common/agent.hpp"
#include "../locomotion/steering_behaviors.hpp"
#include <vector>
#include <memory>

namespace iapv {
namespace crowds {

// Boid agent for flocking behavior
class Boid : public common::Agent {
public:
    Boid(const std::string& id, const common::Vector3D& position = common::Vector3D());
    
    void update(float deltaTime) override;
    void setNeighbors(const std::vector<Boid*>& neighbors) { neighbors_ = neighbors; }
    
    // Flocking parameters
    void setSeparationWeight(float weight) { separationWeight_ = weight; }
    void setAlignmentWeight(float weight) { alignmentWeight_ = weight; }
    void setCohesionWeight(float weight) { cohesionWeight_ = weight; }
    
    void setSeparationRadius(float radius) { separationRadius_ = radius; }
    void setAlignmentRadius(float radius) { alignmentRadius_ = radius; }
    void setCohesionRadius(float radius) { cohesionRadius_ = radius; }

private:
    std::vector<Boid*> neighbors_;
    
    // Flocking weights
    float separationWeight_ = 1.5f;
    float alignmentWeight_ = 1.0f;
    float cohesionWeight_ = 1.0f;
    
    // Flocking radii
    float separationRadius_ = 2.0f;
    float alignmentRadius_ = 4.0f;
    float cohesionRadius_ = 6.0f;
    
    common::Vector3D calculateSeparation();
    common::Vector3D calculateAlignment();
    common::Vector3D calculateCohesion();
};

// Crowd simulation manager
class CrowdSimulation {
public:
    CrowdSimulation() = default;
    
    void addBoid(std::unique_ptr<Boid> boid);
    void update(float deltaTime);
    
    void setNeighborRadius(float radius) { neighborRadius_ = radius; }
    void setBoundary(const common::Vector3D& min, const common::Vector3D& max);
    
    const std::vector<std::unique_ptr<Boid>>& getBoids() const { return boids_; }

private:
    std::vector<std::unique_ptr<Boid>> boids_;
    float neighborRadius_ = 10.0f;
    common::Vector3D boundaryMin_ = common::Vector3D(-50, -50, -50);
    common::Vector3D boundaryMax_ = common::Vector3D(50, 50, 50);
    
    std::vector<Boid*> findNeighbors(const Boid* boid);
    void applyBoundaryForces(Boid* boid);
};

// Crowd density analysis
class CrowdAnalyzer {
public:
    struct DensityData {
        float averageDensity;
        float maxDensity;
        common::Vector3D densityCenter;
        std::vector<common::Vector3D> hotspots;
    };
    
    static DensityData analyzeDensity(const std::vector<std::unique_ptr<Boid>>& boids, float cellSize = 5.0f);
    static float calculateLocalDensity(const common::Vector3D& position, const std::vector<std::unique_ptr<Boid>>& boids, float radius = 5.0f);
};

// Emergent behavior patterns
class EmergentBehaviorDetector {
public:
    enum class BehaviorPattern {
        Flocking,
        Schooling,
        Swarming,
        Milling,
        Splitting,
        Unknown
    };
    
    BehaviorPattern detectPattern(const std::vector<std::unique_ptr<Boid>>& boids);
    
private:
    float calculateAlignmentLevel(const std::vector<std::unique_ptr<Boid>>& boids);
    float calculateCohesionLevel(const std::vector<std::unique_ptr<Boid>>& boids);
    float calculateVelocityVariance(const std::vector<std::unique_ptr<Boid>>& boids);
};

// Crowd pathfinding for groups
class CrowdPathfinding {
public:
    struct FlowField {
        std::vector<std::vector<common::Vector2D>> directions;
        int width, height;
        float cellSize;
    };
    
    CrowdPathfinding(int width, int height, float cellSize = 1.0f);
    
    FlowField generateFlowField(const common::Vector2D& goal);
    common::Vector3D getFlowDirection(const common::Vector3D& position) const;
    
    void addObstacle(const common::Vector2D& position, float radius);
    void removeObstacle(const common::Vector2D& position);

private:
    int width_, height_;
    float cellSize_;
    std::vector<std::vector<bool>> obstacles_;
    FlowField currentFlowField_;
    
    void calculateDistanceField(const common::Vector2D& goal, std::vector<std::vector<float>>& distances);
    common::Vector2D calculateGradient(const std::vector<std::vector<float>>& distances, int x, int y);
};

// Pedestrian agent with more complex behavior
class Pedestrian : public Boid {
public:
    enum class PedestrianState {
        Walking,
        Waiting,
        Following,
        Avoiding
    };
    
    Pedestrian(const std::string& id, const common::Vector3D& position = common::Vector3D());
    
    void update(float deltaTime) override;
    void setDestination(const common::Vector3D& destination) { destination_ = destination; }
    void setGroup(const std::vector<Pedestrian*>& group) { group_ = group; }
    
    PedestrianState getState() const { return state_; }

private:
    common::Vector3D destination_;
    std::vector<Pedestrian*> group_;
    PedestrianState state_ = PedestrianState::Walking;
    float patience_ = 5.0f; // Time to wait before changing behavior
    float waitTimer_ = 0.0f;
    
    common::Vector3D calculateDestinationSeeking();
    common::Vector3D calculateGroupCohesion();
    void updateState(float deltaTime);
};

} // namespace crowds
} // namespace iapv