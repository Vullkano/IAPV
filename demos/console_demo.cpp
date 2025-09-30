#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>

#include "../src/cpp/common/agent.hpp"
#include "../src/cpp/common/math_utils.hpp"
#include "../src/cpp/locomotion/steering_behaviors.hpp"
#include "../src/cpp/crowds/flocking.hpp"
#include "../src/cpp/planning/pathfinding.hpp"

using namespace iapv;

// Simple agent implementation for demo
class DemoAgent : public common::Agent {
public:
    DemoAgent(const std::string& id, const common::Vector3D& position) : Agent(id, position) {
        controller_ = std::make_unique<locomotion::SteeringController>(this);
    }
    
    void update(float deltaTime) override {
        if (controller_) {
            std::vector<common::Agent*> neighbors; // Empty for this simple demo
            controller_->update(deltaTime, neighbors);
        }
    }
    
    void addSteeringBehavior(std::unique_ptr<locomotion::SteeringBehavior> behavior) {
        controller_->addBehavior(std::move(behavior));
    }

private:
    std::unique_ptr<locomotion::SteeringController> controller_;
};

void runFlockingDemo() {
    std::cout << "\n=== Flocking Demo ===" << std::endl;
    
    // Create environment
    common::Environment environment;
    
    // Create flocking agents
    std::vector<std::unique_ptr<crowds::Boid>> boids;
    for (int i = 0; i < 5; ++i) {
        auto boid = std::make_unique<crowds::Boid>("boid_" + std::to_string(i), 
                                                   common::Vector3D(i * 5.0f, 0, i * 3.0f));
        boids.push_back(std::move(boid));
    }
    
    // Create crowd simulation
    crowds::CrowdSimulation crowd;
    for (auto& boid : boids) {
        crowd.addBoid(std::move(boid));
    }
    
    // Run simulation for a few steps
    std::cout << "Running flocking simulation..." << std::endl;
    for (int step = 0; step < 10; ++step) {
        crowd.update(0.1f);
        
        // Print positions every few steps
        if (step % 3 == 0) {
            std::cout << "Step " << step << ":" << std::endl;
            for (const auto& boid : crowd.getBoids()) {
                const auto& pos = boid->getPosition();
                std::cout << "  " << boid->getId() << ": (" 
                         << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
            }
        }
    }
}

void runPathfindingDemo() {
    std::cout << "\n=== Pathfinding Demo ===" << std::endl;
    
    // Create a small grid world
    planning::GridWorld world(10, 10, 1.0f);
    
    // Add some obstacles
    world.setWalkable(3, 3, false);
    world.setWalkable(3, 4, false);
    world.setWalkable(3, 5, false);
    world.setWalkable(4, 3, false);
    world.setWalkable(5, 3, false);
    
    // Create pathfinder
    planning::AStarPathfinder pathfinder(world);
    
    // Find path from (1,1) to (8,8)
    common::Vector2D start(1.0f, 1.0f);
    common::Vector2D goal(8.0f, 8.0f);
    
    std::cout << "Finding path from (" << start.x << ", " << start.y 
              << ") to (" << goal.x << ", " << goal.y << ")" << std::endl;
    
    auto path = pathfinder.findPath(start, goal);
    
    if (!path.empty()) {
        std::cout << "Path found with " << path.size() << " waypoints:" << std::endl;
        for (size_t i = 0; i < path.size(); ++i) {
            std::cout << "  " << i << ": (" << path[i].x << ", " << path[i].y << ")" << std::endl;
        }
    } else {
        std::cout << "No path found!" << std::endl;
    }
}

void runSteeringDemo() {
    std::cout << "\n=== Steering Behaviors Demo ===" << std::endl;
    
    // Create demo agent
    auto agent = std::make_unique<DemoAgent>("demo_agent", common::Vector3D(0, 0, 0));
    
    // Add seek behavior towards target
    common::Vector3D target(10, 0, 10);
    agent->addSteeringBehavior(std::make_unique<locomotion::SeekBehavior>(target));
    
    std::cout << "Agent seeking towards target (" << target.x << ", " << target.y << ", " << target.z << ")" << std::endl;
    
    // Simulate movement
    for (int step = 0; step < 15; ++step) {
        agent->update(0.2f);
        
        const auto& pos = agent->getPosition();
        const auto& vel = agent->getVelocity();
        
        if (step % 3 == 0) {
            std::cout << "Step " << step << ": Position(" << pos.x << ", " << pos.y << ", " << pos.z 
                     << ") Velocity(" << vel.x << ", " << vel.y << ", " << vel.z << ")" << std::endl;
        }
        
        // Check if close to target
        float distance = (pos - target).magnitude();
        if (distance < 1.0f) {
            std::cout << "Reached target! Final distance: " << distance << std::endl;
            break;
        }
    }
}

void printMenu() {
    std::cout << "\n=== IAPV Console Demo ===" << std::endl;
    std::cout << "1. Steering Behaviors Demo" << std::endl;
    std::cout << "2. Pathfinding Demo" << std::endl;
    std::cout << "3. Flocking Demo" << std::endl;
    std::cout << "4. Run All Demos" << std::endl;
    std::cout << "0. Exit" << std::endl;
    std::cout << "Enter your choice: ";
}

int main() {
    std::cout << "Welcome to the IAPV (AI for Virtual Characters) Console Demo!" << std::endl;
    std::cout << "This demonstrates basic AI behaviors implemented in C++." << std::endl;
    
    int choice;
    do {
        printMenu();
        std::cin >> choice;
        
        switch (choice) {
            case 1:
                runSteeringDemo();
                break;
            case 2:
                runPathfindingDemo();
                break;
            case 3:
                runFlockingDemo();
                break;
            case 4:
                runSteeringDemo();
                runPathfindingDemo();
                runFlockingDemo();
                break;
            case 0:
                std::cout << "Thank you for trying the IAPV demo!" << std::endl;
                break;
            default:
                std::cout << "Invalid choice. Please try again." << std::endl;
                break;
        }
        
        if (choice != 0) {
            std::cout << "\nPress Enter to continue...";
            std::cin.ignore();
            std::cin.get();
        }
        
    } while (choice != 0);
    
    return 0;
}