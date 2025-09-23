#pragma once

#include "../common/agent.hpp"
#include "../communication/messaging.hpp"
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace iapv {
namespace teamwork {

// Role definitions for team members
enum class TeamRole {
    Leader,
    Follower,
    Scout,
    Guard,
    Support,
    Specialist
};

// Task types that teams can perform
enum class TaskType {
    Patrol,
    Search,
    Escort,
    Defense,
    Attack,
    Rescue,
    Construction,
    Gathering
};

// Team formation patterns
enum class FormationType {
    Line,
    Column,
    Wedge,
    Circle,
    Box,
    Diamond,
    Loose
};

// Task representation
struct Task {
    std::string id;
    TaskType type;
    common::Vector3D location;
    std::map<std::string, float> parameters;
    float priority = 1.0f;
    float deadline = -1.0f; // -1 means no deadline
    bool completed = false;
    std::vector<std::string> requiredRoles;
    
    Task(const std::string& taskId, TaskType t, const common::Vector3D& loc) 
        : id(taskId), type(t), location(loc) {}
};

// Team member interface
class TeamMember : public communication::CommunicatingAgent {
public:
    TeamMember(const std::string& id, communication::CommunicationChannel* channel, 
               TeamRole role = TeamRole::Follower, const common::Vector3D& position = common::Vector3D());
    
    void update(float deltaTime) override;
    
    // Team management
    void joinTeam(const std::string& teamId);
    void leaveTeam();
    void setRole(TeamRole role) { role_ = role; }
    TeamRole getRole() const { return role_; }
    
    // Task execution
    virtual void assignTask(const Task& task);
    virtual void completeTask();
    virtual bool canExecuteTask(const Task& task) const;
    
    // Formation behavior
    void setFormationPosition(const common::Vector3D& position) { formationPosition_ = position; }
    common::Vector3D getFormationPosition() const { return formationPosition_; }
    void followFormation(bool enable) { followingFormation_ = enable; }
    
    // Coordination
    void reportStatus(const std::string& status);
    void requestHelp(const std::string& reason);
    void offerHelp(const std::string& targetId);

protected:
    TeamRole role_;
    std::string teamId_;
    Task* currentTask_ = nullptr;
    common::Vector3D formationPosition_;
    bool followingFormation_ = false;
    
    virtual void executeCurrentTask(float deltaTime);
    virtual void onTaskAssigned(const Task& task) {}
    virtual void onTaskCompleted() {}
};

// Team formation controller
class FormationController {
public:
    FormationController(FormationType type = FormationType::Line) : formationType_(type) {}
    
    void setFormationType(FormationType type) { formationType_ = type; }
    void setFormationParameters(float spacing = 3.0f, float depth = 2.0f) { 
        spacing_ = spacing; 
        depth_ = depth; 
    }
    
    std::vector<common::Vector3D> calculatePositions(const common::Vector3D& center, 
                                                   const common::Vector3D& direction, 
                                                   int numMembers) const;
    
    void updateFormation(const std::vector<TeamMember*>& members, 
                        const common::Vector3D& target, 
                        const common::Vector3D& direction);

private:
    FormationType formationType_;
    float spacing_ = 3.0f;
    float depth_ = 2.0f;
    
    std::vector<common::Vector3D> calculateLineFormation(const common::Vector3D& center, 
                                                       const common::Vector3D& direction, 
                                                       int numMembers) const;
    std::vector<common::Vector3D> calculateColumnFormation(const common::Vector3D& center, 
                                                         const common::Vector3D& direction, 
                                                         int numMembers) const;
    std::vector<common::Vector3D> calculateWedgeFormation(const common::Vector3D& center, 
                                                        const common::Vector3D& direction, 
                                                        int numMembers) const;
    std::vector<common::Vector3D> calculateCircleFormation(const common::Vector3D& center, 
                                                         int numMembers) const;
};

// Team coordination and management
class Team {
public:
    Team(const std::string& id, communication::CommunicationChannel* channel);
    
    void addMember(TeamMember* member);
    void removeMember(const std::string& memberId);
    void setLeader(const std::string& memberId);
    
    void assignTask(const Task& task);
    void assignTaskToMember(const Task& task, const std::string& memberId);
    
    void setFormation(FormationType formation);
    void moveToPosition(const common::Vector3D& position);
    void followTarget(const std::string& targetId);
    
    void update(float deltaTime);
    
    // Team state
    const std::vector<TeamMember*>& getMembers() const { return members_; }
    TeamMember* getLeader() const { return leader_; }
    const std::string& getId() const { return id_; }
    
    // Coordination methods
    void broadcastMessage(const std::string& message);
    void giveOrder(const std::string& order, const std::string& targetId = "");

private:
    std::string id_;
    std::vector<TeamMember*> members_;
    TeamMember* leader_ = nullptr;
    communication::CommunicationChannel* channel_;
    FormationController formation_;
    
    std::queue<Task> taskQueue_;
    common::Vector3D targetPosition_;
    std::string followTargetId_;
    
    void distributeTask(const Task& task);
    TeamMember* findBestMemberForTask(const Task& task);
    void updateFormation(float deltaTime);
};

// Cooperative behavior patterns
class CooperativeBehavior {
public:
    // Task allocation strategies
    enum class AllocationStrategy {
        Random,
        ByRole,
        ByCapability,
        ByProximity,
        LoadBalancing
    };
    
    static std::vector<std::pair<std::string, Task>> allocateTasks(
        const std::vector<Task>& tasks, 
        const std::vector<TeamMember*>& members,
        AllocationStrategy strategy = AllocationStrategy::ByCapability);
    
    // Consensus and decision making
    static std::string makeGroupDecision(const std::vector<std::string>& options,
                                       const std::vector<TeamMember*>& voters);
    
    // Resource sharing
    static void distributeResources(const std::map<std::string, float>& resources,
                                  const std::vector<TeamMember*>& members);
};

// Specialized team types
class MilitarySquad : public Team {
public:
    MilitarySquad(const std::string& id, communication::CommunicationChannel* channel);
    
    void engageTarget(const common::Vector3D& targetPosition);
    void retreat();
    void holdPosition();
    void flankTarget(const common::Vector3D& targetPosition, bool leftFlank = true);

private:
    bool inCombat_ = false;
    common::Vector3D combatTarget_;
};

class SearchTeam : public Team {
public:
    SearchTeam(const std::string& id, communication::CommunicationChannel* channel);
    
    void searchArea(const common::Vector3D& areaCenter, float radius);
    void setSearchPattern(const std::vector<common::Vector3D>& waypoints);
    
private:
    std::vector<common::Vector3D> searchWaypoints_;
    int currentWaypoint_ = 0;
};

class ConstructionCrew : public Team {
public:
    ConstructionCrew(const std::string& id, communication::CommunicationChannel* channel);
    
    void buildStructure(const common::Vector3D& location, const std::string& structureType);
    void repairStructure(const common::Vector3D& location);
    
private:
    std::string currentProject_;
    float constructionProgress_ = 0.0f;
};

// Team performance metrics
class TeamAnalytics {
public:
    struct TeamMetrics {
        float efficiency = 0.0f;       // Tasks completed per time unit
        float coordination = 0.0f;     // How well members work together
        float communication = 0.0f;    // Message frequency and effectiveness
        float cohesion = 0.0f;         // How close members stay to formation
        int tasksCompleted = 0;
        int tasksFailed = 0;
    };
    
    static TeamMetrics analyzeTeam(const Team& team, float timeWindow = 60.0f);
    static void recordTaskCompletion(const std::string& teamId, const Task& task, bool success);
    static void recordCommunication(const std::string& teamId, int messageCount);

private:
    static std::map<std::string, TeamMetrics> teamHistory_;
};

} // namespace teamwork
} // namespace iapv