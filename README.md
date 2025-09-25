# IAPV - Inteligência Artificial para Personagens Virtuais
# AI for Virtual Characters

A comprehensive C++ and Python framework for implementing AI behaviors in virtual characters, covering planning, decision making, learning, locomotion, navigation, teamwork, crowds, communication, and human-virtual character interaction.

## 🎯 Learning Outcomes

After studying and working with this project, you will understand:

- **Planning & Pathfinding**: A* algorithm, behavior trees, task planning
- **Decision Making**: Finite state machines, decision trees, utility-based AI
- **Learning**: Neural networks, Q-learning, genetic algorithms, imitation learning
- **Locomotion**: Steering behaviors, flocking, crowd simulation
- **Navigation**: Spatial reasoning, obstacle avoidance, flow fields
- **Teamwork**: Formation control, task allocation, coordination protocols
- **Communication**: Message passing, verbal/non-verbal signals, social networks
- **Human-VC Interaction**: Emotion recognition, adaptive responses, social AI

## 🏗️ Project Structure

```
IAPV/
├── src/
│   ├── cpp/                    # C++ implementations
│   │   ├── common/             # Core utilities and base classes
│   │   ├── planning/           # A* pathfinding, behavior trees
│   │   ├── decision_making/    # FSM, decision trees, utility systems
│   │   ├── learning/           # Neural networks, RL algorithms
│   │   ├── locomotion/         # Steering behaviors, animation
│   │   ├── navigation/         # Spatial reasoning, pathfinding
│   │   ├── teamwork/           # Coordination, formation control
│   │   ├── crowds/             # Flocking, crowd simulation
│   │   ├── communication/      # Messaging, social behavior
│   │   └── human_interaction/  # Emotion, response systems
│   └── python/                 # Python implementations
│       ├── common/             # Core utilities
│       ├── planning/           # Planning algorithms
│       ├── locomotion/         # Movement behaviors
│       └── ...                 # Mirror of C++ structure
├── examples/
│   ├── unity/                  # Unity integration examples
│   ├── unreal/                 # Unreal Engine examples
│   └── blender/                # Blender scripting examples
├── notebooks/                  # Jupyter notebooks for learning
├── demos/                      # Interactive demonstrations
├── tests/                      # Unit tests
├── docs/                       # Documentation
└── CMakeLists.txt             # Build configuration
```

## 🚀 Quick Start

### Prerequisites

**For C++:**
- CMake 3.12+
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Optional: Eigen3 for advanced math operations

**For Python:**
- Python 3.7+
- NumPy
- Pygame (for visual demos)
- Jupyter (for notebooks)

### Building C++ Components

```bash
# Clone the repository
git clone https://github.com/Vullkano/IAPV.git
cd IAPV

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j4

# Run tests (optional)
make test
```

### Running Python Demos

```bash
# Install Python dependencies
pip install numpy pygame jupyter matplotlib

# Run the visual demo
cd IAPV
python demos/visual_demo.py
```

### Interactive Demo Controls

- **1**: Flocking behavior demo
- **2**: Pathfinding demo  
- **3**: Seek/Flee behavior demo
- **R**: Reset current demo
- **Space**: Pause/unpause
- **Mouse**: Interact with agents (click to set targets)

## 📚 Core Modules

### 1. Planning System (`src/cpp/planning/`, `src/python/planning/`)

**A* Pathfinding**
```cpp
// C++ Example
GridWorld world(50, 50, 1.0f);
AStarPathfinder pathfinder(world);
auto path = pathfinder.findPath(start, goal);
```

```python
# Python Example
world = GridWorld(50, 50, 1.0)
pathfinder = AStarPathfinder(world)
path = pathfinder.find_path(start, goal)
```

**Behavior Trees**
```cpp
// C++ Example
auto sequence = std::make_unique<SequenceNode>();
sequence->addChild(std::make_unique<ConditionNode>(checkHealth));
sequence->addChild(std::make_unique<ActionNode>(moveToSafety));
BehaviorTree tree(std::move(sequence));
```

### 2. Decision Making (`src/cpp/decision_making/`)

**Finite State Machine**
```cpp
FiniteStateMachine fsm(agent);
fsm.addState("patrol", std::make_unique<PatrolState>(waypoints));
fsm.addState("chase", std::make_unique<ChaseState>(targetId));
fsm.setState("patrol");
```

**Utility System**
```cpp
UtilitySystem utility;
utility.addAction({"eat", eatUtility, eatAction});
utility.addAction({"sleep", sleepUtility, sleepAction});
utility.executeHighestUtilityAction(agent);
```

### 3. Learning System (`src/cpp/learning/`)

**Neural Network**
```cpp
NeuralNetwork network({10, 20, 10, 4}); // 10 inputs, 4 outputs
auto outputs = network.feedForward(inputs);
network.backpropagate(inputs, targets, 0.01f);
```

**Q-Learning**
```cpp
QLearning qlearning(numStates, numActions, 0.1f, 0.9f, 0.1f);
int action = qlearning.selectAction(state);
qlearning.updateQ(state, action, reward, nextState);
```

### 4. Locomotion System (`src/cpp/locomotion/`, `src/python/locomotion/`)

**Steering Behaviors**
```cpp
// C++ Example
SteeringController controller(agent, 10.0f, 5.0f);
controller.addBehavior(std::make_unique<SeekBehavior>(target));
controller.addBehavior(std::make_unique<SeparationBehavior>(3.0f));
controller.update(deltaTime, neighbors);
```

```python
# Python Example
agent = FlockingAgent("boid_1", Vector3D(0, 0, 0))
agent.add_steering_behavior(SeekBehavior(target_position))
agent.update(delta_time)
```

### 5. Crowd Simulation (`src/cpp/crowds/`)

**Flocking Behavior**
```cpp
CrowdSimulation crowd;
crowd.addBoid(std::make_unique<Boid>("boid_1", position));
crowd.setNeighborRadius(10.0f);
crowd.update(deltaTime);

// Analyze crowd behavior
auto pattern = EmergentBehaviorDetector().detectPattern(crowd.getBoids());
```

### 6. Communication System (`src/cpp/communication/`)

**Message Passing**
```cpp
CommunicationChannel channel(50.0f); // 50 unit range
CommunicatingAgent agent("agent1", &channel, position);

agent.say("Hello!", "agent2");
agent.performGesture(GestureType::Wave);
agent.setEmotionalState(EmotionalState::Happy, 0.8f);
```

**Social Networks**
```cpp
SocialNetwork network;
network.updateRelationship("agent1", "agent2", 0.1f, 0.2f, 0.1f);
auto friends = network.getFriends("agent1", 0.7f);
```

### 7. Teamwork System (`src/cpp/teamwork/`)

**Team Coordination**
```cpp
Team squad("alpha_squad", &channel);
squad.addMember(leader);
squad.addMember(soldier1);
squad.setFormation(FormationType::Wedge);
squad.moveToPosition(targetPosition);

Task patrol("patrol_area", TaskType::Patrol, areaCenter);
squad.assignTask(patrol);
```

## 🎮 Game Engine Integration

### Unity Integration

```csharp
// Unity C# wrapper example
public class AICharacterController : MonoBehaviour 
{
    private IntPtr agentPtr;
    
    void Start() 
    {
        agentPtr = CreateAgent(transform.position);
        AddSteeringBehavior(agentPtr, "seek", target.position);
    }
    
    void Update() 
    {
        UpdateAgent(agentPtr, Time.deltaTime);
        transform.position = GetAgentPosition(agentPtr);
    }
}
```

### Unreal Engine Integration

```cpp
// Unreal C++ component example
UCLASS()
class MYGAME_API UAICharacterComponent : public UActorComponent
{
    GENERATED_BODY()

private:
    std::unique_ptr<iapv::common::Agent> Agent;
    std::unique_ptr<iapv::locomotion::SteeringController> Controller;

public:
    virtual void TickComponent(float DeltaTime, ...) override;
    
    UFUNCTION(BlueprintCallable)
    void SetTarget(FVector Target);
};
```

### Blender Integration

```python
# Blender Python script example
import bpy
import sys
sys.path.append("/path/to/IAPV/src/python")

from common.math_utils import Vector3D
from locomotion.steering_behaviors import FlockingAgent

# Create AI agents for Blender objects
for obj in bpy.context.selected_objects:
    agent = FlockingAgent(obj.name, Vector3D(*obj.location))
    # Update object position based on AI behavior
```

## 📓 Jupyter Notebooks

Interactive learning notebooks are provided in the `notebooks/` directory:

1. **01_Basic_Concepts.ipynb** - Introduction to vector math and agents
2. **02_Steering_Behaviors.ipynb** - Locomotion and movement
3. **03_Pathfinding.ipynb** - A* algorithm and navigation
4. **04_Behavior_Trees.ipynb** - Decision making systems
5. **05_Learning_Systems.ipynb** - Neural networks and RL
6. **06_Flocking_Crowds.ipynb** - Emergent group behavior
7. **07_Communication.ipynb** - Agent interaction and messaging
8. **08_Teamwork.ipynb** - Coordination and cooperation

```bash
# Start Jupyter server
cd IAPV
jupyter notebook notebooks/
```

## 🧪 Running Tests

```bash
# C++ tests
cd build
make test

# Python tests
python -m pytest tests/python/
```

## 📈 Performance Considerations

### C++ Optimizations
- Use object pooling for frequently created/destroyed agents
- Implement spatial partitioning for neighbor finding
- Profile steering behavior calculations in large crowds
- Consider SIMD optimizations for vector operations

### Python Optimizations
- Use NumPy arrays for bulk vector operations
- Implement Cython extensions for performance-critical code
- Consider multiprocessing for independent agent updates
- Profile bottlenecks with cProfile

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Standards

**C++:**
- Follow Google C++ Style Guide
- Use modern C++17 features
- Document public APIs with Doxygen comments
- Include unit tests for new features

**Python:**
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Document modules and classes with docstrings
- Include unit tests with pytest

## 📖 Learning Resources

### Books
- "Artificial Intelligence for Games" by Ian Millington
- "Programming Game AI by Example" by Mat Buckland
- "Behavioral Mathematics for Game AI" by Dave Mark

### Papers
- "Steering Behaviors For Autonomous Characters" by Craig Reynolds
- "The Role of AI in Computer Games" by John Laird & Michael van Lent
- "Hierarchical Pathfinding with Real-Time Adaptive A*" by Koenig & Likhachev

### Online Resources
- [AI Game Programming Wisdom](http://www.aiwisdom.com/)
- [Game AI Pro Series](http://www.gameaipro.com/)
- [Red Blob Games](https://www.redblobgames.com/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Craig Reynolds for pioneering work on steering behaviors
- Sebastian Lague for excellent AI tutorials
- The game development community for sharing knowledge and techniques

## 📞 Support

- 📧 Email: [Contact the maintainers]
- 🐛 Issues: [GitHub Issues](https://github.com/Vullkano/IAPV/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Vullkano/IAPV/discussions)

---

*Built with ❤️ for the AI and game development community*
