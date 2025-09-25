#pragma once

#include "../common/agent.hpp"
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <functional>
#include <memory>
#include <set>

namespace iapv {
namespace communication {

// Message types for agent communication
enum class MessageType {
    Verbal,
    Gesture,
    Emotional,
    Positional,
    Command,
    Query,
    Response
};

// Base message structure
struct Message {
    std::string senderId;
    std::string receiverId; // Empty for broadcast
    MessageType type;
    std::string content;
    std::map<std::string, float> parameters; // Additional data
    float timestamp;
    float priority = 1.0f;
    
    Message(const std::string& sender, const std::string& receiver, MessageType t, const std::string& c)
        : senderId(sender), receiverId(receiver), type(t), content(c), timestamp(0.0f) {}
};

// Communication channel for managing message flow
class CommunicationChannel {
public:
    CommunicationChannel(float range = 50.0f) : range_(range) {}
    
    void sendMessage(const Message& message, const common::Vector3D& senderPosition);
    void broadcastMessage(const Message& message, const common::Vector3D& senderPosition);
    
    std::vector<Message> getMessagesFor(const std::string& agentId, const common::Vector3D& position);
    void update(float deltaTime);
    
    void setRange(float range) { range_ = range; }
    float getRange() const { return range_; }

private:
    struct PendingMessage {
        Message message;
        common::Vector3D senderPosition;
        float delay;
    };
    
    std::queue<PendingMessage> messageQueue_;
    std::map<std::string, std::vector<Message>> agentInboxes_;
    float range_;
    
    bool isInRange(const common::Vector3D& sender, const common::Vector3D& receiver) const;
};

// Emotional state representation
enum class EmotionalState {
    Neutral,
    Happy,
    Sad,
    Angry,
    Fearful,
    Surprised,
    Disgusted,
    Excited,
    Calm,
    Aggressive
};

// Gesture types for non-verbal communication
enum class GestureType {
    Wave,
    Point,
    Nod,
    Shake_Head,
    Thumbs_Up,
    Thumbs_Down,
    Stop,
    Come_Here,
    Go_Away,
    Warning
};

// Non-verbal communication component
class NonVerbalCommunication {
public:
    struct Gesture {
        GestureType type;
        common::Vector3D direction; // For directional gestures like pointing
        float intensity = 1.0f;
        float duration = 1.0f;
        
        Gesture(GestureType t) : type(t), direction(0, 0, 0) {}
    };
    
    struct FacialExpression {
        EmotionalState emotion;
        float intensity = 1.0f;
        float duration = 2.0f;
        
        FacialExpression(EmotionalState e) : emotion(e) {}
    };
    
    void performGesture(const Gesture& gesture);
    void setFacialExpression(const FacialExpression& expression);
    void setBodyLanguage(const std::string& posture, float confidence = 1.0f);
    
    Gesture getCurrentGesture() const { return currentGesture_; }
    FacialExpression getCurrentExpression() const { return currentExpression_; }
    std::string getCurrentBodyLanguage() const { return currentBodyLanguage_; }
    
    void update(float deltaTime);

private:
    Gesture currentGesture_ = Gesture(GestureType::Wave);
    FacialExpression currentExpression_ = FacialExpression(EmotionalState::Neutral);
    std::string currentBodyLanguage_ = "relaxed";
    
    float gestureTimer_ = 0.0f;
    float expressionTimer_ = 0.0f;
    float bodyLanguageConfidence_ = 1.0f;
};

// Language understanding and generation
class LanguageProcessor {
public:
    struct Intent {
        std::string action;
        std::map<std::string, std::string> parameters;
        float confidence;
    };
    
    Intent parseMessage(const std::string& message);
    std::string generateResponse(const Intent& intent, const std::map<std::string, std::string>& context);
    
    void addKeyword(const std::string& keyword, const std::string& action);
    void addResponseTemplate(const std::string& action, const std::string& templateStr);

private:
    std::map<std::string, std::string> keywords_;
    std::map<std::string, std::vector<std::string>> responseTemplates_;
    
    std::vector<std::string> tokenize(const std::string& text);
    std::string replaceTemplate(const std::string& templateStr, const std::map<std::string, std::string>& params);
};

// Communicating agent that can send and receive messages
class CommunicatingAgent : public common::Agent {
public:
    CommunicatingAgent(const std::string& id, CommunicationChannel* channel, const common::Vector3D& position = common::Vector3D());
    
    void update(float deltaTime) override;
    
    // Verbal communication
    void say(const std::string& message, const std::string& targetId = "");
    void shout(const std::string& message); // Increased range
    void whisper(const std::string& message, const std::string& targetId);
    
    // Non-verbal communication
    void performGesture(GestureType gesture, const common::Vector3D& direction = common::Vector3D());
    void setEmotionalState(EmotionalState emotion, float intensity = 1.0f);
    void setBodyLanguage(const std::string& posture, float confidence = 1.0f);
    
    // Message handling
    void addMessageHandler(MessageType type, std::function<void(const Message&)> handler);
    
    // Social behavior
    void startConversation(const std::string& targetId);
    void endConversation(const std::string& targetId);
    bool isInConversation() const { return !currentConversationPartner_.empty(); }
    
    // Information sharing
    void shareInformation(const std::string& info, const std::string& targetId = "");
    void requestInformation(const std::string& query, const std::string& targetId);

protected:
    CommunicationChannel* channel_;
    NonVerbalCommunication nonVerbal_;
    LanguageProcessor languageProcessor_;
    
    std::map<MessageType, std::function<void(const Message&)>> messageHandlers_;
    std::string currentConversationPartner_;
    std::queue<std::string> conversationQueue_;
    
    virtual void processMessage(const Message& message);
    virtual void onConversationStart(const std::string& partnerId) {}
    virtual void onConversationEnd(const std::string& partnerId) {}
};

// Social network for tracking relationships
class SocialNetwork {
public:
    struct Relationship {
        float trust = 0.5f;      // 0 = no trust, 1 = full trust
        float affection = 0.5f;  // 0 = dislike, 1 = like
        float respect = 0.5f;    // 0 = no respect, 1 = full respect
        int interactions = 0;
        float lastInteraction = 0.0f;
    };
    
    void updateRelationship(const std::string& agent1, const std::string& agent2, 
                          float trustDelta, float affectionDelta, float respectDelta);
    
    Relationship getRelationship(const std::string& agent1, const std::string& agent2) const;
    std::vector<std::string> getFriends(const std::string& agentId, float threshold = 0.7f) const;
    std::vector<std::string> getEnemies(const std::string& agentId, float threshold = 0.3f) const;
    
    void recordInteraction(const std::string& agent1, const std::string& agent2, float currentTime);

private:
    std::map<std::pair<std::string, std::string>, Relationship> relationships_;
    
    std::pair<std::string, std::string> makeKey(const std::string& a, const std::string& b) const;
};

// Communication patterns and protocols
class CommunicationProtocol {
public:
    enum class ProtocolType {
        DirectMessage,
        Broadcast,
        Gossip,
        Hierarchy,
        Emergency
    };
    
    virtual ~CommunicationProtocol() = default;
    virtual void handleMessage(const Message& message, CommunicatingAgent* agent) = 0;
    virtual bool shouldForward(const Message& message, const std::string& agentId) const = 0;
};

class GossipProtocol : public CommunicationProtocol {
public:
    void handleMessage(const Message& message, CommunicatingAgent* agent) override;
    bool shouldForward(const Message& message, const std::string& agentId) const override;
    
    void setGossipProbability(float probability) { gossipProbability_ = probability; }

private:
    float gossipProbability_ = 0.3f;
    std::map<std::string, std::set<std::string>> messageHistory_; // Track what messages each agent has seen
};

} // namespace communication
} // namespace iapv