#include "messaging.hpp"
#include "../common/math_utils.hpp"
#include <algorithm>
#include <sstream>
#include <set>

namespace iapv {
namespace communication {

// CommunicationChannel implementation
void CommunicationChannel::sendMessage(const Message& message, const common::Vector3D& senderPosition) {
    PendingMessage pending = {message, senderPosition, 0.1f}; // Small delay for realism
    messageQueue_.push(pending);
}

void CommunicationChannel::broadcastMessage(const Message& message, const common::Vector3D& senderPosition) {
    Message broadcastMsg = message;
    broadcastMsg.receiverId = ""; // Empty means broadcast
    sendMessage(broadcastMsg, senderPosition);
}

std::vector<Message> CommunicationChannel::getMessagesFor(const std::string& agentId, const common::Vector3D& position) {
    std::vector<Message> messages;
    auto it = agentInboxes_.find(agentId);
    if (it != agentInboxes_.end()) {
        messages = it->second;
        it->second.clear(); // Clear after reading
    }
    return messages;
}

void CommunicationChannel::update(float deltaTime) {
    std::queue<PendingMessage> remainingMessages;
    
    while (!messageQueue_.empty()) {
        PendingMessage pending = messageQueue_.front();
        messageQueue_.pop();
        
        pending.delay -= deltaTime;
        
        if (pending.delay <= 0) {
            // Deliver message
            if (pending.message.receiverId.empty()) {
                // Broadcast to all agents in range
                for (auto& [agentId, inbox] : agentInboxes_) {
                    if (agentId != pending.message.senderId) {
                        // Check if message should be delivered (in range, etc.)
                        inbox.push_back(pending.message);
                    }
                }
            } else {
                // Direct message
                agentInboxes_[pending.message.receiverId].push_back(pending.message);
            }
        } else {
            remainingMessages.push(pending);
        }
    }
    
    messageQueue_ = remainingMessages;
}

bool CommunicationChannel::isInRange(const common::Vector3D& sender, const common::Vector3D& receiver) const {
    return (sender - receiver).magnitude() <= range_;
}

// NonVerbalCommunication implementation
void NonVerbalCommunication::performGesture(const Gesture& gesture) {
    currentGesture_ = gesture;
    gestureTimer_ = gesture.duration;
}

void NonVerbalCommunication::setFacialExpression(const FacialExpression& expression) {
    currentExpression_ = expression;
    expressionTimer_ = expression.duration;
}

void NonVerbalCommunication::setBodyLanguage(const std::string& posture, float confidence) {
    currentBodyLanguage_ = posture;
    bodyLanguageConfidence_ = confidence;
}

void NonVerbalCommunication::update(float deltaTime) {
    gestureTimer_ -= deltaTime;
    expressionTimer_ -= deltaTime;
    
    // Reset to neutral states when timers expire
    if (gestureTimer_ <= 0 && currentGesture_.type != GestureType::Wave) {
        currentGesture_ = Gesture(GestureType::Wave); // Default neutral gesture
    }
    
    if (expressionTimer_ <= 0 && currentExpression_.emotion != EmotionalState::Neutral) {
        currentExpression_ = FacialExpression(EmotionalState::Neutral);
    }
}

// LanguageProcessor implementation
LanguageProcessor::Intent LanguageProcessor::parseMessage(const std::string& message) {
    Intent intent;
    intent.confidence = 0.0f;
    
    std::vector<std::string> tokens = tokenize(message);
    
    // Simple keyword matching
    for (const auto& token : tokens) {
        auto it = keywords_.find(token);
        if (it != keywords_.end()) {
            intent.action = it->second;
            intent.confidence = 0.8f; // Basic confidence
            break;
        }
    }
    
    // Extract parameters (very basic implementation)
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == "to" && i + 1 < tokens.size()) {
            intent.parameters["target"] = tokens[i + 1];
        }
        if (tokens[i] == "at" && i + 1 < tokens.size()) {
            intent.parameters["location"] = tokens[i + 1];
        }
    }
    
    if (intent.action.empty()) {
        intent.action = "unknown";
        intent.confidence = 0.1f;
    }
    
    return intent;
}

std::string LanguageProcessor::generateResponse(const Intent& intent, const std::map<std::string, std::string>& context) {
    auto it = responseTemplates_.find(intent.action);
    if (it != responseTemplates_.end() && !it->second.empty()) {
        // Pick a random template
        int index = common::Random::range(0, static_cast<int>(it->second.size()) - 1);
        std::string templateStr = it->second[index];
        
        // Combine intent parameters with context
        std::map<std::string, std::string> allParams = intent.parameters;
        allParams.insert(context.begin(), context.end());
        
        return replaceTemplate(templateStr, allParams);
    }
    
    return "I don't understand.";
}

void LanguageProcessor::addKeyword(const std::string& keyword, const std::string& action) {
    keywords_[keyword] = action;
}

void LanguageProcessor::addResponseTemplate(const std::string& action, const std::string& templateStr) {
    responseTemplates_[action].push_back(templateStr);
}

std::vector<std::string> LanguageProcessor::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    
    while (iss >> token) {
        // Convert to lowercase
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        // Remove punctuation
        token.erase(std::remove_if(token.begin(), token.end(), ::ispunct), token.end());
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

std::string LanguageProcessor::replaceTemplate(const std::string& templateStr, const std::map<std::string, std::string>& params) {
    std::string result = templateStr;
    
    for (const auto& [key, value] : params) {
        std::string placeholder = "{" + key + "}";
        size_t pos = result.find(placeholder);
        if (pos != std::string::npos) {
            result.replace(pos, placeholder.length(), value);
        }
    }
    
    return result;
}

// CommunicatingAgent implementation
CommunicatingAgent::CommunicatingAgent(const std::string& id, CommunicationChannel* channel, const common::Vector3D& position)
    : Agent(id, position), channel_(channel) {
    
    // Setup basic language processing
    languageProcessor_.addKeyword("hello", "greet");
    languageProcessor_.addKeyword("hi", "greet");
    languageProcessor_.addKeyword("goodbye", "farewell");
    languageProcessor_.addKeyword("bye", "farewell");
    languageProcessor_.addKeyword("help", "request_help");
    languageProcessor_.addKeyword("follow", "follow");
    languageProcessor_.addKeyword("stop", "stop");
    languageProcessor_.addKeyword("go", "move");
    languageProcessor_.addKeyword("attack", "attack");
    languageProcessor_.addKeyword("retreat", "retreat");
    
    languageProcessor_.addResponseTemplate("greet", "Hello there!");
    languageProcessor_.addResponseTemplate("greet", "Hi! How are you?");
    languageProcessor_.addResponseTemplate("farewell", "Goodbye!");
    languageProcessor_.addResponseTemplate("farewell", "See you later!");
    languageProcessor_.addResponseTemplate("request_help", "I'll help you.");
    languageProcessor_.addResponseTemplate("request_help", "What do you need?");
    languageProcessor_.addResponseTemplate("follow", "I'll follow you.");
    languageProcessor_.addResponseTemplate("stop", "Stopping now.");
    languageProcessor_.addResponseTemplate("unknown", "I don't understand.");
    languageProcessor_.addResponseTemplate("unknown", "Could you repeat that?");
}

void CommunicatingAgent::update(float deltaTime) {
    nonVerbal_.update(deltaTime);
    
    // Process incoming messages
    std::vector<Message> messages = channel_->getMessagesFor(getId(), getPosition());
    for (const auto& message : messages) {
        processMessage(message);
    }
}

void CommunicatingAgent::say(const std::string& message, const std::string& targetId) {
    Message msg(getId(), targetId, MessageType::Verbal, message);
    channel_->sendMessage(msg, getPosition());
}

void CommunicatingAgent::shout(const std::string& message) {
    Message msg(getId(), "", MessageType::Verbal, message);
    msg.parameters["volume"] = 2.0f; // Increased range
    channel_->broadcastMessage(msg, getPosition());
}

void CommunicatingAgent::whisper(const std::string& message, const std::string& targetId) {
    Message msg(getId(), targetId, MessageType::Verbal, message);
    msg.parameters["volume"] = 0.5f; // Reduced range
    channel_->sendMessage(msg, getPosition());
}

void CommunicatingAgent::performGesture(GestureType gesture, const common::Vector3D& direction) {
    NonVerbalCommunication::Gesture g(gesture);
    g.direction = direction;
    nonVerbal_.performGesture(g);
    
    // Also send as message for other agents to see
    Message msg(getId(), "", MessageType::Gesture, std::to_string(static_cast<int>(gesture)));
    msg.parameters["dir_x"] = direction.x;
    msg.parameters["dir_y"] = direction.y;
    msg.parameters["dir_z"] = direction.z;
    channel_->broadcastMessage(msg, getPosition());
}

void CommunicatingAgent::setEmotionalState(EmotionalState emotion, float intensity) {
    NonVerbalCommunication::FacialExpression expr(emotion);
    expr.intensity = intensity;
    nonVerbal_.setFacialExpression(expr);
    
    // Broadcast emotional state
    Message msg(getId(), "", MessageType::Emotional, std::to_string(static_cast<int>(emotion)));
    msg.parameters["intensity"] = intensity;
    channel_->broadcastMessage(msg, getPosition());
}

void CommunicatingAgent::setBodyLanguage(const std::string& posture, float confidence) {
    nonVerbal_.setBodyLanguage(posture, confidence);
}

void CommunicatingAgent::addMessageHandler(MessageType type, std::function<void(const Message&)> handler) {
    messageHandlers_[type] = handler;
}

void CommunicatingAgent::startConversation(const std::string& targetId) {
    if (!isInConversation()) {
        currentConversationPartner_ = targetId;
        onConversationStart(targetId);
        say("Hello! I'd like to talk.", targetId);
    }
}

void CommunicatingAgent::endConversation(const std::string& targetId) {
    if (currentConversationPartner_ == targetId) {
        say("Goodbye!", targetId);
        onConversationEnd(targetId);
        currentConversationPartner_.clear();
    }
}

void CommunicatingAgent::shareInformation(const std::string& info, const std::string& targetId) {
    Message msg(getId(), targetId, MessageType::Response, "INFO: " + info);
    if (targetId.empty()) {
        channel_->broadcastMessage(msg, getPosition());
    } else {
        channel_->sendMessage(msg, getPosition());
    }
}

void CommunicatingAgent::requestInformation(const std::string& query, const std::string& targetId) {
    Message msg(getId(), targetId, MessageType::Query, query);
    channel_->sendMessage(msg, getPosition());
}

void CommunicatingAgent::processMessage(const Message& message) {
    // Call registered handler if available
    auto it = messageHandlers_.find(message.type);
    if (it != messageHandlers_.end()) {
        it->second(message);
        return;
    }
    
    // Default processing based on message type
    switch (message.type) {
        case MessageType::Verbal: {
            auto intent = languageProcessor_.parseMessage(message.content);
            std::map<std::string, std::string> context;
            context["sender"] = message.senderId;
            std::string response = languageProcessor_.generateResponse(intent, context);
            
            if (intent.confidence > 0.5f) {
                say(response, message.senderId);
            }
            break;
        }
        case MessageType::Query: {
            // Respond to information requests
            say("I'll think about that.", message.senderId);
            break;
        }
        case MessageType::Emotional: {
            // React to emotional displays from others
            int emotionInt = std::stoi(message.content);
            EmotionalState emotion = static_cast<EmotionalState>(emotionInt);
            
            // Simple emotional contagion
            if (emotion == EmotionalState::Happy) {
                setEmotionalState(EmotionalState::Happy, 0.5f);
            } else if (emotion == EmotionalState::Fearful) {
                setEmotionalState(EmotionalState::Fearful, 0.3f);
            }
            break;
        }
        default:
            break;
    }
}

// SocialNetwork implementation
void SocialNetwork::updateRelationship(const std::string& agent1, const std::string& agent2, 
                                      float trustDelta, float affectionDelta, float respectDelta) {
    auto key = makeKey(agent1, agent2);
    Relationship& rel = relationships_[key];
    
    rel.trust = std::max(0.0f, std::min(1.0f, rel.trust + trustDelta));
    rel.affection = std::max(0.0f, std::min(1.0f, rel.affection + affectionDelta));
    rel.respect = std::max(0.0f, std::min(1.0f, rel.respect + respectDelta));
}

SocialNetwork::Relationship SocialNetwork::getRelationship(const std::string& agent1, const std::string& agent2) const {
    auto key = makeKey(agent1, agent2);
    auto it = relationships_.find(key);
    return (it != relationships_.end()) ? it->second : Relationship{};
}

std::vector<std::string> SocialNetwork::getFriends(const std::string& agentId, float threshold) const {
    std::vector<std::string> friends;
    
    for (const auto& [key, rel] : relationships_) {
        if ((key.first == agentId || key.second == agentId) && 
            rel.affection >= threshold && rel.trust >= threshold) {
            friends.push_back(key.first == agentId ? key.second : key.first);
        }
    }
    
    return friends;
}

std::vector<std::string> SocialNetwork::getEnemies(const std::string& agentId, float threshold) const {
    std::vector<std::string> enemies;
    
    for (const auto& [key, rel] : relationships_) {
        if ((key.first == agentId || key.second == agentId) && 
            rel.affection <= threshold && rel.trust <= threshold) {
            enemies.push_back(key.first == agentId ? key.second : key.first);
        }
    }
    
    return enemies;
}

void SocialNetwork::recordInteraction(const std::string& agent1, const std::string& agent2, float currentTime) {
    auto key = makeKey(agent1, agent2);
    relationships_[key].interactions++;
    relationships_[key].lastInteraction = currentTime;
}

std::pair<std::string, std::string> SocialNetwork::makeKey(const std::string& a, const std::string& b) const {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

} // namespace communication
} // namespace iapv