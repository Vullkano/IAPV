#include "math_utils.hpp"
#include <random>
#include <cmath>

namespace iapv {
namespace common {

// Static random number generator
static std::random_device rd;
static std::mt19937 gen(rd());

float Random::range(float min, float max) {
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

int Random::range(int min, int max) {
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

Vector2D Random::randomVector2D(float minMagnitude, float maxMagnitude) {
    float angle = range(0.0f, 2.0f * M_PI);
    float magnitude = range(minMagnitude, maxMagnitude);
    return Vector2D(magnitude * std::cos(angle), magnitude * std::sin(angle));
}

Vector3D Random::randomVector3D(float minMagnitude, float maxMagnitude) {
    // Generate random direction using spherical coordinates
    float theta = range(0.0f, 2.0f * M_PI);  // azimuthal angle
    float phi = range(0.0f, M_PI);           // polar angle
    float magnitude = range(minMagnitude, maxMagnitude);
    
    float sinPhi = std::sin(phi);
    return Vector3D(
        magnitude * sinPhi * std::cos(theta),
        magnitude * sinPhi * std::sin(theta),
        magnitude * std::cos(phi)
    );
}

} // namespace common
} // namespace iapv