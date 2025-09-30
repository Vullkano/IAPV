#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <cmath>

namespace iapv {
namespace common {

// Basic vector math utilities
struct Vector2D {
    float x, y;
    
    Vector2D(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}
    
    Vector2D operator+(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }
    
    Vector2D operator-(const Vector2D& other) const {
        return Vector2D(x - other.x, y - other.y);
    }
    
    Vector2D operator*(float scalar) const {
        return Vector2D(x * scalar, y * scalar);
    }
    
    float magnitude() const {
        return std::sqrt(x * x + y * y);
    }
    
    Vector2D normalized() const {
        float mag = magnitude();
        if (mag > 0) {
            return Vector2D(x / mag, y / mag);
        }
        return Vector2D(0, 0);
    }
    
    float dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }
};

struct Vector3D {
    float x, y, z;
    
    Vector3D(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}
    
    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3D operator*(float scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }
    
    float magnitude() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    Vector3D normalized() const {
        float mag = magnitude();
        if (mag > 0) {
            return Vector3D(x / mag, y / mag, z / mag);
        }
        return Vector3D(0, 0, 0);
    }
    
    float dot(const Vector3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    Vector3D cross(const Vector3D& other) const {
        return Vector3D(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
};

// Time utilities
using TimePoint = std::chrono::steady_clock::time_point;
using Duration = std::chrono::duration<float>;

inline TimePoint now() {
    return std::chrono::steady_clock::now();
}

inline float deltaTime(TimePoint start, TimePoint end) {
    return std::chrono::duration<float>(end - start).count();
}

// Random utilities
class Random {
public:
    static float range(float min, float max);
    static int range(int min, int max);
    static Vector2D randomVector2D(float minMagnitude = 0.0f, float maxMagnitude = 1.0f);
    static Vector3D randomVector3D(float minMagnitude = 0.0f, float maxMagnitude = 1.0f);
};

} // namespace common
} // namespace iapv