#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cmath>
#include <iostream>

struct Vec2 {
    double x, y;

    Vec2 operator+(const Vec2& other) const { return {x + other.x, y + other.y}; }
    Vec2 operator-(const Vec2& other) const { return {x - other.x, y - other.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }

    Vec2& operator+=(const Vec2& other) { x += other.x; y += other.y; return *this; }
    Vec2& operator-=(const Vec2& other) { x -= other.x; y -= other.y; return *this; }

    double dot(const Vec2& other) const { return x * other.x + y * other.y; }
    double normSq() const { return x * x + y * y; }
    double norm() const { return std::sqrt(normSq()); }
};

inline Vec2 operator*(double s, const Vec2& v) { return {v.x * s, v.y * s}; }

struct Vec4 {
    double v[4];

    double& operator[](int i) { return v[i]; }
    const double& operator[](int i) const { return v[i]; }

    Vec4 operator+(const Vec4& other) const {
        return {v[0] + other.v[0], v[1] + other.v[1], v[2] + other.v[2], v[3] + other.v[3]};
    }
    Vec4 operator-(const Vec4& other) const {
        return {v[0] - other.v[0], v[1] - other.v[1], v[2] - other.v[2], v[3] - other.v[3]};
    }
    Vec4 operator*(double s) const {
        return {v[0] * s, v[1] * s, v[2] * s, v[3] * s};
    }
    Vec4 operator/(double s) const {
        return {v[0] / s, v[1] / s, v[2] / s, v[3] / s};
    }

    Vec4& operator+=(const Vec4& other) {
        v[0] += other.v[0]; v[1] += other.v[1]; v[2] += other.v[2]; v[3] += other.v[3];
        return *this;
    }
    Vec4& operator-=(const Vec4& other) {
        v[0] -= other.v[0]; v[1] -= other.v[1]; v[2] -= other.v[2]; v[3] -= other.v[3];
        return *this;
    }

    double dot(const Vec4& other) const {
        return v[0] * other.v[0] + v[1] * other.v[1] + v[2] * other.v[2] + v[3] * other.v[3];
    }
};

inline Vec4 operator*(double s, const Vec4& v) {
    return {v[0] * s, v[1] * s, v[2] * s, v[3] * s};
}

#endif
