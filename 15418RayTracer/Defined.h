#pragma once

#include "glm/glm.hpp"
#include <iostream>
#include <io.h>

Vec3 vecNormalize(Vec3 v) {
	glm::vec3 temp = glm::vec3(v.x, v.y, v.z);
	temp = glm::normalize(temp);
	return Vec3(temp.x, temp.y, temp.z);
}

Vec4 vecNormalize(Vec4 v) {
	glm::vec4 temp = glm::vec4(v.x, v.y, v.z, v.w);
	temp = glm::normalize(temp);
	return Vec4(temp.w, temp.x, temp.y, temp.z);
}

Vec2 vecNormalize(Vec2 v) {
	glm::vec2 temp = glm::vec2(v.x, v.y);
	temp = glm::normalize(temp);
	return Vec2(temp.x, temp.y);
}

struct Vec3 {
	Vec3() {
		x = 0.f;
		y = 0.f;
		z = 0.f;
	}
	Vec3(float a, float b, float c) {
		x = a; y = b; z = c;
	}
	Vec3(float a) {
		x = a; y = a; z = a;
	}
	float x;
	float y;
	float z;
};

struct Vec4 {
	Vec4() {
		x = 0.f;
		y = 0.f;
		z = 0.f;
		w = 0.f;
	}
	Vec4(float a, float b, float c, float d) {
		x = a; y = b; z = c, w = d;
	}
	Vec4(float a) {
		w = a;  x = a; y = a; z = a;
	}
	Vec4(Vec3 v, float a) {
		x = v.x; y = v.y; z = v.z;
		w = a;
	}
	float x;
	float y;
	float z;
	float w;
};

struct Vec2 {
	Vec2() {
		x = 0.f;
		y = 0.f;
	}
	Vec2(float a) {
		x = a; y = a;
	}
	Vec2(float a, float b) {
		x = a; y = b;
	}
	float x;
	float y;
};

#define Mat4x4 glm::mat4x4
#define Mat3x3 glm::mat3x3
#define Mat4x3 glm::mat4x3
#define EPSILON 0.00001f
#define PI 3.1415926f
//https://stackoverflow.com/questions/686353/random-float-number-generation
#define randf()  static_cast <float> (rand()) / static_cast <float> (RAND_MAX)
struct Color3
{
	Color3(unsigned char c) {
		r = c;
		g = c;
		b = c;
	};
	Color3(unsigned char a, unsigned char d, unsigned char c) {
		r = a;
		g = d;
		b = c;
	};
	Color3(Vec3 v) { //Assume float in [0.f,1.f]
		auto round = [](float f) {
			return floor(f + 0.5f);
		};
		r = round(v.x * 255.f);
		g = round(v.y * 255.f);
		b = round(v.z * 255.f);
	};
	Color3() {
		r = 255; g = 255; b = 255;
	};
	Vec3 toVec3() {
		return Vec3((float)r/255.f,(float)g/255.f,(float)b/255.f);
	};
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

inline void printVec3(Vec3 v) {
	std::cout << v.x << " " << v.y << " " << v.z;
}

inline Color3 normToColor(Vec3 n) {
	float x = n.x;
	float y = n.y;
	float z = n.z;
	if (x < 0) x = 0.5f;
	else if (x < 0.5) x = 0.f;
	if (y < 0) y = 0.5f;
	else if (y < 0.5f) y = 0.f;
	if (z < 0) z = 0.5f;
	else if (z < 0.5f) z = 0.f;
	return Color3(Vec3(x, y, z));
}

//Note on coordinates:

//Camera normally points along -z in camera space. For convinience then, world space will use right hand rule if need be, and have y be up
//Spherical coordinates: Vertical: phi [0,pi], theta [0,2pi]




inline Vec4 constVecMult(float a, Vec4 v) {
	glm::vec4 temp = a * glm::vec4(v.x, v.y, v.z, v.w);
	return Vec4(temp.w, temp.x, temp.y, temp.z);
}
inline Vec3 constVecMult(float a, Vec3 v) {
	glm::vec3 temp = (a * (glm::vec3(v.x, v.y, v.z)));
	return Vec3(temp.x, temp.y, temp.z);
}
inline Vec2 constVecMult(float a, Vec2 v) {
	glm::vec2 temp = a * glm::vec2(v.x, v.y);
	return Vec2(temp.x, temp.y);
}
inline Vec4 vecVecMult(Vec4 a, Vec4 v) {
	glm::vec4 temp = glm::vec4(a.x, a.y, a.z, a.w) * glm::vec4(v.x, v.y, v.z, v.w);
	return Vec4(temp.w, temp.x, temp.y, temp.z);
}
inline Vec3 vecVecMult(Vec3 a, Vec3 v) {
	glm::vec3 temp = (glm::vec3(a.x, a.y, a.z) * (glm::vec3(v.x, v.y, v.z)));
	return Vec3(temp.x, temp.y, temp.z);
}
inline Vec2 vecVecMult(Vec2 a, Vec2 v) {
	glm::vec2 temp = glm::vec2(a.x, a.y) * glm::vec2(v.x, v.y);
	return Vec2(temp.x, temp.y);
}
inline Vec4 vecVecAdd(Vec4 a, Vec4 v) {
	glm::vec4 temp = glm::vec4(a.x, a.y, a.z, a.w) + glm::vec4(v.x, v.y, v.z, v.w);
	return Vec4(temp.w, temp.x, temp.y, temp.z);
}
inline Vec3 vecVecAdd(Vec3 a, Vec3 v) {
	glm::vec3 temp = (glm::vec3(a.x, a.y, a.z) + (glm::vec3(v.x, v.y, v.z)));
	return Vec3(temp.x, temp.y, temp.z);
}
inline Vec2 vecVecAdd(Vec2 a, Vec2 v) {
	glm::vec2 temp = glm::vec2(a.x, a.y) + glm::vec2(v.x, v.y);
	return Vec2(temp.x, temp.y);
}
inline Vec4 vecVecSub(Vec4 a, Vec4 v) {
	glm::vec4 temp = glm::vec4(a.x, a.y, a.z, a.w) - glm::vec4(v.x, v.y, v.z, v.w);
	return Vec4(temp.w, temp.x, temp.y, temp.z);
}
inline Vec3 vecVecSub(Vec3 a, Vec3 v) {
	glm::vec3 temp = (glm::vec3(a.x, a.y, a.z) - (glm::vec3(v.x, v.y, v.z)));
	return Vec3(temp.x, temp.y, temp.z);
}
inline Vec2 vecVecSub(Vec2 a, Vec2 v) {
	glm::vec2 temp = glm::vec2(a.x, a.y) - glm::vec2(v.x, v.y);
	return Vec2(temp.x, temp.y);
}
inline Vec4 vecVecDiv(Vec4 a, Vec4 v) {
	glm::vec4 temp = glm::vec4(a.x, a.y, a.z, a.w) / glm::vec4(v.x, v.y, v.z, v.w);
	return Vec4(temp.w, temp.x, temp.y, temp.z);
}
inline Vec3 vecVecDiv(Vec3 a, Vec3 v) {
	glm::vec3 temp = (glm::vec3(a.x, a.y, a.z) / (glm::vec3(v.x, v.y, v.z)));
	return Vec3(temp.x, temp.y, temp.z);
}
inline Vec2 vecVecDiv(Vec2 a, Vec2 v) {
	glm::vec2 temp = glm::vec2(a.x, a.y) / glm::vec2(v.x, v.y);
	return Vec2(temp.x, temp.y);
}
inline float dot(Vec2 a, Vec2 b) {
	return a.x * b.x + a.y * b.y;
}
inline float dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline float dot(Vec4 a, Vec4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
