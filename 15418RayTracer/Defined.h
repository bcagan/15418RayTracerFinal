#pragma once
#include "glm/glm.hpp"

#define Vec4 glm::vec4
#define Vec3 glm::vec3
#define Vec2 glm::vec2
#define Mat4x4 glm::mat4x4
#define Mat3x3 glm::mat3x3
#define Mat4x3 glm::mat4x3
#define EPSILON 0.00001f
#define PI 3.1415926f
//https://stackoverflow.com/questions/686353/random-float-number-generation
#define randf()  static_cast <float> (rand()) / static_cast <float> (RAND_MAX)
struct Color3f
{
	float r;
	float g;
	float b;
};

//Note on coordinates:

//Camera normally points along -z in camera space. For convinience then, world space will use right hand rule if need be, and have y be up
//Spherical coordinates: Vertical: phi [0,pi], theta [0,2pi]

Vec2 vecNormalize(Vec2 v) {
	return glm::normalize(v);
}
Vec3 vecNormalize(Vec3 v) {
	return glm::normalize(v);
}
Vec4 vecNormalize(Vec4 v) {
	return glm::normalize(v);
}
