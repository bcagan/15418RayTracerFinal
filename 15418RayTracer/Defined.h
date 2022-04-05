#pragma once

#ifndef DEFINED_H
#define EDFINED_H
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
	Color3f(float c) {
		r = c;
		g = c;
		b = c;
	};
	Color3f(float a,float d,float c) {
		r = a;
		g = d;
		b = c;
	};
	Color3f() {
		r = 1.f; g = 1.f; b = 1.f;
	}
	float r;
	float g;
	float b;
};

//Note on coordinates:

//Camera normally points along -z in camera space. For convinience then, world space will use right hand rule if need be, and have y be up
//Spherical coordinates: Vertical: phi [0,pi], theta [0,2pi]

#define vecNormalize glm::normalize

#endif