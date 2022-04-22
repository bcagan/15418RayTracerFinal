#pragma once

#include "glm/glm.hpp"
#include <iostream>
#include <io.h>
#define vecNormalize glm::normalize

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
		r = round(v.x*255.f);
		g = round(v.y*255.f);
		b = round(v.z*255.f);
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

//Note on coordinates:

//Camera normally points along -z in camera space. For convinience then, world space will use right hand rule if need be, and have y be up
//Spherical coordinates: Vertical: phi [0,pi], theta [0,2pi]

