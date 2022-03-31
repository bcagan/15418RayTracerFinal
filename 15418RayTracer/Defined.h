#pragma once

#define Vec3 glm::vec3
#define EPSILON 0.00001f
#define PI 3.1415926f
#define Vec2 glm::vec2

//Note on coordinates:

//Camera normally points along -z in camera space. For convinience then, world space will use right hand rule if need be, and have y be up
//Spherical coordinates: Vertical: phi [0,pi], theta [0,2pi]

struct Color3f
{
	float r;
	float g;
	float b;
};