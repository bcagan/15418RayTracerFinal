#pragma once
#include "glm/glm.hpp" 
#include "Defined.h"
#include "Material.h"
#include "GLFW/glfw3.h"
#include "Transform.h"

class Ray
{
public:
	Vec3 o; //Origin
	Vec3 d; //ray direction - For consistency's sake I'm going to have as treat this as always normalized. 1 d's worth = 1 time's worth of ray travel
	float mint, maxt; //min distance of ray intersection, max distance of ray intersection
	//Example: mint may have a use I dont remember but we might not use it however when navigating the bvh tree, if we know we intersect at t = 10,
	//then any other intersection at t >= 10 can be ruled out

	Ray transformRay(Transform transform);

	Ray(Vec3 origin, Vec3 dir) {
		mint = EPSILON;
		maxt = INFINITY;
		o = origin;
		d = dir;
	};

	Ray() {
		Vec3 origin(0.f);
		Vec3 dir(1.f);
		mint = EPSILON;
		maxt = INFINITY;
		o = origin;
		d = dir;
	};
private:
	float numBounces = 15; //Default 15 recursive bounces max
};

//Glm has most of the structures we need to do sequential matrix and vecter operations, I defined a lot of useful thing in defined.h

class Hit
{
public:

	Material Mat;
	float t; //t when ray hits
	Vec3 normS; //Surface normal
	Vec3 normG; //Geometric normal
	Vec2 uv; //uv for texture if we decide to go that route, I dont think we will

	Color3 emitted() {
		return Mat.emitted;
	};
	Color3 albedo() {
		return Mat.albedo;
	};
	Vec3 bounce(Ray out); //Return ray bouncing (in opposite direction) into material that will result in out
};