#pragma once
#ifndef OBJECT_H
#define OBJECT_H
#include "Defined.h"
#include "Material.h"
#include "Ray.h"
#include "Transform.h"

class Object
{
public:
	virtual bool hit(Ray ray);//I assumne info of the material will be populated in scene intersection func
};

class BBox{
public:
	BBox(Vec3 minn, Vec3 maxx) : min(minn), max(maxx) {}
	Vec3 min;
	Vec3 max;
	bool hit(const Ray &ray);
};


//Proposed default object types:
//Cube and sphere (easy intersection)
//Triangles (so we can add meshes later, I can do the mesh hit func if we do)
//Meshes ^^^

class Cube : public Object  {
public:
	Cube(Vec3 p, float s, Transform t =Transform()) {
		pos = p;
		size = s;
	}
	Vec3 pos;
	BBox bbox;
	Material Mat;
	float size;
	Transform t;

	bool hit(const Ray &ray) override {
		return bbox.hit(ray);
	};
};

class Sphere : public Object {
public:
	Sphere(Vec3 c, float r) : radius(r), center(c) {}

	BBox bbox;
	Material Mat;
	float size;
	Transform t;

	bool hit(const Ray &ray);
}
