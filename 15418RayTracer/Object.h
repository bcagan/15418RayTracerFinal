#pragma once
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
	Vec3 pos;
	Vec3 size;
	bool hit(Ray ray);
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
		Mat =  defaultMaterial;
	}
	Vec3 pos;
	BBox bbox;
	Material Mat;
	float size;
	Transform t;

	bool hit(Ray ray) override {
		return bbox.hit(ray);
	};
};