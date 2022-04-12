#pragma once
#include "Defined.h"
#include "Material.h"
#include "Ray.h"
#include "Transform.h"

class Object
{
public:
	BBox bbox;
	virtual bool hit(const Ray& ray, Hit& hit);//I assumne info of the material will be populated in scene intersection func
};

class BBox{
public:
	BBox(Vec3 minn, Vec3 maxx) : min(minn), max(maxx) {}
	BBox() : min(Vec3(0.f)), max(Vec3(1.f)) {}
	Vec3 min;
	Vec3 max;
	bool hit(const Ray &ray, Hit& hit);
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
	Material Mat;
	float size;
	Transform t;

	bool hit(const Ray &ray, Hit& hit) override {
		Hit temp;
		if (bbox.hit(ray, temp)) {
			if (temp.t < ray.maxt) {
				hit = temp;
				hit.Mat = Mat;
				hit.normG = glm::normalize((ray.o + hit.t * ray.d) - pos);
				hit.normS = hit.normG;
				hit.uv = Vec2(0.f);//Not doing right now
			}
			return true;
		}
		return false;
	};
};

class Sphere : public Object {
public:
	Sphere(Vec3 c, float r) : radius(r) {
		t.pos = c; 
	}

	Material Mat;
	float size;
	Transform t;
	float radius;

	bool hit(const Ray& ray, Hit& hit);
};