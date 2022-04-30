#pragma once
#include "Defined.h"
#include "Material.h"
#include "Ray.h"
#include "Transform.h"

class BBox {
public:
	BBox(Vec3 minn, Vec3 maxx) : min(minn), max(maxx) {}
	BBox() : min(Vec3(0.f)), max(Vec3(1.f)) {}
	Vec3 min;
	Vec3 max;
	bool hit( Ray& ray, Hit& hit);
};

class Object
{
public:
	Material Mat;
	BBox bbox;
	virtual bool hit( Ray& ray, Hit& hit) {
		printf("generic\n");
		return false;
	}//I assumne info of the material will be populated in scene intersection func
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
		bbox = BBox(p - (s / 2), p + (s / 2));
		//Set bbox
	}
	Cube() {
		pos = Vec3(0.f);
		size = 1.f;
		bbox = BBox(Vec3(-1.f), Vec3(1.f));
		//set bbox
	}
	Vec3 pos;
	float size;
	Transform t;

	bool hit(Ray &ray, Hit& hit) override {
		Hit temp;
		if (bbox.hit(ray, temp)) {
			if (temp.t < ray.maxt) {
				hit.t = temp.t;
				hit.uv = temp.uv;
				hit.Mat = Mat;
				Vec3 normVec = vecNormalize((ray.o + hit.t * ray.d) - pos); 
				if (abs(normVec.x) > abs(normVec.y) && abs(normVec.x) > abs(normVec.z)){
					if (normVec.x < 0) hit.normG = Vec3(-1.f, 0.f, 0.f);
					else hit.normG = Vec3(1.f, 0.f, 0.f);
				}
				else if (abs(normVec.y) > abs(normVec.x) && abs(normVec.y) > abs(normVec.z)){
					if (normVec.y < 0) hit.normG = Vec3(0.f, -1.f, 0.f);
					else hit.normG = Vec3(0.f, 1.f, 0.f);
				}
				else {
					if (normVec.z < 0) hit.normG = Vec3(0.f, 0.f, -1.f);
					else hit.normG = Vec3(0.f, 0.f, 1.f);
				}
				hit.normS = hit.normG;
				hit.uv = Vec2(0.f);//Not doing right now
				ray.maxt = temp.t;
				return true;
			}
		}
		return false;
	};
};

class Sphere : public Object {
public:
	Sphere(Vec3 c, float r) : radius(r) {
		t.pos = c; 
		bbox = BBox(c - r, c + r);
	}

	Sphere(){
		t.pos = Vec3(0.f);
		bbox = BBox(Vec3(-1.f), Vec3(1.f));
	}

	Transform t;
	float radius = 1.f;

	bool hit( Ray& ray, Hit& hit) override;
};