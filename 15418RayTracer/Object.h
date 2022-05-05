#pragma once
#include "Defined.h"
#include "Material.h"
#include "Ray.h"
#include "Transform.h"

enum GeomType {
	gcube,
	gsphere
};

class BBox {
public:
	BBox(Vec3 minn, Vec3 maxx) : min(minn), max(maxx) {}
	BBox() : min(Vec3(0.f)), max(Vec3(1.f)) {}
	Vec3 min;
	Vec3 max;
	__device__ bool hit( Ray& ray, Hit& hit);
};

class Object
{
public:
	Material Mat;
	BBox bbox;
	Transform t;
	// added since hit is a host function and cannot be called from global functions in pathtrace.cu
	GeomType type;
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
		t.pos = p;
		size = s;
		type = gcube;
		bbox = BBox(vecVecAdd(p, Vec3(s / -2.f)), vecVecAdd(p , Vec3(s / 2.f)));
		//Set bbox
	}
	Cube() {
		t.pos = Vec3(0.f);
		size = 1.f;
		bbox = BBox(Vec3(-1.f), Vec3(1.f));
		//set bbox
	}

	float size;

	bool hit(Ray &ray, Hit& hit) override {
		Hit temp;
		if (bbox.hit(ray, temp)) {
			if (temp.t < ray.maxt) {
				hit.t = temp.t;
				hit.uv = temp.uv;
				hit.Mat = Mat;
				Vec3 normVec = vecNormalize(vecVecAdd((vecVecAdd(ray.o , constVecMult(hit.t , ray.d))), constVecMult(-1.f, t.pos))); 
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
		type = gsphere;
		t.radius = r;
		bbox = BBox(vecVecAdd(c,Vec3( - r)), vecVecAdd(c ,Vec3( r)));
	}

	Sphere(){
		t.pos = Vec3(0.f);
		t.radius = 1.f;
		bbox = BBox(Vec3(-1.f), Vec3(1.f));
	}

	float radius = 1.f;

	bool hit( Ray& ray, Hit& hit) override;
};