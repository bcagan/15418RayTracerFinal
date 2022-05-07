#pragma once

#include <vector>
#include "Defined.h"
#include "Ray.h"
#include "Object.h"
#include "Camera.h"

class Scene
{
public:
	void addObj(Object o);
	void addObjseq(Object* o);
	void render(); //Image stored in camera
	Color3 renderC(Ray r, int numBounces);
	bool intersect(Ray ray, Hit& hit);
	Color3 background = Color3(0);
	float sampleNum() { return numSamples; }

	Camera cam;
	std::vector<Object> sceneObjs;
	std::vector<Object*> sceneObjss;
private:
	float numSamples = 10; //Default 10 samples per bounce
	//To make the code easier, I'm going to assume that the objects are always in world space, and that the rays will be transformed from camera to world space.
	
};


