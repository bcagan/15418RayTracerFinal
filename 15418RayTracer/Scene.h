#pragma once

#include <vector>
#include "Defined.h"
#include "Ray.h"
#include "Object.h"
#include "Camera.h"

class Scene
{
public:
	Object addObj(Object o);
	void render(); //Image stored in camera
	bool intersect(Ray ray, Hit& hit);

	Camera cam;
private:
	float numSamples = 5; //Default 5 samples per bounce
	//To make the code easier, I'm going to assume that the objects are always in world space, and that the rays will be transformed from camera to world space.
	std::vector<Object> sceneObjs;
};


