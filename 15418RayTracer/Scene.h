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
	void render();
	bool intersect(Ray ray, Hit hit);

	Camera cam;
private:
	std::vector<Object> sceneObjs;
};

