#include "Camera.h"
#include "Ray.h"
#include "Defined.h"
#include "Transform.h"
#include "Object.h"
#include "Scene.h"
#include <random>



Ray Camera::castRay(int pixX, int pixY) {
	Ray ret(Vec3(0.f),Vec3(1.f));
	float sizeY = 2.f * lensDistance * tan(vFov);
	float sizeX = (float)resX / (float)resY * sizeY;
	float minX = (float)pixX / (float)resX  ;
	float maxX = (float)(pixX + 1) / (float)resX * sizeX - sizeX / 2.f;
	float minY = (float)pixX / (float)resX * sizeX - sizeX / 2.f;
	float maxY = (float)(pixX + 1) / (float)resX * sizeX - sizeX / 2.f;
	float x = randf()*(maxX - minX) + minX;
	float y = randf() * (maxY - minY) + minY;
	float z = -lensDistance; //Camera space always faces -z
	ret.d = vecNormalize(Vec3(x, y, z));
	ret.o = Vec3(0.f);


	return ret.transformRay(transform); //Ray shot out in world space
}

