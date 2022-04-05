#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "Defined.h"
#include<vector>
#include"Ray.h"
#include "Transform.h"
class Camera
{
public:
	Camera() {
		image = std::vector<std::vector<Color3f>>(resY);
		for (int xi = 0; xi < resX; xi++) {
			image[xi] = std::vector<Color3f>(resX);
		}
	};
	Camera(float x, float y) {
		resX = x;
		resY = y;
		image = std::vector<std::vector<Color3f>>(resY);
		for (int xi = 0; xi < resX; xi++) {
			image[xi] = std::vector<Color3f>(resX);
		}

	};
	Transform transform;
	float lensDistance = 1.f; //Zoom
	float vFov = 90; //Fov is defined by vertical, in degrees (or, vFov/2 up, vFov/2 down)
	int resY = 720, resX = 1280; //Resolution

	void focus(Vec3 pos, float dist, float theta, float phi); // Focus at camera a world space point, at an angle, at a distance
	//To be used possibly for orbital camera controls
	Ray castRay(int pixX, int pixY);

	std::vector<std::vector<Color3f>> image;
};

#endif
