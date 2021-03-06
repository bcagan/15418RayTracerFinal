#pragma once

#include "Defined.h"
#include "Transform.h"
#include "Ray.h"
#include <vector>

class Camera{
public:
	Camera() {
		image = std::vector<std::vector<Color3>>(resY);
		for (int yi = 0; yi < resY; yi++) {
			image[yi] = std::vector<Color3>(resX);
		}
		img = std::vector<Color3>(resX * resY);
	};
	Camera(float x, float y) {
		resX = x;
		resY = y;
		image = std::vector<std::vector<Color3>>(resY);
		for (int yi = 0; yi < resY; yi++) {
			image[yi] = std::vector<Color3>(resX);
		}
		img = std::vector<Color3>(resX * resY);
	};
	Transform transform;
	float lensDistance = 1.f; //Zoom
	float vFov = 90; //Fov is defined by vertical, in degrees (or, vFov/2 up, vFov/2 down)
	int resY = 720, resX = 1280; //Resolution

	void focus(Vec3 pos, float dist, float theta, float phi); // Focus at camera a world space point, at an angle, at a distance
	//To be used possibly for orbital camera controls
	Ray castRay(int pixX, int pixY);

	std::vector<std::vector<Color3>> image;
	std::vector<Color3> img;
};

