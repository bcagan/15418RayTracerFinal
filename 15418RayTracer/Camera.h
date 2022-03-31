#pragma once
class Camera
{
public:
	Camera(Vec3 pos = Vec3(0.f), Transform t = Transform()) {};
	Vec3 pos;
	Transform transform;
	float lensDistance = 1.f; //Zoom
	float vFov = 90; //Fov is defined by vertical, in degrees (or, vFov/2 up, vFov/2 down)
	float sizeY = 1.f, sizeX = 1280.f /720.f; //Size of lens plane in worldspace
	int resY = 720, resX = 1280; //Resolution

	void focus(Vec3 pos, float dist, float theta, float phi); // Focus at camera a world space point, at an angle, at a distance
	//To be used possibly for orbital camera controls
	Ray castRay(int pixX, int pixY);
};

