#pragma once

#include "glm/glm.hpp"
#include <iostream>
#include <io.h>
struct Vec3 {
	Vec3() {
		x = 0.f;
		y = 0.f;
		z = 0.f;
	}
	Vec3(float a, float b, float c) {
		x = a; y = b; z = c;
	}
	Vec3(float a) {
		x = a; y = a; z = a;
	}
	float x;
	float y;
	float z;
	float get(int ind) {
		switch (ind)
		{
		case (0):
			return x;
		case(1):
			return y;
		default:
			return z;
		}
	}
	void set(int ind, float a) {
		switch (ind) {
		case(0):
			x = a;
		case(1):
			y = a;
		default:
			z = a;
		}
	}
};

struct Vec4 {
	Vec4() {
		x = 0.f;
		y = 0.f;
		z = 0.f;
		w = 0.f;
	}
	Vec4(float a, float b, float c, float d) {
		x = a; y = b; z = c, w = d;
	}
	Vec4(float a) {
		w = a;  x = a; y = a; z = a;
	}
	Vec4(Vec3 v, float a) {
		x = v.x; y = v.y; z = v.z;
		w = a;
	}
	float x;
	float y;
	float z;
	float w;
	float get(int ind) {
		switch (ind)
		{
		case (0):
			return x;
		case(1):
			return y;
		case(2):
			return z;
		default:
			return w;
		}
	}
	void set(int ind, float a) {
		switch (ind)
		{
		case (0):
			 x = a;
		case(1):
			y = a;
		case(2):
			z = a;
		default:
			w = a;
		}
	}
};

struct Vec2 {
	Vec2() {
		x = 0.f;
		y = 0.f;
	}
	Vec2(float a) {
		x = a; y = a;
	}
	Vec2(float a, float b) {
		x = a; y = b;
	}
	float x;
	float y;
	float get(int ind) {
		switch (ind)
		{
		case (0):
			return x;
		default:
			return y;
		}
	}
	void set(int ind, float a) {
		switch (ind)
		{
		case (0):
			x = a;
		default:
			y = a;
		}
	}
};
inline float vecNorm(Vec2 v) {
	return sqrt(v.x * v.x + v.y * v.y);
}
inline float vecNorm(Vec3 v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
inline float vecNorm(Vec4 v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

struct Mat3x3 {
	Mat3x3() {
		a = Vec3(0.f);
		b = Vec3(0.f);
		c = Vec3(0.f);
	}
	Mat3x3(Vec3 v1, Vec3 v2, Vec3 v3) {
		a = v1;
		b = v2;
		c = v3;
	}
	Mat3x3(float f) {
		a = Vec3(f);
		b = Vec3(f);
		c = Vec3(f);
	}

	Vec3 a;
	Vec3 b;
	Vec3 c;
	float get(int x, int y) {
		switch (x) {
		case(0):
			return a.get(y);
		case(1):
			return b.get(y);
		default:
			return c.get(y);
		}
	}
	void set(int x, int y, float z) {
		switch (x) {
		case(0):
			a.set(y, z);
		case(1):
			b.set(y, z);
		default:
			c.set(y, z);
		}
	}
};

struct Mat4x3 {
	Mat4x3(float f) {
		a = Vec3(f);
		b = Vec3(f);
		c = Vec3(f);
		d = Vec3(f);
	}
	Mat4x3() {
		a = Vec3(0.f);
		b = Vec3(0.f);
		c = Vec3(0.f);
		d = Vec3(0.f);
	}
	Mat4x3(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4) {
		a = v1;
		b = v2;
		c = v3;
		d = v4;
	}
	Mat4x3(Mat3x3 M) {
		a = M.a;
		b = M.b;
		c = M.c;
		d = Vec3(0.f);
	}

	Vec3 a;
	Vec3 b;
	Vec3 c;
	Vec3 d;
	float get(int x, int y) {
		switch (x) {
		case(0):
			return a.get(y);
		case(1):
			return b.get(y);
		case(2):
			return c.get(y);
		default:
			return d.get(y);
		}
	}
	void set(int x, int y, float z) {
		switch (x) {
		case(0):
			a.set(y, z);
		case(1):
			b.set(y, z);
		case(2):
			c.set(y, z);
		default:
			d.set(y, z);
		}
	}
};
struct Mat4x4 {
	Mat4x4(float f) {
		a = Vec4(f);
		b = Vec4(f);
		c = Vec4(f);
		d = Vec4(f);
	}
	Mat4x4() {
		a = Vec4(0.f);
		b = Vec4(0.f);
		c = Vec4(0.f);
		d = Vec4(0.f);
	}
	Mat4x4(Vec4 v1, Vec4 v2, Vec4 v3, Vec4 v4) {
		a = v1;
		b = v2;
		c = v3;
		d = v4;
	}
	Mat4x4(Mat4x3 M) {
		a = Vec4(M.a, 0.f);
		b = Vec4(M.b, 0.f);
		c = Vec4(M.c, 0.f);
		d = Vec4(M.d, 1.f);
	}
	Mat4x4(Mat3x3 M) {
		a = Vec4(M.a, 0.f);
		b = Vec4(M.b, 0.f);
		c = Vec4(M.c, 0.f);
		d = Vec4(0.f,0.f,0.f, 1.f);
	}

	Vec4 a;
	Vec4 b;
	Vec4 c;
	Vec4 d;
	float get(int x, int y) {
		switch (x) {
		case(0):
			return a.get(y);
		case(1):
			return b.get(y);
		case(2):
			return c.get(y);
		default:
			return d.get(y);
		}
	}
	void set(int x, int y, float z) {
		switch (x) {
		case(0):
			a.set(y, z);
		case(1):
			b.set(y, z);
		case(2):
			c.set(y, z);
		default:
			d.set(y, z);
		}
	}
};




#define EPSILON 0.00001f
#define PI 3.1415926f
//https://stackoverflow.com/questions/686353/random-float-number-generation
#define randf()  static_cast <float> (rand()) / static_cast <float> (RAND_MAX)
struct Color3
{
	Color3(unsigned char c) {
		r = c;
		g = c;
		b = c;
	};
	Color3(unsigned char a, unsigned char d, unsigned char c) {
		r = a;
		g = d;
		b = c;
	};
	Color3(Vec3 v) { //Assume float in [0.f,1.f]
		auto round = [](float f) {
			return floor(f + 0.5f);
		};
		r = round(v.x * 255.f);
		g = round(v.y * 255.f);
		b = round(v.z * 255.f);
	};
	Color3() {
		r = 255; g = 255; b = 255;
	};
	Vec3 toVec3() {
		return Vec3((float)r/255.f,(float)g/255.f,(float)b/255.f);
	};
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

inline void printVec3(Vec3 v) {
	std::cout << v.x << " " << v.y << " " << v.z;
}

inline Color3 normToColor(Vec3 n) {
	float x = n.x;
	float y = n.y;
	float z = n.z;
	if (x < 0) x = 0.5f;
	else if (x < 0.5) x = 0.f;
	if (y < 0) y = 0.5f;
	else if (y < 0.5f) y = 0.f;
	if (z < 0) z = 0.5f;
	else if (z < 0.5f) z = 0.f;
	return Color3(Vec3(x, y, z));
}

//Note on coordinates:

//Camera normally points along -z in camera space. For convinience then, world space will use right hand rule if need be, and have y be up
//Spherical coordinates: Vertical: phi [0,pi], theta [0,2pi]




inline Vec4 constVecMult(float a, Vec4 v) {
	return Vec4(a * v.x, a * v.y, a * v.z, a * v.w);
}
inline Vec3 constVecMult(float a, Vec3 v) {
	return Vec3(a * v.x, a * v.y, a * v.z);
}
inline Vec2 constVecMult(float a, Vec2 v) {
	return Vec2(a * v.x, a * v.y);
}
inline Vec4 vecVecMult(Vec4 a, Vec4 v) {
	return Vec4(a.x * v.x, a.y * v.y, a.z * v.z, a.w * v.w);
}
inline Vec3 vecVecMult(Vec3 a, Vec3 v) {
	return Vec3(a.x * v.x, a.y * v.y, a.z * v.z);
}
inline Vec2 vecVecMult(Vec2 a, Vec2 v) {
	return Vec2(a.x * v.x, a.y * v.y);
}
inline Vec4 vecVecAdd(Vec4 a, Vec4 v) {
	return Vec4(a.x + v.x, a.y + v.y, a.z + v.z, a.w + v.w);
}
inline Vec3 vecVecAdd(Vec3 a, Vec3 v) {
	return Vec3(a.x + v.x, a.y + v.y, a.z + v.z);
}
inline Vec2 vecVecAdd(Vec2 a, Vec2 v) {
	return Vec2(a.x + v.x, a.y + v.y);
}
inline Vec4 vecVecSub(Vec4 a, Vec4 v) {
	return Vec4(a.x - v.x, a.y - v.y, a.z - v.z, a.w - v.w);
}
inline Vec3 vecVecSub(Vec3 a, Vec3 v) {
	return Vec3(a.x - v.x, a.y - v.y, a.z - v.z);
}
inline Vec2 vecVecSub(Vec2 a, Vec2 v) {
	return Vec2(a.x - v.x, a.y - v.y);
}
inline Vec4 vecVecDiv(Vec4 a, Vec4 v) {
	return Vec4(a.x / v.x, a.y / v.y, a.z / v.z, a.w/v.w);
}
inline Vec3 vecVecDiv(Vec3 a, Vec3 v) {
	return Vec3(a.x / v.x, a.y / v.y, a.z / v.z);
}
inline Vec2 vecVecDiv(Vec2 a, Vec2 v) {
	return Vec2(a.x / v.x, a.y / v.y);
}
inline float dot(Vec2 a, Vec2 b) {
	return a.x * b.x + a.y * b.y;
}
inline float dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline float dot(Vec4 a, Vec4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline Vec3 vecNormalize(Vec3 v) {
	return vecVecDiv(v, Vec3(vecNorm(v)));
}

inline Vec4 vecNormalize(Vec4 v) {
	return vecVecDiv(v, Vec4(vecNorm(v)));
}

inline Vec2 vecNormalize(Vec2 v) {
	return vecVecDiv(v, Vec2(vecNorm(v)));;
}
