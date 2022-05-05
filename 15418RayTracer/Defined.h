#pragma once

#ifndef  DEF_h
#define DEF_h


#include "glm/glm.hpp"
#include <iostream>
#include <io.h>
#include <cuda_runtime.h>


//All vec operators based on equivalent implementations in Scotty3D


struct Vec3 {
	__device__ Vec3() {
		x = 0.f;
		y = 0.f;
		z = 0.f;
	}
	__device__ Vec3(float a, float b, float c) {
		x = a; y = b; z = c;
	}
	__device__ Vec3(float a) {
		x = a; y = a; z = a;
	}
	float x;
	float y;
	float z;
	__device__ float get(int ind) {
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
	__device__ void set(int ind, float a) {
		switch (ind) {
		case(0):
			x = a;
		case(1):
			y = a;
		default:
			z = a;
		}
	}



	__device__ float& operator[](int idx) {
		assert(idx >= 0 && idx <= 2);
		switch (idx)
		{
		case(0):
			return x;
		case(1):
			return y;
		default:
			return z;
		};
	}
	__device__ float operator[](int idx) const {
		assert(idx >= 0 && idx <= 2);
		switch (idx)
		{
		case(0):
			return x;
		case(1):
			return y;
		default:
			return z;
		};
	}


	__device__ Vec3 operator+=(Vec3 v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}
	__device__ Vec3 operator-=(Vec3 v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	__device__ Vec3 operator*=(Vec3 v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}
	__device__ Vec3 operator/=(Vec3 v) {
		x /= v.x;
		y /= v.y;
		z /= v.z;
		return *this;
	}

	__device__ Vec3 operator+=(float s) {
		x += s;
		y += s;
		z += s;
		return *this;
	}
	__device__ Vec3 operator-=(float s) {
		x -= s;
		y -= s;
		z -= s;
		return *this;
	}
	__device__ Vec3 operator*=(float s) {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}
	__device__ Vec3 operator/=(float s) {
		x /= s;
		y /= s;
		z /= s;
		return *this;
	}

	__device__ Vec3 operator+(Vec3 v) const {
		return Vec3(x + v.x, y + v.y, z + v.z);
	}
	__device__ Vec3 operator-(Vec3 v) const {
		return Vec3(x - v.x, y - v.y, z - v.z);
	}
	__device__ Vec3 operator*(Vec3 v) const {
		return Vec3(x * v.x, y * v.y, z * v.z);
	}
	__device__ Vec3 operator/(Vec3 v) const {
		return Vec3(x / v.x, y / v.y, z / v.z);
	}

	__device__ Vec3 operator+(float s) const {
		return Vec3(x + s, y + s, z + s);
	}
	__device__ Vec3 operator-(float s) const {
		return Vec3(x - s, y - s, z - s);
	}
	__device__ Vec3 operator*(float s) const {
		return Vec3(x * s, y * s, z * s);
	}
	__device__ Vec3 operator/(float s) const {
		return Vec3(x / s, y / s, z / s);
	}

	__device__ bool operator==(Vec3 v) const {
		return x == v.x && y == v.y && z == v.z;
	}
	__device__ bool operator!=(Vec3 v) const {
		return x != v.x || y != v.y || z != v.z;
	}


};

struct Vec4 {
	__device__ Vec4() {
		x = 0.f;
		y = 0.f;
		z = 0.f;
		w = 0.f;
	}
	__device__ Vec4(float a, float b, float c, float d) {
		x = a; y = b; z = c, w = d;
	}
	__device__ Vec4(float a) {
		w = a;  x = a; y = a; z = a;
	}
	__device__ Vec4(Vec3 v, float a) {
		x = v.x; y = v.y; z = v.z;
		w = a;
	}
	float x;
	float y;
	float z;
	float w;

	__device__ float& operator[](int idx) {
		assert(idx >= 0 && idx <= 3);
		switch (idx)
		{
		case(0):
			return x;
		case(1):
			return y;
		case(2):
			return z;
		default:
			return w;
		};
	}
	__device__ float operator[](int idx) const {
		assert(idx >= 0 && idx <= 3);
		switch (idx)
		{
		case(0):
			return x;
		case(1):
			return y;
		case(2):
			return z;
		default:
			return w;
		};
	}

	__device__ Vec4 operator+=(Vec4 v) {
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;
		return *this;
	}
	__device__ Vec4 operator-=(Vec4 v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;
		return *this;
	}
	__device__ Vec4 operator*=(Vec4 v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		w *= v.w;
		return *this;
	}
	__device__ Vec4 operator/=(Vec4 v) {
		x /= v.x;
		y /= v.y;
		z /= v.z;
		w /= v.w;
		return *this;
	}

	__device__ Vec4 operator+=(float s) {
		x += s;
		y += s;
		z += s;
		w += s;
		return *this;
	}
	__device__ Vec4 operator-=(float s) {
		x -= s;
		y -= s;
		z -= s;
		w -= s;
		return *this;
	}
	__device__ Vec4 operator*=(float s) {
		x *= s;
		y *= s;
		z *= s;
		w *= s;
		return *this;
	}
	__device__ Vec4 operator/=(float s) {
		x /= s;
		y /= s;
		z /= s;
		w /= s;
		return *this;
	}

	__device__ Vec4 operator+(Vec4 v) const {
		return Vec4(x + v.x, y + v.y, z + v.z, w + v.w);
	}
	__device__ Vec4 operator-(Vec4 v) const {
		return Vec4(x - v.x, y - v.y, z - v.z, w - v.w);
	}
	__device__ Vec4 operator*(Vec4 v) const {
		return Vec4(x * v.x, y * v.y, z * v.z, w * v.w);
	}
	__device__ Vec4 operator/(Vec4 v) const {
		return Vec4(x / v.x, y / v.y, z / v.z, w / v.w);
	}

	__device__ Vec4 operator+(float s) const {
		return Vec4(x + s, y + s, z + s, w + s);
	}
	__device__ Vec4 operator-(float s) const {
		return Vec4(x - s, y - s, z - s, w - s);
	}
	__device__ Vec4 operator*(float s) const {
		return Vec4(x * s, y * s, z * s, w * s);
	}
	__device__ Vec4 operator/(float s) const {
		return Vec4(x / s, y / s, z / s, w / s);
	}

	__device__ bool operator==(Vec4 v) const {
		return x == v.x && y == v.y && z == v.z && w == v.w;
	}
	__device__ bool operator!=(Vec4 v) const {
		return x != v.x || y != v.y || z != v.z || w != v.w;
	}


	__device__ float get(int ind) {
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
	__device__ void set(int ind, float a) {
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

	__device__ float& operator[](int idx) {
		assert(idx >= 0 && idx <= 1);
		switch (idx)
		{
		case(0):
			return x;
		default:
			return y;
		};
	}
	__device__ float operator[](int idx) const {
		assert(idx >= 0 && idx <= 1);
		switch (idx)
		{
		case(0):
			return x;
		default:
			return y;
		};
	}
	__device__ Vec2() {
		x = 0.f;
		y = 0.f;
	}
	__device__ Vec2(float a) {
		x = a; y = a;
	}
	__device__ Vec2(float a, float b) {
		x = a; y = b;
	}
	float x;
	float y;
	__device__ float get(int ind) {
		switch (ind)
		{
		case (0):
			return x;
		default:
			return y;
		}
	}
	__device__ void set(int ind, float a) {
		switch (ind)
		{
		case (0):
			x = a;
		default:
			y = a;
		}
	}



	__device__ Vec2 operator+=(Vec2 v) {
		x += v.x;
		y += v.y;
		return *this;
	}
	__device__ Vec2 operator-=(Vec2 v) {
		x -= v.x;
		y -= v.y;
		return *this;
	}
	__device__ Vec2 operator*=(Vec2 v) {
		x *= v.x;
		y *= v.y;
		return *this;
	}
	__device__ Vec2 operator/=(Vec2 v) {
		x /= v.x;
		y /= v.y;
		return *this;
	}

	__device__ Vec2 operator+=(float s) {
		x += s;
		y += s;
		return *this;
	}
	__device__ Vec2 operator-=(float s) {
		x -= s;
		y -= s;
		return *this;
	}
	__device__ Vec2 operator*=(float s) {
		x *= s;
		y *= s;
		return *this;
	}
	__device__ Vec2 operator/=(float s) {
		x /= s;
		y /= s;
		return *this;
	}

	__device__ Vec2 operator+(Vec2 v) const {
		return Vec2(x + v.x, y + v.y);
	}
	__device__ Vec2 operator-(Vec2 v) const {
		return Vec2(x - v.x, y - v.y);
	}
	__device__ Vec2 operator*(Vec2 v) const {
		return Vec2(x * v.x, y * v.y);
	}
	__device__ Vec2 operator/(Vec2 v) const {
		return Vec2(x / v.x, y / v.y);
	}

	__device__ Vec2 operator+(float s) const {
		return Vec2(x + s, y + s);
	}
	__device__ Vec2 operator-(float s) const {
		return Vec2(x - s, y - s);
	}
	__device__ Vec2 operator*(float s) const {
		return Vec2(x * s, y * s);
	}
	__device__ Vec2 operator/(float s) const {
		return Vec2(x / s, y / s);
	}

	__device__ bool operator==(Vec2 v) const {
		return x == v.x && y == v.y;
	}
	__device__ bool operator!=(Vec2 v) const {
		return x != v.x || y != v.y;
	}
};
__device__ inline float vecNorm(Vec2 v) {
	return sqrt(v.x * v.x + v.y * v.y);
}
__device__ inline float vecNorm(Vec3 v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
__device__ inline float vecNorm(Vec4 v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

//All operators based on equivalent implementations
//From Scotty3D
struct Mat3x3;
struct Mat4x3;
struct Mat4x4;

struct Mat3x3 {



	__device__ Vec3& operator[](int idx) {
		assert(idx >= 0 && idx <= 2);
		switch (idx) {
		case(0):
			return a;
		case(1):
			return b;
		default:
			return c;
		}
	}
	__device__ Vec3 operator[](int idx) const {
		assert(idx >= 0 && idx <= 2);
		switch (idx) {
		case(0):
			return a;
		case(1):
			return b;
		default:
			return c;
		}
	}

	__device__ Mat3x3 operator+=(const Mat3x3& m) {
		a += m.a;
		b += m.b;
		c += m.c;
		return *this;
	}
	__device__ Mat3x3 operator-=(const Mat3x3& m) {
		a -= m.a;
		b -= m.b;
		c -= m.c;
		return *this;
	}

	__device__ Mat3x3 operator+=(float s) {
		a += s;
		b += s;
		c += s;
		return *this;
	}
	__device__ Mat3x3 operator-=(float s) {
		a -= s;
		b -= s;
		c -= s;
		return *this;
	}
	__device__ Mat3x3 operator*=(float s) {
		a *= s;
		b *= s;
		c *= s;
		return *this;
	}
	__device__ Mat3x3 operator/=(float s) {
		a /= s;
		b /= s;
		c /= s;
		return *this;
	}

	__device__ Mat3x3 operator+(const Mat3x3& m) const {
		Mat3x3 r;
		r.a = a + m.a;
		r.b = b + m.b;
		r.c = c + m.c;
		return r;
	}
	__device__ Mat3x3 operator-(const Mat3x3& m) const {
		Mat3x3 r;
		r.a = a - m.a;
		r.b = b - m.b;
		r.c = c - m.c;
		return r;
	}

	__device__ Mat3x3 operator+(float s) const {
		Mat3x3 r;
		r.a = a + s;
		r.b = b + s;
		r.c = c + s;
		return r;
	}
	__device__ Mat3x3 operator-(float s) const {
		Mat3x3 r;
		r.a = a - s;
		r.b = b - s;
		r.c = c - s;
		return r;
	}
	__device__ Mat3x3 operator*(float s) const {
		Mat3x3 r;
		r.a = a * s;
		r.b = b * s;
		r.c = c * s;
		return r;
	}
	__device__ Mat3x3 operator/(float s) const {
		Mat3x3 r;
		r.a = a / s;
		r.b = b / s;
		r.c = c / s;
		return r;
	}

	__device__ Mat3x3 operator*=(const Mat3x3& v) {
		*this = *this * v;
		return *this;
	}
	__device__ Mat3x3 operator*(const Mat3x3& m) const {
		Mat3x3 ret;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				ret[i][j] = 0.0f;
				ret[i][j] += m[i][0] * a[j];
				ret[i][j] += m[i][1] * b[j];
				ret[i][j] += m[i][2] * c[j];
			}
		}
		return ret;
	}

	__device__ Vec3 operator*(Vec3 v) const {
		return a * v[0] + b * v[1] + c * v[2];
	}
	__device__ Mat3x3() {
		a = Vec3(0.f);
		b = Vec3(0.f);
		c = Vec3(0.f);
	}
	__device__ Mat3x3(Vec3 v1, Vec3 v2, Vec3 v3) {
		a = v1;
		b = v2;
		c = v3;
	}
	__device__ Mat3x3(float f) {
		a = Vec3(f);
		b = Vec3(f);
		c = Vec3(f);
	}

	Vec3 a;
	Vec3 b;
	Vec3 c;
	__device__ float get(int x, int y) {
		switch (x) {
		case(0):
			return a.get(y);
		case(1):
			return b.get(y);
		default:
			return c.get(y);
		}
	}
	__device__ void set(int x, int y, float z) {
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


	__device__ Vec3& operator[](int idx) {
		assert(idx >= 0 && idx <= 3);
		switch (idx) {
		case(0):
			return a;
		case(1):
			return b;
		case(2):
			return c;
		default:
			return d;
		}
	}
	__device__ Vec3 operator[](int idx) const {
		assert(idx >= 0 && idx <= 3);
		switch (idx) {
		case(0):
			return a;
		case(1):
			return b;
		case(2):
			return c;
		default:
			return d;
		}
	}

	__device__ Mat4x3 operator+=(const Mat4x3& m) {
		a += m.a;
		b += m.b;
		c += m.c;
		d += m.d;
		return *this;
	}
	__device__ Mat4x3 operator-=(const Mat4x3& m) {
		a -= m.a;
		b -= m.b;
		c -= m.c;
		d -= m.d;
		return *this;
	}

	__device__ Mat4x3 operator+=(float s) {
		a += s;
		b += s;
		c += s;
		d += s;
		return *this;
	}
	__device__ Mat4x3 operator-=(float s) {
		a -= s;
		b -= s;
		c -= s;
		d -= s;
		return *this;
	}
	__device__ Mat4x3 operator*=(float s) {
		a *= s;
		b *= s;
		c *= s;
		d *= s;
		return *this;
	}
	__device__ Mat4x3 operator/=(float s) {
		a /= s;
		b /= s;
		c /= s;
		d /= s;
		return *this;
	}

	__device__ Mat4x3 operator+(const Mat4x3& m) const {
		Mat4x3 r;
		r.a = a + m.a;
		r.b = b + m.b;
		r.c = c + m.c;
		r.d = d + m.d;
		return r;
	}
	__device__ Mat4x3 operator-(const Mat4x3& m) const {
		Mat4x3 r;
		r.a = a - m.a;
		r.b = b - m.b;
		r.c = c - m.c;
		r.d = d - m.d;
		return r;
	}

	__device__ Mat4x3 operator+(float s) const {
		Mat4x3 r;
		r.a = a + s;
		r.b = b + s;
		r.c = c + s;
		r.d = d + s;
		return r;
	}
	__device__ Mat4x3 operator-(float s) const {
		Mat4x3 r;
		r.a = a - s;
		r.b = b - s;
		r.c = c - s;
		r.d = d - s;
		return r;
	}
	__device__ Mat4x3 operator*(float s) const {
		Mat4x3 r;
		r.a = a * s;
		r.b = b * s;
		r.c = c * s;
		r.d = d * s;
		return r;
	}
	__device__ Mat4x3 operator/(float s) const {
		Mat4x3 r;
		r.a = a / s;
		r.b = b / s;
		r.c = c / s;
		r.d = d / s;
		return r;
	}

	__device__ Mat4x3 operator*(const Mat4x4& m) const;

	__device__ Vec3 operator*(Vec4 v) const {
		return a * v[0] + b * v[1] + c * v[2] + d * v[3];
	}
	__device__ Mat4x3(float f) {
		a = Vec3(f);
		b = Vec3(f);
		c = Vec3(f);
		d = Vec3(f);
	}
	__device__ Mat4x3() {
		a = Vec3(0.f);
		b = Vec3(0.f);
		c = Vec3(0.f);
		d = Vec3(0.f);
	}
	__device__ Mat4x3(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4) {
		a = v1;
		b = v2;
		c = v3;
		d = v4;
	}
	__device__ Mat4x3(Mat3x3 M) {
		a = M.a;
		b = M.b;
		c = M.c;
		d = Vec3(0.f);
	}

	Vec3 a;
	Vec3 b;
	Vec3 c;
	Vec3 d;
	__device__ float get(int x, int y) {
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
	__device__ void set(int x, int y, float z) {
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


	__device__ Mat3x3 toMat3x3() {
		Mat3x3 ret;
		ret.a[0] = (*this)[0][0];
		ret.a[1] = (*this)[0][1];
		ret.a[2] = (*this)[0][2];
		ret.b[0] = (*this)[1][0];
		ret.b[1] = (*this)[1][1];
		ret.b[2] = (*this)[1][2];
		ret.c[0] = (*this)[2][0];
		ret.c[1] = (*this)[2][1];
		ret.c[2] = (*this)[2][2];
		return ret;
	}

	__device__ Vec4& operator[](int idx) {
		assert(idx >= 0 && idx <= 3);
		switch (idx) {
		case(0):
			return a;
		case(1):
			return b;
		case(2):
			return c;
		default:
			return d;
		}
	}
	__device__ Vec4 operator[](int idx) const {
		assert(idx >= 0 && idx <= 3);
		switch (idx) {
		case(0):
			return a;
		case(1):
			return b;
		case(2):
			return c;
		default:
			return d;
		}
	}

	__device__ Mat4x4 operator+=(const Mat4x4& m) {
		a += m.a;
		b += m.b;
		c += m.c;
		d += m.d;
		return *this;
	}
	__device__ Mat4x4 operator-=(const Mat4x4& m) {
		a -= m.a;
		b -= m.b;
		c -= m.c;
		d -= m.d;
		return *this;
	}

	__device__ Mat4x4 operator+=(float s) {
		a += s;
		b += s;
		c += s;
		d += s;
		return *this;
	}
	__device__ Mat4x4 operator-=(float s) {
		a -= s;
		b -= s;
		c -= s;
		d -= s;
		return *this;
	}
	__device__ Mat4x4 operator*=(float s) {
		a *= s;
		b *= s;
		c *= s;
		d *= s;
		return *this;
	}
	__device__ Mat4x4 operator/=(float s) {
		a /= s;
		b /= s;
		c /= s;
		d /= s;
		return *this;
	}

	__device__ Mat4x4 operator+(const Mat4x4& m) const {
		Mat4x4 r;
		r.a = a + m.a;
		r.b = b + m.b;
		r.c = c + m.c;
		r.d = d + m.d;
		return r;
	}
	__device__ Mat4x4 operator-(const Mat4x4& m) const {
		Mat4x4 r;
		r.a = a - m.a;
		r.b = b - m.b;
		r.c = c - m.c;
		r.d = d - m.d;
		return r;
	}

	__device__ Mat4x4 operator+(float s) const {
		Mat4x4 r;
		r.a = a + s;
		r.b = b + s;
		r.c = c + s;
		r.d = d + s;
		return r;
	}
	__device__ Mat4x4 operator-(float s) const {
		Mat4x4 r;
		r.a = a - s;
		r.b = b - s;
		r.c = c - s;
		r.d = d - s;
		return r;
	}
	__device__ Mat4x4 operator*(float s) const {
		Mat4x4 r;
		r.a = a * s;
		r.b = b * s;
		r.c = c * s;
		r.d = d * s;
		return r;
	}
	__device__ Mat4x4 operator/(float s) const {
		Mat4x4 r;
		r.a = a / s;
		r.b = b / s;
		r.c = c / s;
		r.d = d / s;
		return r;
	}

	__device__ Mat4x4 operator*=(const Mat4x4& v) {
		*this = *this * v;
		return *this;
	}
	__device__ Mat4x4 operator*(const Mat4x4& m) const {
		Mat4x4 ret;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				ret[i][j] = 0.0f;
				ret[i][j] += m[i][0] * a[j];
				ret[i][j] += m[i][1] * b[j];
				ret[i][j] += m[i][2] * c[j];
				ret[i][j] += m[i][3] * d[j];
			}
		}
		return ret;
	}

	__device__ Vec4 operator*(Vec4 v) const {
		return a * v[0] + b * v[1] + c * v[2] + d * v[3];
	}


	__device__ Mat4x4(float f) {
		a = Vec4(f);
		b = Vec4(f);
		c = Vec4(f);
		d = Vec4(f);
	}
	__device__ Mat4x4() {
		a = Vec4(0.f);
		b = Vec4(0.f);
		c = Vec4(0.f);
		d = Vec4(0.f);
	}
	__device__ Mat4x4(Vec4 v1, Vec4 v2, Vec4 v3, Vec4 v4) {
		a = v1;
		b = v2;
		c = v3;
		d = v4;
	}
	__device__ Mat4x4(Mat4x3 M) {
		a = Vec4(M.a, 0.f);
		b = Vec4(M.b, 0.f);
		c = Vec4(M.c, 0.f);
		d = Vec4(M.d, 1.f);
	}
	__device__ Mat4x4(Mat3x3 M) {
		a = Vec4(M.a, 0.f);
		b = Vec4(M.b, 0.f);
		c = Vec4(M.c, 0.f);
		d = Vec4(0.f, 0.f, 0.f, 1.f);
	}

	Vec4 a;
	Vec4 b;
	Vec4 c;
	Vec4 d;
	__device__ float get(int x, int y) {
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
	__device__ void set(int x, int y, float z) {
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
	__device__ Color3(unsigned char c) {
		r = c;
		g = c;
		b = c;
	};
	__device__ Color3(unsigned char a, unsigned char d, unsigned char c) {
		r = a;
		g = d;
		b = c;
	};
	__device__ Color3(Vec3 v) { //Assume float in [0.f,1.f]
		auto round = [](float f) {
			return floor(f + 0.5f);
		};
		r = round(v.x * 255.f);
		g = round(v.y * 255.f);
		b = round(v.z * 255.f);
	};
	__device__ Color3() {
		r = 255; g = 255; b = 255;
	};
	__device__ Vec3 toVec3() {
		return Vec3((float)r / 255.f, (float)g / 255.f, (float)b / 255.f);
	};
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

__device__ inline void printVec3(Vec3 v) {
	std::cout << v.x << " " << v.y << " " << v.z;
}

__device__ inline Color3 normToColor(Vec3 n) {
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




__device__ inline Vec4 constVecMult(float a, Vec4 v) {
	return v * a;
}
__device__ inline Vec3 constVecMult(float a, Vec3 v) {
	return v * a;
}
__device__ inline Vec2 constVecMult(float a, Vec2 v) {
	return v * a;
}
__device__ inline Vec4 vecVecMult(Vec4 a, Vec4 v) {
	return a * v;
}
__device__ inline Vec3 vecVecMult(Vec3 a, Vec3 v) {
	return a * v;
}
__device__ inline Vec2 vecVecMult(Vec2 a, Vec2 v) {
	return a * v;
}
__device__ inline Vec4 vecVecAdd(Vec4 a, Vec4 v) {
	return a + v;
}
__device__ inline Vec3 vecVecAdd(Vec3 a, Vec3 v) {
	return a + v;
}
__device__ inline Vec2 vecVecAdd(Vec2 a, Vec2 v) {
	return a + v;
}
__device__ inline Vec4 vecVecSub(Vec4 a, Vec4 v) {
	return a - v;
}
__device__ inline Vec3 vecVecSub(Vec3 a, Vec3 v) {
	return a - v;
}
__device__ inline Vec2 vecVecSub(Vec2 a, Vec2 v) {
	return a - v;
}
__device__ inline Vec4 vecVecDiv(Vec4 a, Vec4 v) {
	return a / v;
}
__device__ inline Vec3 vecVecDiv(Vec3 a, Vec3 v) {
	return a / v;
}
__device__ inline Vec2 vecVecDiv(Vec2 a, Vec2 v) {
	return a / v;
}
__device__ inline float dot(Vec2 a, Vec2 b) {
	return a.x * b.x + a.y * b.y;
}
__device__ inline float dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float dot(Vec4 a, Vec4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__device__ inline Vec3 vecNormalize(Vec3 v) {
	return vecVecDiv(v, Vec3(vecNorm(v)));
}

__device__ inline Vec4 vecNormalize(Vec4 v) {
	return vecVecDiv(v, Vec4(vecNorm(v)));
}

__device__ inline Vec2 vecNormalize(Vec2 v) {
	return vecVecDiv(v, Vec2(vecNorm(v)));;
}



inline Mat4x3 Mat4x3::operator*(const Mat4x4& m) const {
	Mat4x3 ret;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 3; j++) {
			ret[i][j] = 0.0f;
			ret[i][j] += m[i][0] * a[j];
			ret[i][j] += m[i][1] * b[j];
			ret[i][j] += m[i][2] * c[j];
			ret[i][j] += m[i][3] * d[j];
		}
	}
	return ret;
}


#endif // ! DEF_h
