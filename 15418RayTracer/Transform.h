#pragma once
#include "Defined.h"
#include "glm/glm.hpp"
class Transform
{
public:
	//All of these will be reimplemented in cuda, so Im making these functions now to simplify that conversion
	Mat3x3 matMult(Mat3x3 A, Mat3x3 B) {
		return A * B;
	}
	Mat4x4 matMult(Mat4x4 A, Mat4x4 B) {
		return A * B;
	}
	Mat4x3 matMult(Mat4x3 A, Mat3x3 B) {
		return B*A;
	}
	Mat4x4 matInverse(Mat4x4 M) {
		return glm::inverse(M);
	}
	Mat4x3 matInverse(Mat4x3 M) {
		//return Mat4x3( glm::inverse(Mat4x4(M)));
		return M;
	}
	Mat3x3 matInverse(Mat3x3 M) {
		return glm::inverse(M);
	}
	Vec4 matVecMult(Mat4x4 M, Vec4 v) {
		return M * v;
	}
	Vec3 matVecMult(Mat4x4 M, Vec3 v) {
		return Vec3(M * Vec4(v,1.f));
	}
	Vec3 matVecMult(Mat3x3 M, Vec3 v) {
		return M * v;
	}
	float dot(Vec2 a, Vec2 b) {
		return glm::dot(a, b);
	}
	float dot(Vec3 a, Vec3 b) {
		return glm::dot(a, b);
	}
	float dot(Vec4 a, Vec4 b) {
		return glm::dot(a, b);
	}

	//Transform structure
	Vec3 pos = Vec3(0.0f, 0.0f, 0.0f);
	Vec3 rot = Vec3(0.f, 0.f, 0.f); //Rotations are
	Vec3 scale = Vec3(1.0f, 1.0f, 1.0f);

	//Take above structures and create transformation matrix
	Mat4x4 makeTransform();//Of note, 4 columns, 3 rows, homogenous coordinate row not accounted for. GLM has a weird naming format
	Mat4x4 makeAndSaveTransform(){
		tempMatrix = makeTransform();
		return tempMatrix;
	}//Above but saves to temp matrix for performance reasons
	Mat4x4 tempMatrix = Mat4x3(0.f); //^^^^
	bool tempMatrixFilled = false;

	//Parent space transformations
	Mat4x4 localToParent();
	Mat4x4 parentToLocal();

	//World space transformations
	Mat4x4 localToWorld();
	Mat4x4 worldToLocal();

	Transform* parent = nullptr;

	//Need to implement in a way thatll work on cuda
	Vec3 normTransform(Vec3 v) {
		return glm::normalize(glm::transpose(glm::inverse(localToWorld())) * Vec4(v, 0.f));
	}

	Transform(Vec3 p, Vec3 r, Vec3 s) {
		pos = p;
		rot = r;
		scale = s;
	}

	Transform() {
		pos = Vec3(0.f);
		rot = Vec3(0.f);
		scale = Vec3(1.f);
	}

	Transform(float x, float y, float z, float thx, float thy, float thz, float sx, float sy, float sz) {
		pos = Vec3(x,y,z);
		rot = Vec3(thx,thy,thz);
		scale = Vec3(sx,sy,sz);
	}
};
