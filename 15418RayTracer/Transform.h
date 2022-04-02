#pragma once
#include "Defined.h"
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
	Vec4 matVecMult(Mat4x4 M, Vec4 v) {
		return M * v;
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
	Mat4x3 makeTransform();//Of note, 4 columns, 3 rows, homogenous coordinate row not accounted for. GLM has a weird naming format
	Mat4x3 makeAndSaveTransform(){
		tempMatrix = makeTransform();
	}//Above but saves to temp matrix for performance reasons
	Mat4x3 tempMatrix; //^^^^

	//Parent space transformations
	Mat4x3 localToParent() const;
	Mat4x3 parentToLocal() const;

	//World space transformations
	Mat4x3 localToWorld() const;
	Mat4x3 worldToLocal() const;

	Transform* parent = nullptr;

	Transform();
};

