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
		return Mat4x3( glm::inverse(Mat4x4(M)));
	}
	Mat3x3 matInverse(Mat3x3 M) {
		return glm::inverse(M);
	}
	Mat4x4 matTranspose(Mat4x4 M) {
		return glm::transpose(M);
	}
	Mat4x3 matTranspose(Mat4x3 M) {
		return glm::transpose(M);
	}
	Mat3x3 matTranspose(Mat3x3 M) {
		return glm::transpose(M);
	}
	Vec4 matVecMult(Mat4x4 M, Vec4 v) {
		glm::vec4 temp = M * glm::vec4(v.x, v.y, v.z, v.w);
		return Vec4(temp.w, temp.x, temp.y, temp.z);
	}
	Vec3 matVecMult(Mat4x4 M, Vec3 v) {
		glm::vec3 temp = (M * glm::vec4(glm::vec3(v.x, v.y, v.z), 1.f));
		return Vec3(temp.x, temp.y, temp.z);
	}
	Vec3 matVecMult(Mat3x3 M, Vec3 v) {
		glm::vec3 temp = M * glm::vec3(v.x, v.y, v.z);
		return Vec3(temp.x, temp.y, temp.z);
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
		Vec4 newV = vecNormalize(matVecMult(matTranspose(matInverse(localToWorld())), Vec4(v.x, v.y, v.z, 0.f)));
		return Vec3(newV.x,newV.y,newV.z);
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