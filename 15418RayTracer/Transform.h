#pragma once
#include "Defined.h"
#include "glm/glm.hpp"
#include "cublas_v2.h"
#include <cuda.h>
#include <driver_functions.h>
#include <cuda_runtime.h>


inline float rowCol(Mat3x3 A, Mat3x3 B, int a, int b) {
	float res = 0.f;
	for (int j = 0; j < 3; j++) {
		res += A.get(j, a) * B.get(b, j);
	}
	return res;
}
inline float rowCol(Mat4x4 A, Mat4x4 B, int a, int b) {
	float res = 0.f;
	for (int j = 0; j < 4; j++) {
		res += A.get(j, a) * B.get(b, j);
	}
	return res;
}

inline glm::mat4x4 toMat(Mat4x4 M) {
	glm::vec4 a = glm::vec4(M.a.x, M.a.y, M.a.z, M.a.w);
	glm::vec4 b = glm::vec4(M.b.x, M.b.y, M.b.z, M.b.w);
	glm::vec4 c = glm::vec4(M.c.x, M.c.y, M.c.z, M.c.w);
	glm::vec4 d = glm::vec4(M.d.x, M.d.y, M.d.z, M.d.w);
	return glm::mat4x4(a, b, c, d);
}

inline glm::mat3x3 toMat(Mat3x3 M) {
	glm::vec3 a = glm::vec3(M.a.x, M.a.y, M.a.z);
	glm::vec3 b = glm::vec3(M.b.x, M.b.y, M.b.z);
	glm::vec3 c = glm::vec3(M.c.x, M.c.y, M.c.z);
	return glm::mat3x3(a, b, c);
}

class Transform
{
public:
	//All of these will be reimplemented in cuda, so Im making these functions now to simplify that conversion
	Mat3x3 matMult(Mat3x3 A, Mat3x3 B) {
		Mat3x3 res;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				res.set(j, i, rowCol(A, B, i, j));
			}
		}
		return res;
	}
	Mat4x4 matMult(Mat4x4 A, Mat4x4 B) {
		Mat4x4 res;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				res.set(j, i, rowCol(A, B, i, j));
			}
		}
		return res;
	}
	Mat4x3 matMult(Mat3x3 A, Mat4x3 B) {
		Mat4x4 temp = (Mat4x4(A), Mat4x4(B));
		Mat4x3 res;
		res.a = Vec3(temp.a.x, temp.a.y, temp.a.z);
		res.b = Vec3(temp.b.x, temp.b.y, temp.b.z);
		res.c = Vec3(temp.c.x, temp.c.y, temp.c.z);
		res.d = Vec3(temp.d.x, temp.d.y, temp.d.z);
		return res;
	} 
	//THESE 2 WILL NEED TO USE CUDA LA LIBRARY
	Mat4x4 matInverse(Mat4x4 M) {

		float* devMatrix;
		cudaMalloc(&devMatrix, sizeof(float) * 16);
		assert(cublasSetMatrix(4, 4, sizeof(float), (void*)&M, 4, devMatrix, 4) == CUBLAS_STATUS_SUCCESS);
		//Calc inverse
		assert(cublasGetMatrix(4,4,sizeof(float),devMatrix,4,&M,4) == CUBLAS_STATUS_SUCCESS);

		glm::mat4x4 temp = glm::inverse(toMat(M));
		Vec4 a = Vec4(temp[0][0], temp[0][1], temp[0][2], temp[0][3]);
		Vec4 b = Vec4(temp[1][0], temp[1][1], temp[1][2], temp[1][3]);
		Vec4 c = Vec4(temp[2][0], temp[2][1], temp[2][2], temp[2][3]);
		Vec4 d = Vec4(temp[3][0], temp[3][1], temp[3][2], temp[3][3]);
		return Mat4x4(a,b,c,d);
	}
	Mat3x3 matInverse(Mat3x3 M) {
		glm::mat3x3 temp = glm::inverse(toMat(M));
		Vec3 a = Vec3(temp[0][0], temp[0][1], temp[0][2]);
		Vec3 b = Vec3(temp[1][0], temp[1][1], temp[1][2]);
		Vec3 c = Vec3(temp[2][0], temp[2][1], temp[2][2]);
		return Mat3x3(a, b, c);
	}
	Mat4x4 matTranspose(Mat4x4 M) {
		Mat4x4 res;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				res.set(j, i, M.get(i, j));
			}
		}
		return res;
	}
	Mat3x3 matTranspose(Mat3x3 M) {
		Mat3x3 res;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				res.set(j, i, M.get(i, j));
			}
		}
		return res;
	}
	Vec4 matVecMult(Mat4x4 M, Vec4 v) {
		Vec4 temp;
		for (int i = 0; i < 4; i++) {
			float sum = 0.f;
			for(int j = 0; j < 4; j++){
				sum += M.get(j, i) * v.get(j);
			}
			temp.set(i, sum);
		}
		return temp;
	}
	Vec3 matVecMult(Mat4x4 M, Vec3 v) {
		Vec4 temp;
		for (int i = 0; i < 4; i++) {
			float sum = 0.f;
			for (int j = 0; j < 4; j++) {
				sum += M.get(j, i) * Vec4(v,1.f).get(j);
			}
			temp.set(i, sum);
		}
		return Vec3(temp.x,temp.y,temp.z);
	}
	Vec3 matVecMult(Mat3x3 M, Vec3 v) {
		Vec3 temp;
		for (int i = 0; i < 3; i++) {
			float sum = 0.f;
			for (int j = 0; j < 3; j++) {
				sum += M.get(j, i) * v.get(j);
			}
			temp.set(i, sum);
		}
		return temp;
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
	Mat4x4 tempMatrix = Mat4x4(Mat4x3(0.f)); //^^^^
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