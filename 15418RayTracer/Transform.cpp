#include "Transform.h"
#include "Defined.h"
#include "glm/glm.hpp"
//Noting the issues with euler angle rotations (gimble lock) I will
//Be using them over quaternions because quaternions are black magic

//Create transformation matrix from Transform
__device__ Mat4x4 Transform::makeTransform() {
	//Create rotation matrices
	//Will do out just to make CUDA translation possible
	//Mat3x3 defined by each column 

	//Rotation
	Mat3x3 Rx = Mat3x3(Vec3(1.f,0.f,0.f),Vec3(0.f,cos(rot.x),sin(rot.x)),Vec3(0.f,-sin(rot.x),cos(rot.x)));
	Mat3x3 Ry = Mat3x3(Vec3(cos(rot.y), 0.f, -sin(rot.y)), Vec3(0.f, 1.f,0.f ), Vec3(sin(rot.y), 0.f, cos(rot.y)));
	Mat3x3 Rz = Mat3x3(Vec3(cos(rot.z), sin(rot.z), 0.f), Vec3(-sin(rot.z),cos(rot.z) , 0.f), Vec3(0.f, 0.f, 1.f));
	Mat3x3 R = matMult(Rx, matMult(Ry, Rz));

	//Position
	Mat4x3 P = Mat4x3(Vec3(1.f,0.f,0.f), Vec3(0.f,1.f,0.f), Vec3(0.f,0.f,1.f),pos);

	//Scaling
	Mat3x3 S = Mat3x3(Vec3(scale.x,0.f,0.f), Vec3(0.f,scale.y,0.f), Vec3(0.f,0.f,scale.z));

	//Scale and rotate before position, scaling and rotation can be swapped 
	Mat3x3 RS = matMult(R, S);
	Mat4x4 preRes =  Mat4x4(matMult(P, RS));
	preRes.set(3,3,1.f);
	return preRes;
}




//Parent space transformations
//Transformation defines transform from parent to local space
__device__ Mat4x4 Transform::localToParent() {
	if (!tempMatrixFilled) makeAndSaveTransform();
	return tempMatrix; //So, take the local space, and apply the parent to local transform to get the parent space equivalent
}
__device__ Mat4x4 Transform::parentToLocal() {
	if (!tempMatrixFilled) makeAndSaveTransform();
	return matInverse(tempMatrix);
}

//World space transformations
__device__ Mat4x4 Transform::localToWorld() {
	if (!tempMatrixFilled) makeAndSaveTransform();
	Mat4x4 res = tempMatrix; //So, take the local spa
	if (parent != nullptr) res = matMult((parent->localToWorld()),  (Mat4x4)res);
	return res;
}

__device__ Mat4x4 Transform::worldToLocal() {
	if (!tempMatrixFilled) makeAndSaveTransform();
	Mat4x4 res = tempMatrix; //So, take the local spa
	if (parent != nullptr) res = matMult((parent->localToWorld()) , (Mat4x4)res);
	return matInverse(res);

}