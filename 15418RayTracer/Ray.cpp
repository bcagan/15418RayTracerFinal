#include "Ray.h"
#include "Defined.h"
#include "Transform.h"

__device__ Ray Ray::transformRay(Transform transform) {
	Transform vecTransform = transform;
	vecTransform.pos = Vec3(0.f);
	Ray retRay;
	retRay.d = vecTransform.matVecMult(vecTransform.localToWorld(), d);
	retRay.o = transform.matVecMult(transform.localToWorld(), o);
	// printf("ray: %i %i %i \n", retRay.d.x, retRay.d.y, retRay.d.z);

	return retRay;
}


//Derived for 15-468's DIRT path tracer base code
__device__ inline Vec3 randomOnUnitSphere(float cosphi, float theta){

	float sinphi = sqrt(1.f - cosphi * cosphi);
	float x = cos(theta) * sinphi;
	float z = sin(theta) * sinphi;
	float y = cosphi;
	return Vec3(x, y, z);
}



Vec3 Hit::bounce(Ray out) {
	float theta = 2.f* randf()*PI;
	float cosphi = 2.f * randf() - 1.f;
	return vecNormalize(vecVecAdd(normS ,randomOnUnitSphere(cosphi, theta)));
}