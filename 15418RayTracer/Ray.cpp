#include "Ray.h"
#include "Defined.h"
#include "Transform.h"

Ray Ray::transformRay(Transform transform) {
	Transform vecTransform = transform;
	vecTransform.pos = Vec3(0.f);
	Ray retRay;
	retRay.d = vecTransform.matVecMult(vecTransform.localToWorld(), d);
	retRay.o = transform.matVecMult(transform.localToWorld(), o);
	return retRay;
}


//Ray hit::bounce(Ray out); //Return ray bouncing (in opposite direction) into material that will result in out