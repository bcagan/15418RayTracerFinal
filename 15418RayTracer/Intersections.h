#pragma once

#include "Object.h"
#include "Defined.h"
#include "Ray.h"
#include "glm/glm.hpp"
#include <utility>


__device__ void swap(float a, float b) {
    float c(a); a = b; b = c;
}

__device__ bool sphereHit(Object& o, Ray& r, Hit& h) {
    float t0, t1;
    const Vec3 L = vecVecAdd(vecVecSub(o.t.pos, r.o), constVecMult(r.mint, r.d));
    const double tca = dot(L, r.d);
    // ignore if vector is facing the opposite way in any direction
    if (tca < 0) return false;
    const double d2 = dot(L, L) - tca * tca;
    const double radius2 = o.t.radius * o.t.radius;
    if (d2 > radius2) return false;
    const double thc = sqrt(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;

    if (t0 > t1) swap(t0, t1);

    if (t0 < 0) {
        t0 = t1; // if t0 is negative, let's use t1 instead 
        if (t0 < 0) return false; // both t0 and t1 are negative 
    }

    if (h.t >= t0 && t0 > r.mint) {
        if (t0 > r.mint) {
            h.t = t0;
        }
        else {
            h.t = r.mint;
        }
        h.normG = vecNormalize(vecVecSub(vecVecAdd(r.o, constVecMult(h.t, r.d)), o.t.pos));
        h.normS = h.normG;
        h.uv = Vec2(0.f);
    }
    return true;
}