#pragma once

#include "Object.h"
#include "Defined.h"
#include "Ray.h"
#include "glm/glm.hpp"
#include <utility>


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

    if (t0 > t1) std::swap(t0, t1);

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

__device__ bool cubeHit(Object& o, Ray& ray, Hit& hit, Hit& temp) {
    if (temp.t < ray.maxt) {
        hit.t = temp.t;
        hit.uv = temp.uv;
        hit.Mat = o.Mat;
        const Vec3 normVec = vecNormalize(vecVecAdd((vecVecAdd(ray.o, constVecMult(hit.t, ray.d))), constVecMult(-1.f, o.t.pos)));
        if (abs(normVec.x) > abs(normVec.y) && abs(normVec.x) > abs(normVec.z)) {
            if (normVec.x < 0) hit.normG = Vec3(-1.f, 0.f, 0.f);
            else hit.normG = Vec3(1.f, 0.f, 0.f);
        }
        else if (abs(normVec.y) > abs(normVec.x) && abs(normVec.y) > abs(normVec.z)) {
            if (normVec.y < 0) hit.normG = Vec3(0.f, -1.f, 0.f);
            else hit.normG = Vec3(0.f, 1.f, 0.f);
        }
        else {
            if (normVec.z < 0) hit.normG = Vec3(0.f, 0.f, -1.f);
            else hit.normG = Vec3(0.f, 0.f, 1.f);
        }
        hit.normS = hit.normG;
        hit.uv = Vec2(0.f);//Not doing right now
        ray.maxt = temp.t;
        return true;
    }
    return false;
}