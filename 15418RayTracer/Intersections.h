#pragma once

#include "Object.h"
#include "Defined.h"
#include "Ray.h"
#include "glm/glm.hpp"
#include <utility>

__device__ double stdmin(double a, double b) {
    if (a > b) return b;
    else return a;
}

__device__ double stdmax(double a, double b) {
    if (a > b) return a;
    else return b;
}

__device__ bool bboxHit(BBox& b, Ray& r, Hit& hit) {
    double tmin = -INFINITY, tmax = INFINITY;

    Vec3 invdir = vecVecDiv(Vec3(1.f), r.d);

    // value of t in the parametric ray equation where ray intersects min coordinate with dimension i
    double t1 = (b.min.x - r.o.x) * invdir.x;
    // value of t in the parametric ray equation where ray intersects max coordinate with dimension i
    double t2 = (b.max.x - r.o.x) * invdir.x;

    tmin = stdmax(tmin, stdmin(t1, t2));
    tmax = stdmin(tmax, stdmax(t1, t2));

    t1 = (b.min.y - r.o.y) * invdir.y;
    t2 = (b.max.y - r.o.y) * invdir.y;

    tmin = stdmax(tmin, stdmin(t1, t2));
    tmax = stdmin(tmax, stdmax(t1, t2));

    t1 = (b.min.z - r.o.z) * invdir.z;
    t2 = (b.max.z - r.o.z) * invdir.z;

    tmin = stdmax(tmin, stdmin(t1, t2));
    tmax = stdmin(tmax, stdmax(t1, t2));

    //printf("hit: %f %f \n", r.maxt, tmin);

    if (r.maxt >= tmin && tmin > EPSILON) {
        hit.t = tmin;
        Vec3 pos = vecVecDiv(vecVecAdd(b.max, b.min), Vec3(2.0f));
        hit.uv = Vec2(0.f);
        return true;
    }
    return false;
}


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