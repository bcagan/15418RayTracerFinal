#include "Object.h"
#include "Defined.h"
#include "Ray.h"
#include "glm/glm.hpp"
#include <utility>

__device__ double sstdmin(double a, double b) {
    if (a > b) return b;
    else return a;
}

__device__ double sstdmax(double a, double b) {
    if (a > b) return a;
    else return b;
}

__device__ bool BBox::hit( Ray &r, Hit& hit) {
    double tmin = -INFINITY, tmax = INFINITY;

    Vec3 invdir = vecVecDiv(Vec3(1.f), r.d); 

    // value of t in the parametric ray equation where ray intersects min coordinate with dimension i
    double t1 = (min.x - r.o.x) * invdir.x;
    // value of t in the parametric ray equation where ray intersects max coordinate with dimension i
    double t2 = (max.x - r.o.x) * invdir.x;

    tmin = sstdmax(tmin, sstdmin(t1, t2));
    tmax = sstdmin(tmax, sstdmax(t1, t2));

    t1 = (min.y - r.o.y) * invdir.y;
    t2 = (max.y - r.o.y) * invdir.y;

    tmin = sstdmax(tmin, sstdmin(t1, t2));
    tmax = sstdmin(tmax, sstdmax(t1, t2));

    t1 = (min.z - r.o.z) * invdir.z;
    t2 = (max.z - r.o.z) * invdir.z;

    tmin = sstdmax(tmin, sstdmin(t1, t2));
    tmax = sstdmin(tmax, sstdmax(t1, t2));

    if(r.maxt >= tmin && tmin > EPSILON){
        hit.t = tmin;
        Vec3 pos = vecVecDiv(vecVecAdd(max , min) , Vec3(2.0f));
        hit.uv = Vec2(0.f);
        return true;
    }
    return false;
}

__device__ bool Sphere::hit( Ray& r, Hit& h) {
   
    double t0, t1;
    Vec3 L = vecVecAdd(vecVecSub(t.pos , r.o) , constVecMult(r.mint,r.d));
    double tca = dot(L, r.d);
    // ignore if vector is facing the opposite way in any direction
    if (tca < 0) return false;
    double d2 = dot(L, L) - tca * tca;
    double radius2 = radius * radius;
    if (d2 > radius2) return false;
    double thc = sqrt(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;

    // if (t0 > t1) std::swap(t0, t1);

    if (t0 < 0) {
        t0 = t1; // if t0 is negative, let's use t1 instead 
        if (t0 < 0) return false; // both t0 and t1 are negative 
    }

    if (h.t >= t0 && t0 > r.mint) {
        h.t = std::max((float)t0, r.mint);
        h.normG = vecNormalize(vecVecSub(vecVecAdd(r.o , constVecMult( h.t , r.d)) , t.pos));
        h.normS = h.normG;
        h.uv = Vec2(0.f);
    }
    return true;
}
