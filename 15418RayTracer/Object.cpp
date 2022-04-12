#include "Object.h"
#include "Defined.h"
#include "Ray.h"
#include "glm/glm.hpp"
#include <utility>

bool BBox::hit(const Ray &r, Hit& hit) {
    double tmin = -INFINITY, tmax = INFINITY;

    Vec3 invdir = 1.f / r.d; 

    // value of t in the parametric ray equation where ray intersects min coordinate with dimension i
    double t1 = (min.x - r.o.x) * invdir.x;
    // value of t in the parametric ray equation where ray intersects max coordinate with dimension i
    double t2 = (max.x - r.o.x) * invdir.x;

    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));

    double t1 = (min.y - r.o.y) * invdir.y;
    double t2 = (max.y - r.o.y) * invdir.y;

    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));

    double t1 = (min.z - r.o.z) * invdir.z;
    double t2 = (max.z - r.o.z) * invdir.z;

    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));

    if(tmax >= std::max(tmin, 0.0)){
        hit.t = std::max(tmin, 0.0);
        return true;
    }
    return false;
}

bool Sphere::hit(const Ray &r, Hit& hit) {
    
    float t0, t1;

    Vec3 L = t.pos - r.o;
    float tca = dot(L, r.d);
    // ignore if vector is facing the opposite way in any direction
    if (tca < 0) return false;
    float d2 = dot(L, L) - tca * tca;
    float radius2 = radius * radius;
    if (d2 > radius2) return false;
    float thc = sqrt(radius2 - d2);
    //t0 = tca - thc;
    //t1 = tca + thc;
    

    return true;
}
