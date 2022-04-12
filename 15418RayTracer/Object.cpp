#include "Object.h"
#include "Defined.h"
#include "Ray.h"

bool BBox::hit(const Ray &r, Hit& hit) {
    /*double tmin = -INFINITY, tmax = INFINITY;

    Vec3f invdir = 1.f / r.d; 

    // value of t in the parametric ray equation where ray intersects min coordinate with dimension i
    double t1 = (min.x - r.o.x) * invdir.x;
    // value of t in the parametric ray equation where ray intersects max coordinate with dimension i
    double t2 = (max.x - r.o.x) * invdir.x;

    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));

    double t1 = (min.y - r.o.y) * invdir.y;
    double t2 = (max.y - r.o.y) * invdir.y;

    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));

    double t1 = (min.z - r.o.z) * invdir.z;
    double t2 = (max.z - r.o.z) * invdir.z;

    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));

    if(tmax >= max(tmin, 0.0f)){
        hit.t = max(tmin, 0.0f);
        return true;
    }*/
    return false;
}

bool Sphere::hit(const Ray &r, Hit& hit) {
    /*
    float t0, t1;

    Vec3 L = center - r.o;
    float tca = dot(L, dir);
    // ignore if vector is facing the opposite way in any direction
    if (tca < 0) return false;
    float d2 = dot(L, L) - tca * tca;
    float radius2 = radius * radius;
    if (d2 > radius2) return false;
    float thc = sqrt(radius2 - d2);
    //t0 = tca - thc;
    //t1 = tca + thc;
    */

    return true;
}
