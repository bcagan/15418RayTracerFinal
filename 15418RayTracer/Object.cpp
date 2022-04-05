#include "Object.h"

bool BBox::intersect(const Ray &r) {
    double tmin = -INFINITY, tmax = INFINITY;

    Vec3f invdir = 1 / r.d; 

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

    return tmax >= max(tmin, 0.0);
}