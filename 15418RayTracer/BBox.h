#pragma once
#include "glm/glm.hpp"

class BBox 
{
public:
    Vec3 min;
    Vec3 max;
    BBox(const Vec3 &vmin, const Vec3 &vmax) {
        min = vmin;
        max = vmax;
    }

    bool intersect(const Ray &r);
};

