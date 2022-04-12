#include "Scene.h"
#include "Defined.h"
#include "Transform.h"
#include "Ray.h"
#include "Camera.h"
#include "glm/glm.hpp" 
#include <memory>

void Scene::addObj(Object o) {
    sceneObjs.push_back(o);
}

bool Scene::intersect(Ray ray, Hit& hit) {
    //Presume transform is just position, not rotation or scaling, so transform defines objects world space pos
    bool hitBool = false;
    for (auto obj : sceneObjs) {
        Hit temp;
        if (obj.bbox.hit(ray,temp)) {
            if (obj.hit(ray, hit)) {
                if(hit.t < ray.maxt) ray.maxt = hit.t;
                hitBool = true; //hit itself must be updated here
            }
        }
    }
    return hitBool;
}

// helper function for render (recursion as deep as the original ray's bounce count)
Color3 Scene::renderC(Ray r, int numBounces) {
    if (numBounces > 0) {
        Hit hit = Hit(); //initialize hit here
        if (Scene::intersect(r, hit)) {
            Ray newR = hit.bounce(r);
            Vec3 colorVec = hit.emitted().toVec3() + hit.albedo().toVec3() * renderC(newR, numBounces - 1).toVec3();
            return Color3(colorVec);
        }
    }

    return background;

}
void Scene::render() {

    // resolution
    int height = cam.resY;
    int width = cam.resX;
    //std::unique_ptr<Vec3[width * height]> frame(new Vec3[width * height]);
    //Vec3* pix = frame.get();
    float lensDistance = cam.lensDistance;
    float vFov = cam.vFov;
    // float scale = tan(deg2rad(options.fov * 0.5));
    // float imageAspectRatio = width / (float)height;
    
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            Color3 rgb;
            int sampleCount = 0;
            std::unique_ptr<Color3[]> samples(new Color3[numSamples]);
            Color3* s = samples.get(); 

            while (sampleCount < numSamples) {
                Ray r = cam.castRay(i, j);
                rgb = renderC(r, r.numBounces);
                *(s++) = rgb;
               
                sampleCount++;
            }

            sampleCount = 1;
            s = samples.get();
            rgb = *s;
            s++;
            while(sampleCount < numSamples) {
                rgb.r += s->r;
                rgb.g += s->g;
                rgb.b += s->b;
                s++;
                sampleCount++;
            }
            rgb.r /= (float)sampleCount;
            rgb.g /= (float)sampleCount;
            rgb.b /= (float)sampleCount;
            
            //*(pix++) = rgb.toVec3();
            cam.image[j][i] = rgb;
        }
    }
}

