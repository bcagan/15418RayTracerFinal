#include "Scene.h"
#include "Defined.h"
#include "Transform.h"
#include "Ray.h"
#include "Camera.h"
#include "glm/glm.hpp" 

bool Scene::intersect(Ray ray, Hit& hit) {
    // use BVH tree / seq. BBox calculation
    return true;
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

    //assume background is black for now
    return Color3(0, 0, 0);

}
void Scene::render() {

    // resolution
    int height = cam.resY;
    int width = cam.resX;
    std::unique_ptr<Vec3[]> frame(new Vec3[width * height]);
    Vec3* pix = frame.get();
    float lensDistance = cam.lensDistance;
    float vFov = cam.vFov;
    // float scale = tan(deg2rad(options.fov * 0.5));
    // float imageAspectRatio = width / (float)height;
    
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            Color3 rgb;
            int sampleCount = 0;
            std::unique_ptr<Vec3[]> samples(new Vec3[this->numSamples]);
            Vec3* s = samples.get();

            while (sampleCount < this->numSamples) {
                // from scratchapixel, not sure if this is correct, currently just emits rays from the pixel location
                // 
                // int x = (2 * (i + 0.5) / (float)width - 1) * imageAspectRatio * scale;
                // int y = (1 - 2 * (j + 0.5) / (float)height) * scale;
                Ray r = cam.castRay(i, j);
                rgb = renderC(r, r.numBounces);
                *(s++) = rgb.toVec3();
               
                sampleCount++;
            }

            // should combine the samples here (currently just uses the last sample)
            *(pix++) = rgb.toVec3();
            
        }
    }

    // all rgb values are stored in frame
    // should be used here to actually create the image
	
}

