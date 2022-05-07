#include "Scene.h"
#include "Defined.h"
#include "Transform.h"
#include "Ray.h"
#include "Camera.h"
#include "glm/glm.hpp" 
#include <memory>
#include <io.h>
#include <iostream>


//Note while Vec3 is used in this code, it will instead be using a redesigned struct, as using cuda's LA library wouldnt be useful here

void Scene::addObj(Object o) {
    sceneObjs.push_back(o);
}
void Scene::addObjseq(Object* o) {
    sceneObjss.push_back(o);
}

__device__ bool Scene::intersect(Ray ray, Hit& hit) {
    //Presume transform is just position, not rotation or scaling, so transform defines objects world space pos
    bool hitBool = false;
    for (int o = 0; o < sceneObjs.size(); o++) {
        auto obj = sceneObjss[o];
        Hit temp;
        if (obj->bbox.hit(ray,temp)) {
            if (obj->hit(ray, hit)) {
                if (hit.t < ray.maxt) {
                    ray.maxt = hit.t;
                    hit.Mat = obj->Mat;
                }
                hitBool = true;
            }
            
        }
    }
    return hitBool;
}

// helper function for render (recursion as deep as the original ray's bounce count)
__device__ Color3 Scene::renderC(Ray r, int numBounces) {
    if (numBounces > 0) {
        Hit hit = Hit(); //initialize hit here
        if (Scene::intersect(r, hit)) { 
            printf("oldr direction x: %f y: %f z: %f  \n", r.d.x, r.d.y, r.d.z);
            Vec3 bouncedHit = hit.bounce(r);
            Ray newR = Ray(vecVecAdd(constVecMult(hit.t , r.d) , r.o), bouncedHit);
            printf("newr direction x: %f y: %f z: %f  \n", newR.d.x, newR.d.y, newR.d.z);
            Vec3 renderCRes = renderC(newR, numBounces - 1).toVec3();
            Vec3 pos = vecVecAdd(constVecMult(hit.t, r.d) , r.o);
            Vec3 colorVec = vecVecAdd(hit.emitted().toVec3() , vecVecMult(hit.albedo().toVec3() , renderCRes));
            return Color3(colorVec) ;
        }
    }

    
    return background;

}

__device__ void Scene::render() {

    // resolution
    int height = cam.resY;
    int width = cam.resX;
    //std::unique_ptr<Vec3[width * height]> frame(new Vec3[width * height]);
    //Vec3* pix = frame.get();
    float lensDistance = cam.lensDistance;
    float vFov = cam.vFov;
    // float scale = tan(deg2rad(options.fov * 0.5));
    // float imageAspectRatio = width / (float)height;
    int lastPerc = 0;
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
                //All are black here
                //if(rgb.r != 0)std::cout << sampleCount << " sample " << j << "row " << i << "col " << (int)rgb.r << " r " << (int)rgb.g << " g " << (int)rgb.b << "b\n";
               
                sampleCount++;
            }

            sampleCount = 0;
            s = samples.get();
            std::vector <int> col(3);
            while(sampleCount < numSamples) {
                col[0] += s->r;
                col[1] += s->g;
                col[2] += s->b;
                s++;
                sampleCount++;
            }

            rgb.r = (unsigned char)(col[0] /= sampleCount);
            rgb.g = (unsigned char)(col[1] /= sampleCount);
            rgb.b = (unsigned char)(col[2] /= sampleCount);

            //*(pix++) = rgb.toVec3();
            cam.image[j][i] = rgb;
            int idx = j * width + i + 1;
            int per = 100 * idx / (width * height);
            if (per > lastPerc) {
                lastPerc = per;
                // std::cout << "Render: " << per << "%\n";
                printf("Render: %d percent \n", per);
            }
        }
    }
}

