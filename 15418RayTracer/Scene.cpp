#include "Scene.h"
#include "Defined.h"
#include "Transform.h"
#include "Ray.h"
#include "Camera.h"
#include "glm/glm.hpp" 
#include <memory>
#include <io.h>
#include <iostream>

void Scene::addObj(Object* o) {
    sceneObjs.push_back(o);
}

bool Scene::intersect(Ray ray, Hit& hit) {
    //Presume transform is just position, not rotation or scaling, so transform defines objects world space pos
    bool hitBool = false;
    for (int o = 0; o < sceneObjs.size(); o++) {
        auto obj = sceneObjs[o];
        Hit temp;
        if (true){//(obj->bbox.hit(ray,temp)) {
            if (obj->hit(ray, hit)) {
                if (hit.t < ray.maxt) {
                    ray.maxt = hit.t;
                    hit.Mat = obj->Mat;
                }
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
            Vec3 bouncedHit = hit.bounce(r);
            Ray newR = Ray(hit.t * r.d + r.o, bouncedHit);
            Vec3 renderCRes = renderC(newR, numBounces - 1).toVec3();
            Vec3 pos = hit.t * r.d + r.o;
            //if (hit.emitted().toVec3().x == 0 && abs(bouncedHit.y) < 0.1f && abs(bouncedHit.z) < 0.1f  && hit.normS.x > 0.95f) std::cout << "the bounce was" << bouncedHit.x << " " << bouncedHit.y << " " << bouncedHit.z <<
            //    "while the surface hit pos was " << pos.x << " " << pos.y << " " << pos.z << std::endl;
            //if(numBounces < 15) std::cout << "num bounces " << numBounces << std::endl;
            Vec3 colorVec = hit.emitted().toVec3() + hit.albedo().toVec3() * renderCRes;
            //if (hit.emitted().toVec3().x == 0 && abs(bouncedHit.y) < 0.1f && abs(bouncedHit.z) < 0.1f  && hit.normS.x > 0.95f) std::cout << " albedo " << colorVec.x << " " << colorVec.y << " " << colorVec.z << std::endl;
            //if (numBounces < 15 ) std::cout << " outColor == " << colorVec.x << " " << colorVec.y << " " << colorVec.z << " and the emitted r " << (int) hit.emitted().r << std::endl;
            return Color3(colorVec) ;
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
                std::cout << "Render: " << per << "%\n";
            }
        }
    }
}

