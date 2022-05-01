#include <cstdio>
#include <cuda.h>
#include <cmath>

#include "Scene.h"
#include "Scene.cpp"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	const Camera& cam = hst_scene->state.cam;
	const int pixelcount = cam.resX * cam.resY;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_rays, pixelcount * sizeof(Ray));

	cudaMalloc(&dev_objs, scene->sceneObjs.size() * sizeof(Object));
	cudaMemcpy(dev_objs, scene->sceneObjs.data(), scene->sceneObjs.size() * sizeof(Object), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_hits, pixelcount * sizeof(Hit));
	cudaMemset(dev_hits, 0, pixelcount * sizeof(Hit));


}

void pathtraceFree() {
	cudaFree(dev_image);  
	cudaFree(dev_paths);
	cudaFree(dev_objs);
	cudaFree(dev_intersections);

}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, Ray* rays)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resX && y < cam.resY) {
		int index = x + (y * cam.resX);
		Ray& ray = rays[index];

		float sizeY = 2.f * cam.lensDistance * tan(cam.vFov);
		float sizeX = (float)cam.resX / (float)cam.resY * sizeY;
		float minX = (float)x / (float)cam.resX * sizeX - sizeX / 2.f;
		float maxX = (float)(x + 1) / (float)cam.resX * sizeX - sizeX / 2.f;
		float minY = (float)y / (float)cam.resY * sizeY - sizeY / 2.f;
		float maxY = (float)(y + 1) / (float)cam.resY * sizeY - sizeY / 2.f;
		float x = randf() * (maxX - minX) + minX; 
		float y = randf() * (maxY - minY) + minY; 
		float z = -cam.lensDistance; 
		
		Vec3 d = vecNormalize(Vec3(x, y, z));
		Vec3 o = Vec3(0.f);

		Transform vecTransform = cam.transform;
		vecTransform.pos = Vec3(0.f);
		ray.d = vecTransform.matVecMult(vecTransform.localToWorld(), d);
		ray.o = cam.transform.matVecMult(cam.transform.localToWorld(), o);

	}
}

__global__ void computeIntersections(
	int depth, int num_rays, Ray* rays, int objs_size, Objs* objs, Hit* hits
)
{
	int ray_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (ray_index < num_rays)
	{
		Ray ray = rays[ray_index];

		if (pathSegment.pixelIndex == 400) {
			int pixelIndex = pathSegment.pixelIndex;
			pixelIndex++;
		}

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_obj_index = -1;
		Hit& h;

		for (int i = 0; i < objs_size; i++)
		{
			Object& obj = objs[i];

			
			t = obj.hit(ray, h)
			
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
			}
		}

		if (hit_geom_index == -1)
		{
			hits[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			hits[ray_index].t = t_min;
			hits[ray_index].Mat = objs[hit_obj_index].Mat;
			hits[ray_index].surfaceNormal = h.normS;
		}
	}
}

void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resX * cam.resY;
	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resX + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resY + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_rays);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_ray_end = dev_rays + pixelcount;
	int num_rays = dev_ray_end - dev_rays;

	startCpuTimer();
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_hits, 0, pixelcount * sizeof(Hit));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_rays + blockSize1d - 1) / blockSize1d;

		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_rays
			, dev_rays
			, dev_objs
			, hst_scene->objs.size()
			, dev_hits
			);

		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;


		//printf("num paths: %i , depth: %i \n", num_paths, depth);
		if (num_rays == 0 || depth > traceDepth) {
			iterationComplete = true; // TODO: should be based off stream compaction results.
		}
	}
	num_rays = dev_ray_end - dev_rays;
	printf("Iteration Done\n");
	endCpuTimer();
	printTime();
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	// finalGather << <numBlocksPixels, blockSize1d >> > (num_rays, dev_image, dev_rays);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->cam.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}