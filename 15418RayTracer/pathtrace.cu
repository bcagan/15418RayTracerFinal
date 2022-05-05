#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>
#include <cmath>

#include "Scene.h"
#include "device_launch_parameters.h"
#include "Intersections.h"
#include "Scene.cpp"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

#define cudaCheckError(ans)  cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s at %s:%d\n",
			cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

static Scene* hst_scene = NULL;
static Vec3* dev_image = NULL;
static Object* dev_objs = NULL;
static Material* dev_materials = NULL;
static Ray* dev_rays = NULL;
static Hit* dev_hits = NULL;
static int* dev_hitPeaks = NULL;
static int* dev_hitIndices = NULL;


/////////////////////////////
//Scan code from assignment 2
/////////////////////////////

static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upsweepKernel(int N, int* data, int twod1, int twod) {
    // toWrite[0] = 51;
    // data[0] = 42;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = twod1 * index;
    //data[index] = data[i + twod1 -1] + data[i + twod -1] ;
    if (i < N) {
        int res = data[i + twod1 - 1] + data[i + twod - 1];
        data[i + twod1 - 1] = res;
    }
}

__global__ void downsweepKernel(int N, int* data, int twod1, int twod) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index * twod1;
    if (i < N) {
        int t = data[i + twod - 1] + data[i + twod1 - 1];
        int s = data[i + twod1 - 1];
        data[i + twod - 1] = s;
        data[i + twod1 - 1] = t;
    }
}

__global__ void set0(int N, int* deviceData) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == N - 1) deviceData[index] = 0;
    //else deviceData[index] = 1;
}

void exclusive_scan(int* device_data, int length) {
 
    int N = nextPow2(length);
    const int threadsPerBlock = 512;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    for (int twod = 1; twod < N; twod *= 2) {
        int twod1 = twod * 2;
        blocks = (N / twod1 + 1 + threadsPerBlock - 1) / threadsPerBlock;

        upsweepKernel <<<blocks, threadsPerBlock>>> (N, device_data, twod1, twod);
    }
    blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    set0 << <blocks, threadsPerBlock >> > (N, device_data);

    for (int twod = N / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        blocks = (N / twod1 + 1 + threadsPerBlock - 1) / threadsPerBlock;
        downsweepKernel << <blocks, threadsPerBlock >> > (N, device_data, twod1, twod);
    }

}



double cudaScan(int* inarray, int* end, int* resultarray) {
    int* device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaCheckError(cudaMalloc((void**)&device_data, sizeof(int) * rounded_length));

    cudaCheckError(cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice));

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost));
}



__global__ void intSet(int N, int* set, int to) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    set[index] = to;
}

__global__ void contractOut(int N, int* rays, int* indices, int* out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N && rays[index] != 0) {
        out[indices[index]] = index;
    }
}

void debugAssist(int* from, int N) {
    //    printf("deb\n");
	int printArr[1000]; // int printArr[N];
    cudaCheckError(cudaMemcpy(printArr, from, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int c = 0; c < N; c++) {
        printf("%d ", printArr[c]);
    }
    printf("\n\n\n");
}

int concat_rays(int num_rays, int numblocksPathSegmentTracing, int blockSize1d, int* device_output) {

    //Peaks have been found in dev_hitPeaks

    //Get number of elements
    int* device_num;
    cudaCheckError(cudaMalloc(&device_num, sizeof(int) * (num_rays + 1)));
    cudaScan(dev_hitPeaks, dev_hitPeaks + (num_rays + 1), device_num);
    int numberRays = 0;
    cudaCheckError(cudaMemcpy(&numberRays, device_num + (num_rays), sizeof(int), cudaMemcpyDeviceToHost));

    contractOut <<<numblocksPathSegmentTracing, blockSize1d >> > (num_rays, dev_hitPeaks, device_num, device_output);

    cudaCheckError(cudaDeviceSynchronize());
	cudaFree(device_num);

    return numberRays;

}

//New Path Tracer code

void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	const Camera& cam = hst_scene->cam;
	const int pixelcount = cam.resX * cam.resY;

	cudaMalloc(&dev_image, pixelcount * sizeof(Vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(Vec3));

	cudaMalloc(&dev_rays, pixelcount * sizeof(Ray));

	cudaMalloc(&dev_objs, scene->sceneObjs.size() * sizeof(Object));
	cudaMemcpy(dev_objs, scene->sceneObjs.data(), scene->sceneObjs.size() * sizeof(Object), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_hits, pixelcount * sizeof(Hit));
	cudaMemset(dev_hits, 0, pixelcount * sizeof(Hit));

	cudaMalloc(&dev_hitPeaks, (pixelcount+1) * sizeof(int));
	cudaMemset(dev_hitPeaks, 0, (pixelcount+1) * sizeof(int));

	cudaMalloc(&dev_hitIndices, (pixelcount) * sizeof(int));
	cudaMemset(dev_hitIndices, 0, (pixelcount) * sizeof(int));


}

void pathtraceFree() {
	cudaFree(dev_image);  
	cudaFree(dev_rays);
	cudaFree(dev_objs);
	cudaFree(dev_hits);
	cudaFree(dev_hitPeaks);
	cudaFree(dev_hitIndices);
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

		curandState state;
		curand_init(4321 + x, 0, 0, &state);

		float rand = curand_uniform_double(&state);


		float x = rand * (maxX - minX) + minX; 
		float y = rand * (maxY - minY) + minY; 
		float z = -cam.lensDistance; 
		
		Vec3 d = vecNormalize(Vec3(x, y, z));
		Vec3 o = Vec3(0.f);

		Transform vecTransform = cam.transform;
		vecTransform.pos = Vec3(0.f);
		ray.d = vecTransform.matVecMult(vecTransform.localToWorld(), d);
		ray.o = cam.transform.matVecMult(cam.transform.localToWorld(), o);
		ray.pixelIndex = index;

	}
}

__global__ void calculateColor(Camera cam, Ray* rays, Hit* hits, int iter, int num_rays)
{
	int ray_index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (ray_index < num_rays) {
		
		Ray& r = rays[ray_index];
		Hit& hit = hits[ray_index];

		Color3 rgb;

		if (hit.t != -1.0f) {
			// calculate bounce ray
			Vec3 bouncedHit = hit.bounce(r);
			Ray newR = Ray(vecVecAdd(constVecMult(hit.t, r.d), r.o), bouncedHit);
			newR.color = vecVecAdd(hit.emitted().toVec3(), vecVecMult(hit.albedo().toVec3(), r.color));

			// set up for next bounce 
			rays[ray_index] = newR;
			Vec3 pos = vecVecAdd(constVecMult(hit.t, r.d), r.o);
		}
		

	}
}


__global__ void computeIntersections(
	int depth, int num_rays, Ray* rays, int objs_size, Object* objs, Hit* hits, int* hitPeaks, int* hitIndices
)
{
	int ray_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (ray_index < num_rays)
	{
		Ray ray = rays[ray_index];

		float t;
		Vec3 intersect_point;
		Vec3 normal;
		float t_min = FLT_MAX;
		int hit_obj_index = -1;
		Hit h;

		for (int i = 0; i < objs_size; i++)
		{
			Object& obj = objs[i];
			Hit temp;

			if (obj.bbox.hit(ray, temp)) {
				if (obj.type == gcube) {
					cubeHit(obj, ray, h, temp);
				}
				else if (obj.type == gsphere) {
					sphereHit(obj, ray, h);
				}
				else {
					t_min = -1.0f;
					hit_obj_index = -1;
				}
				
			}
			
			
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_obj_index = i;
			}
		}
		
		if (hit_obj_index == -1)
		{
			hits[ray_index].t = -1.0f;
			hitPeaks[ray_index + 1] = 0;
		}
		else
		{
			//The ray hits something
			hits[ray_index].t = t_min;
			hits[ray_index].Mat = objs[hit_obj_index].Mat;
			hits[ray_index].normS = h.normS;
			hitPeaks[ray_index + 1] = 1;
			
		}

		
	}
}

__global__ void fillIndices(int num_rays, int* hitIndices) {
	int indexIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexIndex < num_rays) {
		hitIndices[indexIndex] = indexIndex;
	}
}

__global__ void finalGather(int num_rays, Vec3* image, Ray* rays)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < num_rays)
	{
		Ray ray = rays[index];
		image[ray.pixelIndex] += ray.color;
	}
}

void pathtrace(int frame, int iter) {
	// it might make more sense to define number of ray bounces in the scene rather than the ray
	const int traceDepth = 15;
	const Camera& cam = hst_scene->cam;
	const int pixelcount = cam.resX * cam.resY;
	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resX + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resY + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_rays);

	int depth = 0;
	Ray* dev_ray_end = dev_rays + pixelcount;
	int num_rays = dev_ray_end - dev_rays;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_hits, 0, pixelcount * sizeof(Hit));

		// tracing
		int numblocksPathSegmentTracing = (num_rays + blockSize1d - 1) / blockSize1d;

		fillIndices << <numblocksPathSegmentTracing, blockSize1d >> > (num_rays, dev_hitIndices); //Fill init indices to 1...num_rays

		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_rays
			, dev_rays
			, hst_scene->sceneObjs.size()
			, dev_objs
			, dev_hits
			, dev_hitPeaks
			, dev_hitIndices
			);

		calculateColor << <numblocksPathSegmentTracing, blockSize1d >> > (cam, dev_rays, dev_hits, depth, num_rays);

		
		cudaDeviceSynchronize();
		depth++;

		//Use find rays to contract rays into those that have ended and those that havent

		num_rays = concat_rays(num_rays, numblocksPathSegmentTracing, blockSize1d, dev_hitIndices);

		//printf("num paths: %i , depth: %i \n", num_paths, depth);
		if (num_rays == 0 || depth > traceDepth) {
			iterationComplete = true; // TODO: should be based off stream compaction results.
		}
	}
	num_rays = dev_ray_end - dev_rays;
	printf("Iteration Done\n");
	
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_rays, dev_image, dev_rays);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	// sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->cam.image.data(), dev_image,
		pixelcount * sizeof(Vec3), cudaMemcpyDeviceToHost);

}