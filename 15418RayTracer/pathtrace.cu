#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>
#include <cmath>

#include "Scene.h"
#include "device_launch_parameters.h"
#include "Intersections.h"
#include "Ray.h"
//#include "Ray.cpp"
#include "Transform.h"
//#include "Transform.cpp"
#include "Object.h"
//#include "Object.cpp"
//#include "Scene.cpp"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"


//https://stackoverflow.com/questions/6061565/setting-up-visual-studio-intellisense-for-cuda-kernel-calls
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

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
static Color3* dev_image = NULL;
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

        upsweepKernel CUDA_KERNEL(blocks, threadsPerBlock) (N, device_data, twod1, twod);
		//upsweepKernel<<<blocks,threadsPerBlock>>>(N, device_data, twod1, twod);
    }
    blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    set0 CUDA_KERNEL(blocks, threadsPerBlock) (N, device_data);

    for (int twod = N / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        blocks = (N / twod1 + 1 + threadsPerBlock - 1) / threadsPerBlock;
        downsweepKernel CUDA_KERNEL(blocks, threadsPerBlock) (N, device_data, twod1, twod);
    }

}



void cudaScan(int* inarray, int* end, int* resultarray) {
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
	int rounded_length = nextPow2(num_rays);

    cudaCheckError(cudaMalloc(&device_num, sizeof(int) * (rounded_length)));
	
    cudaScan(dev_hitPeaks, dev_hitPeaks + (num_rays + 1), device_num);
    int numberRays = 0;
    cudaCheckError(cudaMemcpy(&numberRays, device_num + (num_rays), sizeof(int), cudaMemcpyDeviceToHost));
	
    contractOut CUDA_KERNEL(numblocksPathSegmentTracing, blockSize1d) (num_rays, dev_hitPeaks, device_num, device_output);

    cudaCheckError(cudaDeviceSynchronize());
	cudaFree(device_num);

    return numberRays;

}

//New Path Tracer code

void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	const Camera& cam = hst_scene->cam;
	const int pixelcount = cam.resX * cam.resY;

	cudaMalloc(&dev_image, pixelcount * sizeof(Color3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(Color3));

	cudaMalloc(&dev_rays, pixelcount * sizeof(Ray));
	cudaMemset(dev_rays, 0, pixelcount * sizeof(Ray));

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

__device__ Mat4x4 tmakeTransform(Transform* t) {
	//Create rotation matrices
	//Will do out just to make CUDA translation possible
	//Mat3x3 defined by each column 

	//Rotation
	Mat3x3 Rx = Mat3x3(Vec3(1.f, 0.f, 0.f), Vec3(0.f, cos(t->rot.x), sin(t->rot.x)), Vec3(0.f, -sin(t->rot.x), cos(t->rot.x)));
	Mat3x3 Ry = Mat3x3(Vec3(cos(t->rot.y), 0.f, -sin(t->rot.y)), Vec3(0.f, 1.f, 0.f), Vec3(sin(t->rot.y), 0.f, cos(t->rot.y)));
	Mat3x3 Rz = Mat3x3(Vec3(cos(t->rot.z), sin(t->rot.z), 0.f), Vec3(-sin(t->rot.z), cos(t->rot.z), 0.f), Vec3(0.f, 0.f, 1.f));
	Mat3x3 R = t->matMult(Rx, t->matMult(Ry, Rz));

	//Position
	Mat4x3 P = Mat4x3(Vec3(1.f, 0.f, 0.f), Vec3(0.f, 1.f, 0.f), Vec3(0.f, 0.f, 1.f), t->pos);

	//Scaling
	Mat3x3 S = Mat3x3(Vec3(t->scale.x, 0.f, 0.f), Vec3(0.f, t->scale.y, 0.f), Vec3(0.f, 0.f, t->scale.z));

	//Scale and rotate before position, scaling and rotation can be swapped 
	Mat3x3 RS = t->matMult(R, S);
	Mat4x4 preRes = Mat4x4(t->matMult(P, RS));
	preRes.set(3, 3, 1.f);
	t->tempMatrix = preRes;
}

__device__ Mat4x4 localToWorld(Transform t) {
	if (!t.tempMatrixFilled) tmakeTransform(&t);
	Mat4x4 res = t.tempMatrix; //So, take the local spa
	//printf("parent: %lu\n", (unsigned long) t.parent);
	Transform* tp = t.parent;
	while (tp != nullptr) {
		res = (t.parent)->tempMatrix * res;
		tp = t.parent;
	}
	/*if (t.parent != nullptr) 
	res = localToWorld(*(t.parent)) * res;*/
	return res;
}

__global__ void generateRayFromCamera(Camera cam, int traceDepth, Ray* rays)
{
	int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
	int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (ix < cam.resX && iy < cam.resY) {
		int index = ix + (iy * cam.resX);

		// printf("Hello from pixel %d\n", index);
		Ray& ray = rays[index];

		float sizeY = 2.f * cam.lensDistance * tan(cam.vFov);
		float sizeX = (float)cam.resX / (float)cam.resY * sizeY;
		float minX = (float)ix / (float)cam.resX * sizeX - sizeX / 2.f;
		float maxX = (float)(ix + 1) / (float)cam.resX * sizeX - sizeX / 2.f;
		float minY = (float)iy / (float)cam.resY * sizeY - sizeY / 2.f;
		float maxY = (float)(iy + 1) / (float)cam.resY * sizeY - sizeY / 2.f;

		curandState state;
		curand_init(12345 + index, 0, 0, &state);

		float rand1 = curand_uniform_double(&state);
		float rand2 = curand_uniform_double(&state);


		float x = rand1 * (maxX - minX) + minX; 
		float y = rand2 * (maxY - minY) + minY; 
		float z = -cam.lensDistance; 

		
		Vec3 d = vecNormalize(Vec3(x, y, z));
		Vec3 o = Vec3(0.f);

		Transform vecTransform = cam.transform;
		vecTransform.pos = Vec3(0.f);

		ray.d = vecTransform.matVecMult(localToWorld(vecTransform), d);
		ray.o = cam.transform.matVecMult(localToWorld(cam.transform), o);
		ray.pixelIndex = index;
		ray.maxt = INFINITY;
		ray.mint = EPSILON;
		ray.numBounces = traceDepth;
		ray.color = Color3().toVec3();
		ray.storeColor = Vec3(1.f);

		//printf("pixX %d pixY %d minX maxX miny maxY %f %f %f %f x y %f %f \n", ix, iy, minX, maxX, minY, maxY, x, y);
		// if(index == 200) printf("xx y z %f %f %f \n", rays[index].maxt, rays[index].mint, rays[index].d.z);

	}
}

__device__ inline Vec3 rrandomOnUnitSphere(float cosphi, float theta) {

	float sinphi = sqrt(1.f - cosphi * cosphi);
	float x = cos(theta) * sinphi;
	float z = sin(theta) * sinphi;
	float y = cosphi;
	return Vec3(x, y, z);
}

__device__ Vec3 bounce(Hit* hits, int ray_index) {
	Hit& h = hits[ray_index];
	curandState state;
	curand_init(4321 + ray_index, 0, 0, &state);

	float rand1 = curand_uniform_double(&state);
	float rand2 = curand_uniform_double(&state);

	float theta = 2.f * rand1 * PI;
	float cosphi = 2.f * rand2 - 1.f;
	return vecNormalize(vecVecAdd(h.normS, rrandomOnUnitSphere(cosphi, theta)));
}

__global__ void calculateColor(Camera cam, Ray* rays, Hit* hits, int iter, int num_rays)
{
	int ray_index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (ray_index < num_rays) {
		
		Ray& r = rays[ray_index];
		Hit& hit = hits[ray_index];

		// printf("hit: %f \n", hit.t);
		// if(ray_index == 200) printf("x y z %f %f %f \n", rays[ray_index].maxt, rays[ray_index].mint, rays[ray_index].d.z);

		if (hit.t != -1.0f) {
			// calculate bounce ray
			/*if (ray_index == 200) {
				printf("pre rgb r: %f, g: %f, b: %f asdf %f %f %f \n", r.color.x, r.color.y, r.color.z, hit.albedo().toVec3().x, hit.albedo().toVec3().y, hit.albedo().toVec3().z);
				printf("emitasdf %f %f %f ", hit.emitted().toVec3().x, hit.emitted().toVec3().y, hit.emitted().toVec3().z);
			}*/
			Vec3 bouncedHit = bounce(hits, ray_index);
			Ray newR = Ray(vecVecAdd(constVecMult(hit.t, r.d), r.o), bouncedHit);
			newR.color += hit.emitted().toVec3() * r.storeColor;
			newR.storeColor = r.storeColor * hit.albedo().toVec3();
			//printf("original store color %f  %f %f current color %f %f %f \n", r.storeColor.x, r.storeColor.y, r.storeColor.z, r.color.x, r.color.y, r.color.z);


			//printf("ray: %f %f %f hit %f color %f %f %f \n", r.d.x, r.d.y, r.d.z, hit.t, newR.color.x, newR.color.y, newR.color.z);
			// set up for next bounce 
			r.d = newR.d;
			r.o = newR.o;
			r.mint = EPSILON;
			r.maxt = INFINITY;
			r.color = newR.color;
			r.storeColor = newR.storeColor;
			r.numBounces--;
			//if(r.numBounces > 2) printf("store color %f  %f %f current color %f %f %f \n", r.storeColor.x, r.storeColor.y, r.storeColor.z, r.color.x, r.color.y, r.color.z);
			

			if (ray_index == 23523) {
				printf("rgb r: %f, g: %f, b: %f \n", r.numBounces, r.color.y, r.color.z);
			}

		}
		

	}
}

__device__ double stdmin(double a, double b) {
	if (a > b) return b;
	else return a;
}

__device__ double stdmax(double a, double b) {
	if (a > b) return a;
	else return b;
}

//__device__ bool bboxHit(Object& o, Ray& r, Hit& hit, int ray_index) {
__device__ bool bboxHit(int obj_index, int ray_index, Ray* rays, Object* objs, Hit* hits) {
	double tmin = -INFINITY, tmax = INFINITY;
	BBox& b = objs[obj_index].bbox;
	Ray& r = rays[ray_index];
	//Hit& hit = hits[ray_index];

	/*if (ray_index == 200 && obj_index == 0) {
		printf("bbox min x y z %f %f %f\n", b.min.x, b.min.y, b.min.z);
		printf("bbox max x y z %f %f %f\n", b.max.x, b.max.y, b.max.z);
		printf("ray x y z %f %f %f\n", r.d.x, r.d.y, r.d.z);
	}*/

	Vec3 invdir = vecVecDiv(Vec3(1.f), r.d);

	// value of t in the parametric ray equation where ray intersects min coordinate with dimension i
	double t1 = (b.min.x - r.o.x) * invdir.x;
	// value of t in the parametric ray equation where ray intersects max coordinate with dimension i
	double t2 = (b.max.x - r.o.x) * invdir.x;

	tmin = stdmax(tmin, stdmin(t1, t2));
	tmax = stdmin(tmax, stdmax(t1, t2));

	t1 = (b.min.y - r.o.y) * invdir.y;
	t2 = (b.max.y - r.o.y) * invdir.y;

	tmin = stdmax(tmin, stdmin(t1, t2));
	tmax = stdmin(tmax, stdmax(t1, t2));

	t1 = (b.min.z - r.o.z) * invdir.z;
	t2 = (b.max.z - r.o.z) * invdir.z;

	tmin = stdmax(tmin, stdmin(t1, t2));
	tmax = stdmin(tmax, stdmax(t1, t2));

	// printf("hit: %f %f \n", r.maxt, tmin);

	if (r.maxt >= tmin && tmin > EPSILON) {
		hits[ray_index].t = tmin;
		Vec3 pos = vecVecDiv(vecVecAdd(b.max, b.min), Vec3(2.0f));
		hits[ray_index].uv = Vec2(0.f);
		return true;
	}
	return false;
}

__device__ bool cubeHit(int obj_index, int ray_index, Ray* rays, Object* objs, Hit* hits) {
	Object& o = objs[obj_index];
	Ray& ray = rays[ray_index];
	Hit& hit = hits[ray_index];
	if (hit.t < ray.maxt) {
		hits[ray_index].Mat = o.Mat;
		const Vec3 normVec = vecNormalize(vecVecAdd((vecVecAdd(ray.o, constVecMult(hit.t, ray.d))), constVecMult(-1.f, o.t.pos)));
		if (abs(normVec.x) > abs(normVec.y) && abs(normVec.x) > abs(normVec.z)) {
			if (normVec.x < 0) hits[ray_index].normG = Vec3(-1.f, 0.f, 0.f);
			else hits[ray_index].normG = Vec3(1.f, 0.f, 0.f);
		}
		else if (abs(normVec.y) > abs(normVec.x) && abs(normVec.y) > abs(normVec.z)) {
			if (normVec.y < 0) hits[ray_index].normG = Vec3(0.f, -1.f, 0.f);
			else hits[ray_index].normG = Vec3(0.f, 1.f, 0.f);
		}
		else {
			if (normVec.z < 0) hits[ray_index].normG = Vec3(0.f, 0.f, -1.f);
			else hits[ray_index].normG = Vec3(0.f, 0.f, 1.f);
		}
		hits[ray_index].normS = hit.normG;
		hit.uv = Vec2(0.f);//Not doing right now
		ray.maxt = hits[ray_index].t;
		return true;
	}
	return false;
}

__global__ void computeIntersections(
	int depth, int num_rays, Ray* rays, int objs_size, Object* objs, Hit* hits, int* hitPeaks, int* hitIndices
)
{
	int ray_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (ray_index < num_rays)
	{
		Ray& ray = rays[ray_index];

		float t_min = FLT_MAX;
		int hit_obj_index = -1;
		Hit& h = hits[ray_index];

		for (int i = 0; i < objs_size; i++)
		{
			Object obj = objs[i];
			Hit temp;

			/*if (ray_index == 200 && i == 0) {
				printf("bbox pref max x y z %f %f %f\n", obj.bbox.max.x, obj.bbox.max.y, obj.bbox.max.z);
				printf("bbox pref min x y z %f %f %f\n", obj.bbox.min.x, obj.bbox.min.y, obj.bbox.min.z);
			}*/
			
			if (bboxHit(i, ray_index, rays, objs, hits)) {
				if (obj.type == gcube) {
					if (cubeHit(i, ray_index, rays, objs, hits)) {
						ray.maxt = h.t;
						h.Mat = obj.Mat;
					}
					else {
						t_min = -1.0f;
						hit_obj_index = -1;
					}
				}
				else if (obj.type == gsphere) {
					if (sphereHit(obj, ray, h)) {
						ray.maxt = h.t;
						h.Mat = obj.Mat;
					}
					else {
						t_min = -1.0f;
						hit_obj_index = -1;
					}
				}
				else {
					t_min = -1.0f;
					hit_obj_index = -1;
				}
				
			}
			
			
			if (h.t > 0.0f && t_min > h.t)
			{
				t_min = h.t;
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

__global__ void finalGather(int num_rays, Color3* image, Ray* rays)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < num_rays)
	{
		Ray ray = rays[index];
		image[ray.pixelIndex] = Color3(ray.color);
	}
}

void pathtrace(int iter) {
	// it might make more sense to define number of ray bounces in the scene rather than the ray
	const int traceDepth = iter;
	const Camera& cam = hst_scene->cam;
	const int pixelcount = cam.resX * cam.resY;
	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resX + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resY + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;
	// printf("camx camy %d %d \n", cam.resX, cam.resY);
	generateRayFromCamera CUDA_KERNEL(blocksPerGrid2d, blockSize2d) (cam, traceDepth, dev_rays);

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

		fillIndices CUDA_KERNEL(numblocksPathSegmentTracing, blockSize1d) (num_rays, dev_hitIndices); //Fill init indices to 1...num_rays
		
		computeIntersections CUDA_KERNEL(numblocksPathSegmentTracing, blockSize1d) (
			depth
			, num_rays
			, dev_rays
			, hst_scene->sceneObjs.size()
			, dev_objs
			, dev_hits
			, dev_hitPeaks
			, dev_hitIndices
			);

		calculateColor CUDA_KERNEL(numblocksPathSegmentTracing, blockSize1d) (cam, dev_rays, dev_hits, depth, num_rays);
		
		
		cudaDeviceSynchronize();
		depth++;
		
		//Use find rays to contract rays into those that have ended and those that havent

		//num_rays = concat_rays(num_rays, numblocksPathSegmentTracing, blockSize1d, dev_hitIndices);
		
		printf("num rays: %i , depth: %i, tracedepth: %i \n", num_rays, depth, traceDepth);
		if (num_rays == 0 || depth > traceDepth) {
			iterationComplete = true; // TODO: should be based off stream compaction results.
		}
	}
	num_rays = dev_ray_end - dev_rays;
	//printf("gathered %d\n", dev_rays[2000].color);
	printf("Iteration Done\n");
	
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather CUDA_KERNEL(numBlocksPixels, blockSize1d) (num_rays, dev_image, dev_rays);

	
	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	// sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->cam.img.data(), dev_image,
		pixelcount * sizeof(Color3), cudaMemcpyDeviceToHost);

}