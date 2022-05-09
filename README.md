# 15418FinalProj
Ben Cagan and Kiran Khambhla: Path Tracing with CUDA

**PRESENTATION**
https://drive.google.com/file/d/1p8AookxAq4dRjuBfD1vTGcy_yCCRoTag/view?usp=sharing

**Summary:**

We are going to implement simple path tracing through BVH trees on the NVIDIA GPUs in the lab. Will have basic material support, with possible reflections and refractions added later, on custom scenes loaded in through files.

**Background**

In real life, objects are illuminated by rays of light from a source, with rays bouncing off their surfaces. Light sources most often do not light objects directly, rather through an indirect path, bouncing from one object to another. However, if we wanted to render every ray in a 3D image, we would quickly exceed the computer&#39;s ability, as it would have to calculate a tremendous amount of rays and their trajectories. In order to perform a more efficient render, path tracing is used, which only considers paths that are visible to the camera. This involves emitting rays from the camera (i.e. a calculation for every pixel), effectively tracing them back to the objects in the scene, and finally back to the light source or to some set depth.

![image](https://user-images.githubusercontent.com/20400307/159813510-e44f00cd-a80a-4da6-9c13-66fddf3f4747.png)

Bounding Volume Hierarchy (BVH) Trees, with respect to path tracing, are used to easily detect if rays intersect with certain objects in the image. This is done by creating a tree-like structure where objects are the leaf nodes, and are then grouped into sets based on their position in partitions of the image, also known as bounding volumes. In our trace rendering, we can travel down our BVH tree based on the trajectory of the ray, omitting objects that the ray does not intersect with based on their bounding volumes, and finally finding the object where the ray intersects at a leaf node.

![image](https://user-images.githubusercontent.com/20400307/159813634-6022464f-8cd9-4dba-8f83-65693c26baec.png)

A BVH tree of objects, organized by rectangles

We plan to implement our BVH tree traversals with CUDA, recursively performing intersection tests on each of the node&#39;s children in parallel across all pixels of the camera. If a child node is hit, we continuously test its children until we hit a leaf node. For each leaf node we hit, we store the closest intersection to keep comparing to until we know the closest final intersection. To gain more precise results, we can increase the maximum BVH tree depth, and see how that affects the time for construction and calculation. Since all pixels perform very similar operations, this lends itself to CUDA very well. As an extension, we may also try to perform the BVH construction in parallel.

**Challenges**

The challenge is just the implementation of our algorithm on CUDA, specifically, refractions may be difficult to debug, file loading and saving implementation currently unknown. Variation in the number of ray bounces in the scene may lead to load imbalance on GPUs. Refractions will pose a challenge as they are far more costly to calculate than lambertian or mirror reflection scattering, and in turn, may lead to increased divergence in threads. The solution to increasing this parallelization may be similar to divergent rays over multiple bounces in general.

**Goals/Deliverables**

75% - Bare minimum: Shoot rays out of camera in parallel, bounce through scene, do x number of recursive bounces, basic materials

100% BVH tree support

125% Basic material additions: Reflections, Refractions

If possible, real-time demo of tracing renders on certain scenes. Also, show differences in speedup on resolution, number of bounces per ray, and also scene, for different number of CUDA threads each.

**Resources**

We plan to use the GHC machines with their RTX 2080s. We mostly will start from scratch, but use the image output code from Assignments 1 + 2, and the scene file loading code from the ray tracer from 15-468 (Spring 2022) (specifically 15-468 uses json files as the format). We also will follow the practices described in [Nvidia&#39;s Thinking Parallel](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/) for BVH traversals.

**Platform Choice**

GHC 2080s - If possible also run on local GPU. C++/CUDA - Rendering image: If possible output to window via OpenGL, if not possible (due to SSH), based on image saving from Assignment 2. We need to compute a lot (many rays being traced at once) of very similar protocols in parallel, CUDA seems to be the best option.

**Schedule**

Week 1 (3/24-3/26): Build initial Camera, Ray Shooting, and Rendering routines

Rendering will build an image, and either save it, output like Assignments 1 and 2, or render as a textured quad in OpenGL

Rendering and ray shooting will do the basic randomized ray creation for each pixel, which can be done in parallel

Week 2 (3/27-4/2): Make Ray Shooting Parallel, Build Scene-Ray-Light intersection routines

Check all meshes in scene/objects

Calculate first hit

Calculate resulting bounce reflection if there are bounces left

Recursively ray trace bounce reflection

Evaluate resulting color (albedo) of hit multiplied by resulting bounce light

Add emitted light (if hitting light source)

Week 3 (4/3-4/9): Finish basic rendering integration, Begin building BVH support

Navigate BVH tree, searching in parallel if possible, in order to get first hit faster

Building BVH tree is difficult to do in parallel, probably impossible to be done efficiently in parallel on a GPU.

Week 4 (4/10-4/16): Continue BVH support building, make BVH tree parallel

Week 5 (4/17- 4/23): Debug and optimize all of the above, add file reading for scenes

Scene files may use code from either Scotty3D or 15-468&#39;s Dirt if I am allowed to, if not, we will rig up a basic file format that uses pre-built primitives to focus on building the ray tracer itself.

Week 6 (4/24- 4/30): Debugging and optimizing, add material (reflection and refraction support)

Two basic materials, refraction for transparency (dielectric material), and a reflection (perfect reflected bounce light around normal) which has a particular albedo (reflectivity) which scales how different wavelengths (R,G,B) reflect.

Week 7 (5/1-5/5): Debugging and optimization, create example scenes

**Milestone Report** 

So far, we have created the definitions and almost complete sequential implementations for our Camera, Scene, Rays, Objects and their Materials, and Bounding Box Implementation. This includes basic OpenGL rendering, casting rays from the camera, and ray-object intersection tests.
We still haven’t gotten to our CUDA implementation  for the camera to emit rays and BVH tree creation, simply due to the fact that our sequential implementations took longer than expected, but we do have some cushion towards the end to make up for falling behind.  
We plan to show both demos of our actual render output as well as graphs of perceived speedup under different CUDA parameters. The rest of the work is pretty clear, it’s just a matter of coding and doing the work.

**Updated Schedule**

Week 4 (4/10-4/16): Make Ray Shooting Parallel, Build Scene-Ray-Light intersection routines
	Check all meshes in scene/objects
	
Week 5 (4/17- 4/23): Finish basic rendering integration, Begin building BVH support
	Navigate BVH tree, searching in parallel if possible, in order to get first hit faster add file reading for scenes
	
Week 6 (4/24- 4/30): Debugging and optimizing, add material (reflection and refraction support)
	Two basic materials, refraction for transparency (dielectric material), and a reflection (perfect reflected bounce light around normal) which has a particular albedo (reflectivity) which scales how different wavelengths (R,G,B) reflect.
	
Week 7 (5/1-5/5): Debugging and optimization, create example scenes
