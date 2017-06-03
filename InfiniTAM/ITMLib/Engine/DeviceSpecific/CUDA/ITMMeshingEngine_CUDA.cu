// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include <iostream>
#include "ITMMeshingEngine_CUDA.h"
#include "../../DeviceAgnostic/ITMMeshingEngine.h"
#include "ITMCUDAUtils.h"

#include "../../../../ORUtils/CUDADefines.h"

template<class TVoxel>
__global__ void meshScene_device(ITMMesh::Triangle *triangles, unsigned int *noTriangles_device, float factor, int noTotalEntries,
	int noMaxTriangles, const Vector4s *visibleBlockGlobalPos, const TVoxel *localVBA, const ITMHashEntry *hashTable);

__global__ void findAllocateBlocks(Vector4s *visibleBlockGlobalPos, const ITMHashEntry *hashTable, int noTotalEntries);

using namespace ITMLib::Engine;

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ITMMeshingEngine_CUDA(long sdfLocalBlockNum)
	: sdfLocalBlockNum(sdfLocalBlockNum)
{
	ITMSafeCall(cudaMalloc((void**)&visibleBlockGlobalPos_device, sdfLocalBlockNum * sizeof(Vector4s)));
	ITMSafeCall(cudaMalloc((void**)&noTriangles_device, sizeof(unsigned int)));
}

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel,ITMVoxelBlockHash>::~ITMMeshingEngine_CUDA(void)
{
	ITMSafeCall(cudaFree(visibleBlockGlobalPos_device));
	ITMSafeCall(cudaFree(noTriangles_device));
}

/// \brief Hacky operator for easily displaying CUDA dim3 objects.
std::ostream& operator<<(std::ostream &out, dim3 dim) {
	out << "[" << dim.x << ", " << dim.y << ", " << dim.z << "]";
	return out;
}

template<class TVoxel>
void ITMMeshingEngine_CUDA<TVoxel, ITMVoxelBlockHash>::MeshScene(ITMMesh *mesh, const ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	// TODO(andrei): This doesn't seem to work well if swapping is enabled. That is, it only saves
	// the active mesh, and doesn't attempt to somehow stream all the blocks which have been swapped
	// out to RAM.

	ITMMesh::Triangle *triangles = mesh->triangles->GetData(MEMORYDEVICE_CUDA);
	const TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	const ITMHashEntry *hashTable = scene->index.GetEntries();

	int noMaxTriangles = mesh->noMaxTriangles, noTotalEntries = scene->index.noTotalEntries;
	float factor = scene->sceneParams->voxelSize;

	ITMSafeCall(cudaMemset(noTriangles_device, 0, sizeof(unsigned int)));
	ITMSafeCall(cudaMemset(visibleBlockGlobalPos_device, 0, sizeof(Vector4s) * sdfLocalBlockNum));

	{ // identify used voxel blocks
		dim3 cudaBlockSize(256); 
		dim3 gridSize((int)ceil((float)noTotalEntries / (float)cudaBlockSize.x));

		std::cout << "Sanity check for existing errors..." << std::endl;
		ITMSafeCall(cudaGetLastError());
		std::cout << "Sanity check passed..." << std::endl;

      	std::cout << "Launching block allocation kernel..." << std::endl;
		std::cout << "Using cuda grid size: " << gridSize << std::endl;
		std::cout << "Using cuda block size: " << cudaBlockSize << std::endl;

		findAllocateBlocks << <gridSize, cudaBlockSize >> >(visibleBlockGlobalPos_device, hashTable, noTotalEntries);
		ITMSafeCall(cudaGetLastError());
	}

	{ // mesh used voxel blocks
		dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
		dim3 gridSize(sdfLocalBlockNum / 16, 16);

		std::cout << "Checking for existing errors (would be bad if we got here with a leftover "
				  << "error from somewhere else)..." << std::endl << std::flush;
		ITMSafeCall(cudaGetLastError());
      	std::cout << "No existing error. Launching kernel..." << std::endl;

		std::cout << "Using cuda grid size: " << gridSize << std::endl;
		std::cout << "Using cuda block size: " << cudaBlockSize << std::endl;

		meshScene_device<TVoxel> << <gridSize, cudaBlockSize >> >(triangles, noTriangles_device, factor, noTotalEntries, noMaxTriangles,
			visibleBlockGlobalPos_device, localVBA, hashTable);
		ITMSafeCall(cudaGetLastError());

		ITMSafeCall(cudaMemcpy(&mesh->noTotalTriangles, noTriangles_device, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		printf("%d/%d triangles in mesh.\n", mesh->noTotalTriangles, mesh->noMaxTriangles);
	}
}

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel,ITMPlainVoxelArray>::ITMMeshingEngine_CUDA(long sdfLocalBlockNum)
		: sdfLocalBlockNum(sdfLocalBlockNum)
{}

template<class TVoxel>
ITMMeshingEngine_CUDA<TVoxel,ITMPlainVoxelArray>::~ITMMeshingEngine_CUDA(void) 
{}

template<class TVoxel>
void ITMMeshingEngine_CUDA<TVoxel, ITMPlainVoxelArray>::MeshScene(ITMMesh *mesh, const ITMScene<TVoxel, ITMPlainVoxelArray> *scene)
{}

__global__ void findAllocateBlocks(Vector4s *visibleBlockGlobalPos, const ITMHashEntry *hashTable, int noTotalEntries)
{
	int entryId = threadIdx.x + blockIdx.x * blockDim.x;
	if (entryId > noTotalEntries - 1) return;

	const ITMHashEntry &currentHashEntry = hashTable[entryId];

	if (currentHashEntry.ptr >= 0) 
		visibleBlockGlobalPos[currentHashEntry.ptr] = Vector4s(currentHashEntry.pos.x, currentHashEntry.pos.y, currentHashEntry.pos.z, 1);
}

template<class TVoxel>
__global__ void meshScene_device(ITMMesh::Triangle *triangles, unsigned int *noTriangles_device, float factor, int noTotalEntries, 
	int noMaxTriangles, const Vector4s *visibleBlockGlobalPos, const TVoxel *localVBA, const ITMHashEntry *hashTable)
{
	const Vector4s globalPos_4s = visibleBlockGlobalPos[blockIdx.x + gridDim.x * blockIdx.y];

	if (globalPos_4s.w == 0) return;

	Vector3i globalPos = Vector3i(globalPos_4s.x, globalPos_4s.y, globalPos_4s.z) * SDF_BLOCK_SIZE;

	Vector3f vertList[12];

	Vector3i localPos = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);
	int cubeIndex = buildVertList(vertList, globalPos, localPos, localVBA, hashTable);

	if (cubeIndex < 0) return;

//	Vector4f color =
//			VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::interpolate(
//					localVBA,
//					hashTable,
////					globalPos + Vector3i(threadIdx.x, threadIdx.y, threadIdx.z));
//                    Vector3f(threadIdx.x + globalPos.x, threadIdx.y + globalPos.y, threadIdx.z + globalPos.z));

	// TODO(andrei): See if you can implement this correctly.
	// In marching cubes, a corresponding poly is fit to each group of 8 voxels (cube) in the volume.
	// It would therefore make sense to use the vertex outputs generated by marching cubes to do
	// the color lookup, right? Then we could do proper stuff like proper per-vertex coloring and whatnot.
//	Vector3f color =
//			VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::uninterpolate(
//					localVBA,
//					hashTable,
//					globalPos + Vector3i(threadIdx.x, threadIdx.y, threadIdx.z));

	for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
	{
		int triangleId = atomicAdd(noTriangles_device, 1);

		if (triangleId < noMaxTriangles - 1)
		{
			Vector3f p0 = vertList[triangleTable[cubeIndex][i]];
			Vector3f p1 = vertList[triangleTable[cubeIndex][i + 1]];
			Vector3f p2 = vertList[triangleTable[cubeIndex][i + 2]];
			triangles[triangleId].p0 = p0 * factor;
			triangles[triangleId].p1 = p1 * factor;
			triangles[triangleId].p2 = p2 * factor;

          	// TODO(andrei): Reduce code duplication...
//			Vector4f c0 =
//					VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::interpolate(
//							localVBA,
//							hashTable,
//                            Vector3f(p0.x, p0.y, p0.z));
//			Vector4f c1 =
//					VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::interpolate(
//							localVBA,
//							hashTable,
//							Vector3f(p1.x, p1.y, p1.z));
//			Vector4f c2 =
//					VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::interpolate(
//							localVBA,
//							hashTable,
//							Vector3f(p2.x, p2.y, p2.z));

//          	triangles[triangleId].c0.r = c0.r;
//			triangles[triangleId].c0.g = c0.g;
//			triangles[triangleId].c0.b = c0.b;
//			triangles[triangleId].c1.r = c1.r;
//			triangles[triangleId].c1.g = c1.g;
//			triangles[triangleId].c1.b = c1.b;
//			triangles[triangleId].c2.r = c2.r;
//			triangles[triangleId].c2.g = c2.g;
//			triangles[triangleId].c2.b = c2.b;

			Vector3f c0 =
					VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::uninterpolate(
							localVBA,
							hashTable,
							Vector3i(p0.x, p0.y, p0.z));
			Vector3f c1 =
					VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::uninterpolate(
							localVBA,
							hashTable,
							Vector3i(p1.x, p1.y, p1.z));
			Vector3f c2 =
					VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::uninterpolate(
							localVBA,
							hashTable,
							Vector3i(p2.x, p2.y, p2.z));

            triangles[triangleId].c0 = c0;
			triangles[triangleId].c1 = c1;
			triangles[triangleId].c2 = c2;
		}
	}
}

template class ITMLib::Engine::ITMMeshingEngine_CUDA<ITMVoxel, ITMVoxelIndex>;
