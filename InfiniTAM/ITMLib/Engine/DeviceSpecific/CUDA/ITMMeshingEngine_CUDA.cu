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

		// This call seems to fail with any map larger than a couple of frames...
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
	int cubeIndex = buildVertList(vertList, globalPos, Vector3i(threadIdx.x, threadIdx.y, threadIdx.z), localVBA, hashTable);

	if (cubeIndex < 0) return;

  // TODO add to the CPU mesh builder as well.
	Vector3f color =
			VoxelColorReader<TVoxel::hasColorInformation, TVoxel, ITMVoxelBlockHash>::uninterpolate(
					localVBA,
					hashTable,
					globalPos + Vector3i(threadIdx.x, threadIdx.y, threadIdx.z));

	for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
	{
		int triangleId = atomicAdd(noTriangles_device, 1);

		if (triangleId < noMaxTriangles - 1)
		{
			triangles[triangleId].p0 = vertList[triangleTable[cubeIndex][i]] * factor;
			triangles[triangleId].p1 = vertList[triangleTable[cubeIndex][i + 1]] * factor;
			triangles[triangleId].p2 = vertList[triangleTable[cubeIndex][i + 2]] * factor;
          	triangles[triangleId].color = color;
		}
	}
}

template class ITMLib::Engine::ITMMeshingEngine_CUDA<ITMVoxel, ITMVoxelIndex>;
