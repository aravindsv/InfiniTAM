// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine_CUDA.h"
#include "ITMCUDAUtils.h"
#include "../../DeviceAgnostic/ITMSceneReconstructionEngine.h"
#include "../../../Objects/ITMRenderState_VH.h"

struct AllocationTempData {
	int noAllocatedVoxelEntries;
	int noAllocatedExcessEntries;
	int noVisibleEntries;
};

using namespace ITMLib::Engine;

template<class TVoxel, bool stopMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *localVBA, const ITMHashEntry *hashTable, int *noVisibleEntryIDs,
	const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, Vector2i imgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d, 
	Vector4f projParams_rgb, float _voxelSize, float mu, int maxW);

template<class TVoxel, bool stopMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *voxelArray, const ITMPlainVoxelArray::ITMVoxelArrayInfo *arrayInfo,
	const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, Vector2i depthImgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d, 
	Vector4f projParams_rgb, float _voxelSize, float mu, int maxW);

__global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords, const float *depth,
	Matrix4f invM_d, Vector4f projParams_d, float mu, Vector2i _imgSize, float _voxelSize, ITMHashEntry *hashTable, float viewFrustum_min,
	float viewFrustrum_max);

__global__ void allocateVoxelBlocksList_device(int *voxelAllocationList, int *excessAllocationList, ITMHashEntry *hashTable, int noTotalEntries,
	AllocationTempData *allocData, uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords);

__global__ void reAllocateSwappedOutVoxelBlocks_device(int *voxelAllocationList, ITMHashEntry *hashTable, int noTotalEntries,
	AllocationTempData *allocData, uchar *entriesVisibleType);

__global__ void setToType3(uchar *entriesVisibleType, int *visibleEntryIDs, int noVisibleEntries);

template<bool useSwapping>
__global__ void buildVisibleList_device(ITMHashEntry *hashTable, ITMHashSwapState *swapStates, int noTotalEntries,
	int *visibleEntryIDs, AllocationTempData *allocData, uchar *entriesVisibleType,
	Matrix4f M_d, Vector4f projParams_d, Vector2i depthImgSize, float voxelSize);

/// \brief Erases blocks whose weight is smaller than 'maxWeight', and marks blocks which become
///        empty in the process as pending deallocation in `outBlocksToDeallocate`.
/// \tparam TVoxel The type of voxel representation to operate on (grayscale/color, float/short, etc.)
/// \param localVBA The raw storage where the hash map entries reside.
/// \param hashTable Maps entry IDs to addresses in the local VBA.
/// \param visibleEntryIDs A list of blocks on which to operate (typically, this is the list
///                        containing the visible blocks $k$ frames ago. The size of the list should
///                        be known in advance, and be implicitly range-checked by setting the grid
///                        size's x dimension to it.
/// \param maxWeight Voxels with a *depth* weight smaller than or equal to this decay.
/// \param outBlocksToDeallocate Will contain the blocks which became completely empty as a result
///                              of the decay process.
/// \param outToDeallocateCount Will contain the number of blocks to deallocate.
template<class TVoxel>
__global__ void decay_device(TVoxel *localVBA,
							 const ITMHashEntry *hashTable,
							 int *visibleEntryIDs,
							 int maxWeight,
							 int *outBlocksToDeallocate,
							 int *outToDeallocateCount,
							 int maxBlocksToDeallocate);

/// \brief Frees the VBA blocks with the IDs contained in `blockIdsToCleanup`.
/// Based on 'cleanMemory_device' from the swapping engine.
/// Does **NOT** work with swapping enabled. (Adding support for cleanup+decay doesn't seem too
/// hard, but would require changes to the swapping mechanism in order to make it lazier.)
template<class TVoxel>
__global__ void freeBlocks_device(
		int *voxelAllocationList,
		int *lastFreeBlockId,
		ITMHashEntry *hashTable,
		int *blockIdsToCleanup,
		int blockIdsCount);


// host methods

template<class TVoxel>
ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ITMSceneReconstructionEngine_CUDA(void) 
{
	ITMSafeCall(cudaMalloc((void**)&allocationTempData_device, sizeof(AllocationTempData)));
	ITMSafeCall(cudaMallocHost((void**)&allocationTempData_host, sizeof(AllocationTempData)));

	int noTotalEntries = ITMVoxelBlockHash::noTotalEntries;
	ITMSafeCall(cudaMalloc((void**)&entriesAllocType_device, noTotalEntries));
	ITMSafeCall(cudaMalloc((void**)&blockCoords_device, noTotalEntries * sizeof(Vector4s)));

	ITMSafeCall(cudaMalloc((void**)&blocksToDeallocate_device, maxBlocksToDeallocate * sizeof(int)));
	ITMSafeCall(cudaMalloc((void**)&blocksToDeallocateCount_device, 1 * sizeof(int)));
}

template<class TVoxel>
ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::~ITMSceneReconstructionEngine_CUDA(void) 
{
	ITMSafeCall(cudaFreeHost(allocationTempData_host));
	ITMSafeCall(cudaFree(allocationTempData_device));
	ITMSafeCall(cudaFree(entriesAllocType_device));
	ITMSafeCall(cudaFree(blockCoords_device));

	ITMSafeCall(cudaFree(blocksToDeallocate_device));
	ITMSafeCall(cudaFree(blocksToDeallocateCount_device));
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	memsetKernel<TVoxel>(voxelBlocks_ptr, TVoxel(), numBlocks * blockSize);
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	fillArrayKernel<int>(vbaAllocationList_ptr, numBlocks);
	scene->localVBA.lastFreeBlockId = numBlocks - 1;

	ITMHashEntry tmpEntry;
	memset(&tmpEntry, 0, sizeof(ITMHashEntry));
	tmpEntry.ptr = -2;
	ITMHashEntry *hashEntry_ptr = scene->index.GetEntries();
	memsetKernel<ITMHashEntry>(hashEntry_ptr, tmpEntry, scene->index.noTotalEntries);
	int *excessList_ptr = scene->index.GetExcessAllocationList();
	fillArrayKernel<int>(excessList_ptr, SDF_EXCESS_LIST_SIZE);

	scene->index.SetLastFreeExcessListId(SDF_EXCESS_LIST_SIZE - 1);
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash>::AllocateSceneFromDepth(
		ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
		const ITMView *view,
		const ITMTrackingState *trackingState,
		const ITMRenderState *renderState,
		bool onlyUpdateVisibleList
) {
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, invM_d;
	Vector4f projParams_d, invProjParams_d;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;
	M_d = trackingState->pose_d->GetM(); M_d.inv(invM_d);

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	invProjParams_d = projParams_d;
	invProjParams_d.x = 1.0f / invProjParams_d.x;
	invProjParams_d.y = 1.0f / invProjParams_d.y;

	float mu = scene->sceneParams->mu;

	float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	int *voxelAllocationList = scene->localVBA.GetAllocationList();
	int *excessAllocationList = scene->index.GetExcessAllocationList();
	ITMHashEntry *hashTable = scene->index.GetEntries();
	ITMHashSwapState *swapStates = scene->useSwapping ? scene->globalCache->GetSwapStates(true) : 0;

	// The sum of the nr of buckets, plus the excess list size
	int noTotalEntries = scene->index.noTotalEntries;
	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
	uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();

	dim3 cudaBlockSizeHV(16, 16);
	dim3 gridSizeHV((int)ceil((float)depthImgSize.x / (float)cudaBlockSizeHV.x), (int)ceil((float)depthImgSize.y / (float)cudaBlockSizeHV.y));

	dim3 cudaBlockSizeAL(256, 1);
	dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));

	dim3 cudaBlockSizeVS(256, 1);
	dim3 gridSizeVS((int)ceil((float)renderState_vh->noVisibleEntries / (float)cudaBlockSizeVS.x));

	float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);

	AllocationTempData *tempData = static_cast<AllocationTempData*>(allocationTempData_host);
	tempData->noAllocatedVoxelEntries = scene->localVBA.lastFreeBlockId;
	tempData->noAllocatedExcessEntries = scene->index.GetLastFreeExcessListId();
	tempData->noVisibleEntries = 0;
	ITMSafeCall(cudaMemcpyAsync(allocationTempData_device, tempData, sizeof(AllocationTempData), cudaMemcpyHostToDevice));

	ITMSafeCall(cudaMemsetAsync(entriesAllocType_device, 0, sizeof(unsigned char)* noTotalEntries));

	if (gridSizeVS.x > 0) {
		// TODO(andrei): What does visibility type 3 mean?
		setToType3<<<gridSizeVS, cudaBlockSizeVS>>>(entriesVisibleType, visibleEntryIDs, renderState_vh->noVisibleEntries);
	}

	buildHashAllocAndVisibleType_device << <gridSizeHV, cudaBlockSizeHV >> >(entriesAllocType_device, entriesVisibleType,
		blockCoords_device, depth, invM_d, invProjParams_d, mu, depthImgSize, oneOverVoxelSize, hashTable,
		scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max);

	bool useSwapping = scene->useSwapping;
	if (onlyUpdateVisibleList) useSwapping = false;
	if (!onlyUpdateVisibleList)
	{
		allocateVoxelBlocksList_device << <gridSizeAL, cudaBlockSizeAL >> >(voxelAllocationList, excessAllocationList, hashTable,
			noTotalEntries, (AllocationTempData*)allocationTempData_device, entriesAllocType_device, entriesVisibleType,
			blockCoords_device);
	}

	if (useSwapping) {
		buildVisibleList_device<true> << <gridSizeAL, cudaBlockSizeAL >> >(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize);
	}
	else {
		buildVisibleList_device<false> << <gridSizeAL, cudaBlockSizeAL >> >(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize);
	}

	if (useSwapping)
	{
		reAllocateSwappedOutVoxelBlocks_device << <gridSizeAL, cudaBlockSizeAL >> >(voxelAllocationList, hashTable, noTotalEntries,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType);
	}

	ITMSafeCall(cudaMemcpy(tempData, allocationTempData_device, sizeof(AllocationTempData), cudaMemcpyDeviceToHost));
	renderState_vh->noVisibleEntries = tempData->noVisibleEntries;
	scene->localVBA.lastFreeBlockId = tempData->noAllocatedVoxelEntries;
	scene->index.SetLastFreeExcessListId(tempData->noAllocatedExcessEntries);

	// visibleEntryIDs is now populated with block IDs which are visible.
	int totalBlockCount = scene->index.getNumAllocatedVoxelBlocks();
	size_t visibleBlockCount = static_cast<size_t>(tempData->noVisibleEntries);

	size_t visibleEntryIDsByteCount = visibleBlockCount * sizeof(int);
	auto *visibleEntryIDsCopy = new ORUtils::MemoryBlock<int>(
			visibleEntryIDsByteCount, MEMORYDEVICE_CUDA);

	ITMSafeCall(cudaMemcpy(visibleEntryIDsCopy->GetData(MEMORYDEVICE_CUDA),
						   visibleEntryIDs,
						   visibleEntryIDsByteCount,
						   cudaMemcpyDeviceToDevice));
	VisibleBlockInfo visibleBlockInfo = {
		visibleBlockCount,
		visibleEntryIDsCopy
	};
	frameVisibleBlocks.push(visibleBlockInfo);

	// This just returns the size of the pre-allocated buffer.
	long allocatedBlocks = scene->index.getNumAllocatedVoxelBlocks();
	// This is the number of blocks we are using out of the chunk that was allocated initially on
	// the GPU (for non-swapping case).
	long usedBlocks = allocatedBlocks - scene->localVBA.lastFreeBlockId;
	long allocatedExcessBlocks = SDF_EXCESS_LIST_SIZE;
	long usedExcessBlocks = allocatedExcessBlocks - tempData->noAllocatedExcessEntries;

	if (usedBlocks > allocatedBlocks) {
		usedBlocks = allocatedBlocks;
	}
	if (usedExcessBlocks > allocatedExcessBlocks) {
		usedExcessBlocks = allocatedExcessBlocks;
	}

	// Display some memory stats, useful for debugging mapping failures.
	float percentFree = 100.0f * (1.0f - static_cast<float>(usedBlocks) / allocatedBlocks);
	printf("Visible entries: %8d | Used blocks in main GPU buffer: %8ld/%ld (%.2f%% free) | "
			"Used blocks in excess list: %8ld/%ld | Allocated size: %8d\n",
			tempData->noVisibleEntries,
			usedBlocks,
			allocatedBlocks,
			percentFree,
			usedExcessBlocks,
			allocatedExcessBlocks,
			scene->localVBA.allocatedSize);

	ITMHashEntry *entries = scene->index.GetEntries();
	if (scene->localVBA.lastFreeBlockId < 0) {
		fprintf(stderr, "ERROR: Last free block ID was negative (%d). This may indicate an "
				"allocation failure, causing your map to stop being able to grow.\n", scene->localVBA.lastFreeBlockId);
		throw std::runtime_error(
				"Invalid free voxel block ID. InfiniTAM has likely run out of GPU memory.");
	}
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash>::IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;
	if (renderState_vh->noVisibleEntries == 0) {
		// Our view has no useful data, so there's nothing to allocate. This happens, e.g., when
		// we fuse frames belonging to object instances, in which the actual instance is too far
		// away. Its depth values are over the max depth threshold (and, likely too noisy) and
		// they get ignored, leading to a blank ITMView with nothing new to integrate.
		return;
	}

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CUDA);
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();

	dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
	dim3 gridSize(renderState_vh->noVisibleEntries);

	// These kernels are launched over ALL visible blocks, whose IDs are placed conveniently as the
	// first `renderState_vh->noVisibleEntries` elements of the `visibleEntryIDs` array, which could,
	// in theory, accommodate ALL possible blocks, but usually contains O(10k) blocks.
	if (scene->sceneParams->stopIntegratingAtMaxW) {
		if (trackingState->requiresFullRendering) {
			integrateIntoScene_device<TVoxel, true, false> << < gridSize, cudaBlockSize >> > (
				localVBA, hashTable, visibleEntryIDs, rgb, rgbImgSize, depth, depthImgSize,
				M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		} else {
			integrateIntoScene_device<TVoxel, true, true> << < gridSize, cudaBlockSize >> > (
				localVBA, hashTable, visibleEntryIDs, rgb, rgbImgSize, depth, depthImgSize,
				M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		}
	}
	else {
		if (trackingState->requiresFullRendering) {
			integrateIntoScene_device<TVoxel, false, false> << < gridSize, cudaBlockSize >> > (
					localVBA, hashTable, visibleEntryIDs, rgb, rgbImgSize, depth, depthImgSize,
						M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		}
		else {
			integrateIntoScene_device<TVoxel, false, true> << < gridSize, cudaBlockSize >> > (
				localVBA, hashTable, visibleEntryIDs, rgb, rgbImgSize, depth, depthImgSize,
						M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		}
	}
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash>::Decay(
		ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
		int maxWeight,
		int minAge
) {
	// As of late June 2017, this method takes ~0.5ms at every frame, which is quite OK given that
	// we have lower-hanging fruit to speed up, such as the actual semantic segmentation.

	const bool deallocateEmptyBlocks = true;

	if (frameVisibleBlocks.size() > minAge) {
		VisibleBlockInfo visible = frameVisibleBlocks.front();
		frameVisibleBlocks.pop();

		dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
		dim3 gridSize(static_cast<uint32_t>(visible.count));

		TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
		ITMHashEntry *hashTable = scene->index.GetEntries();

		ITMSafeCall(cudaMemsetAsync(blocksToDeallocateCount_device, 0, 1 * sizeof(int)));
		decay_device<TVoxel> <<<gridSize, cudaBlockSize>>>(
				localVBA,
				hashTable,
				visible.blockIDs->GetData(MEMORYDEVICE_CUDA),
				maxWeight,
				blocksToDeallocate_device,
				blocksToDeallocateCount_device,
				maxBlocksToDeallocate);

		int freedBlockCount;
		ITMSafeCall(cudaMemcpy(&freedBlockCount,
							   blocksToDeallocateCount_device,
							   1 * sizeof(int),
							   cudaMemcpyDeviceToHost));
		// Convert from a last-element index to a count.
		freedBlockCount += 1;

		if (deallocateEmptyBlocks) {
			// Note: no explicit cleanup was done at the end of the sequences!
			// Mini-bench: 50 frames starting with 3900 in odometry sequence 08.
			// sdfLocalBlockNum: 0x80000
			// With 			ELAS no pruning: 90.50% free
			// With (w=1, a=3)  ELAS pruning:    95.35% free
			// With (w=3, a=10) ELAS pruning:    95.57% free
			// With (w=5, a=10) ELAS extreme:    96.92% free
			//
			// 75 frames now, 0x40000 (old InfiniTAM default)
			// With 			dispnet no pruning: 51.04% free
			// With (w=1, a=3)  dispnet pruning:    68.89% free
			// With (w=2, a=10) dispnet pruning:    71.40% free (visually this seems the best)
			// With (w=3, a=10) dispnet pruning:    75.63% free (a little harsh, but looks OK)
			// With (w=5, a=15) dispnet extreme:    79.22% free (extreme)

			// Print a warning in the unlikely case that we filled the `blocksToDeallocate_device`
			// buffer.
			if (freedBlockCount > maxBlocksToDeallocate) {
				fprintf(stderr, "Warning: reached maximum deallocation limit. Some blocks may have "
						"been eligible for deallocation after voxel decay, but could not be "
						"deallocated. Found %d blocks to free and the limit is %d.\n",
						freedBlockCount, maxBlocksToDeallocate);
				freedBlockCount = maxBlocksToDeallocate;
			}

			// TODO-LOW(andrei): Run capabilities check on startup, and then query caps object for
			// max number of threads per block (can't do it here without introducing brittle
			// assumptions about, e.g., GPU ID).
			dim3 freeBlockSize(1024, 1, 1);
			dim3 freeGridSize((freedBlockCount - 1) / freeBlockSize.x + 1);

			int *voxelAllocationList = scene->localVBA.GetAllocationList();
			int *excessAllocationList = scene->index.GetExcessAllocationList();

			int *lastFreeBlockId_d;
			ITMSafeCall(cudaMalloc(&lastFreeBlockId_d, 1 * sizeof(int)));
			ITMSafeCall(cudaMemcpy(lastFreeBlockId_d, &(scene->localVBA.lastFreeBlockId),
								   1 * sizeof(int),
								   cudaMemcpyHostToDevice));

			freeBlocks_device<TVoxel> <<<freeGridSize, freeBlockSize>>> (
					voxelAllocationList,
					lastFreeBlockId_d,
					hashTable,
					blocksToDeallocate_device,
					freedBlockCount);

			ITMSafeCall(cudaMemcpy(&(scene->localVBA.lastFreeBlockId), lastFreeBlockId_d,
								   1 * sizeof(int),
								   cudaMemcpyDeviceToHost));
			ITMSafeCall(cudaFree(lastFreeBlockId_d));
		}
		else {
			size_t savings = sizeof(TVoxel) * 512 * freedBlockCount;
			float savingsMb = (savings / 1024.0f / 1024.0f);

			printf("Found %d candidate blocks to deallocate with weight [%d] or below and age [%d]. "
				   "Can save %.2fMb (but deallocation is disabled).\n",
				   freedBlockCount,
				   maxWeight,
				   minAge,
				   savingsMb);
		}

		delete visible.blockIDs;
	}
}

// plain voxel array

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel,ITMPlainVoxelArray>::ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	memsetKernel<TVoxel>(voxelBlocks_ptr, TVoxel(), numBlocks * blockSize);
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	fillArrayKernel<int>(vbaAllocationList_ptr, numBlocks);
	scene->localVBA.lastFreeBlockId = numBlocks - 1;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMPlainVoxelArray>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList)
{
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMPlainVoxelArray>::IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib->trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib->intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib->intrinsics_rgb.projectionParamsSimple.all;

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CUDA);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CUDA);
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	const ITMPlainVoxelArray::ITMVoxelArrayInfo *arrayInfo = scene->index.getIndexData();

	dim3 cudaBlockSize(8, 8, 8);
	dim3 gridSize(
		scene->index.getVolumeSize().x / cudaBlockSize.x,
		scene->index.getVolumeSize().y / cudaBlockSize.y,
		scene->index.getVolumeSize().z / cudaBlockSize.z);

	if (scene->sceneParams->stopIntegratingAtMaxW) {
		if (trackingState->requiresFullRendering)
			integrateIntoScene_device < TVoxel, true, false> << <gridSize, cudaBlockSize >> >(localVBA, arrayInfo,
				rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		else
			integrateIntoScene_device < TVoxel, true, true> << <gridSize, cudaBlockSize >> >(localVBA, arrayInfo,
				rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
	}
	else
	{
		if (trackingState->requiresFullRendering)
			integrateIntoScene_device < TVoxel, false, false> << <gridSize, cudaBlockSize >> >(localVBA, arrayInfo,
				rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		else
			integrateIntoScene_device < TVoxel, false, true> << <gridSize, cudaBlockSize >> >(localVBA, arrayInfo,
				rgb, rgbImgSize, depth, depthImgSize, M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
	}
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMPlainVoxelArray>::Decay(
		ITMScene<TVoxel, ITMPlainVoxelArray> *scene,
		int maxWeight,
		int minAge
) {
  throw std::runtime_error("Map decay is not supported in conjunction with plain voxel arrays, "
						   "only with voxel block hashing.");
}


// device functions

template<class TVoxel, bool stopMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *voxelArray, const ITMPlainVoxelArray::ITMVoxelArrayInfo *arrayInfo,
	const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, Vector2i depthImgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d, 
	Vector4f projParams_rgb, float _voxelSize, float mu, int maxW)
{
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	int z = blockIdx.z*blockDim.z+threadIdx.z;

	Vector4f pt_model; int locId;

	locId = x + y * arrayInfo->size.x + z * arrayInfo->size.x * arrayInfo->size.y;
	
	if (stopMaxW) if (voxelArray[locId].w_depth == maxW) return;
//	if (approximateIntegration) if (voxelArray[locId].w_depth != 0) return;

	pt_model.x = (float)(x + arrayInfo->offset.x) * _voxelSize;
	pt_model.y = (float)(y + arrayInfo->offset.y) * _voxelSize;
	pt_model.z = (float)(z + arrayInfo->offset.z) * _voxelSize;
	pt_model.w = 1.0f;

	ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation,TVoxel>::compute(
			voxelArray[locId],
			pt_model,
			M_d,
			projParams_d,
			M_rgb,
			projParams_rgb,
			mu,
			maxW,
			depth,
			depthImgSize,
			rgb,
			rgbImgSize);
}

template<class TVoxel, bool stopMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *localVBA,
										  const ITMHashEntry *hashTable,
										  int *visibleEntryIDs,
										  const Vector4u *rgb,
										  Vector2i rgbImgSize,
										  const float *depth, Vector2i depthImgSize,
										  Matrix4f M_d, Matrix4f M_rgb,
										  Vector4f projParams_d,
										  Vector4f projParams_rgb,
										  float _voxelSize,
										  float mu,
										  int maxW)
{
	Vector3i globalPos;
	int entryId = visibleEntryIDs[blockIdx.x];

	const ITMHashEntry &currentHashEntry = hashTable[entryId];

	if (currentHashEntry.ptr < 0) return;

	globalPos = currentHashEntry.pos.toInt() * SDF_BLOCK_SIZE;

	TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3]);

	int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;

	Vector4f pt_model; int locId;

	locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	if (stopMaxW) if (localVoxelBlock[locId].w_depth == maxW) return;
	if (approximateIntegration) if (localVoxelBlock[locId].w_depth != 0) return;

	pt_model.x = (float)(globalPos.x + x) * _voxelSize;
	pt_model.y = (float)(globalPos.y + y) * _voxelSize;
	pt_model.z = (float)(globalPos.z + z) * _voxelSize;
	pt_model.w = 1.0f;

	ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation,TVoxel>::compute(localVoxelBlock[locId], pt_model, M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, depth, depthImgSize, rgb, rgbImgSize);
}

__global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords, const float *depth,
	Matrix4f invM_d, Vector4f projParams_d, float mu, Vector2i _imgSize, float _voxelSize, ITMHashEntry *hashTable, float viewFrustum_min,
	float viewFrustum_max)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > _imgSize.x - 1 || y > _imgSize.y - 1) return;

	buildHashAllocAndVisibleTypePP(entriesAllocType, entriesVisibleType, x, y, blockCoords, depth, invM_d,
		projParams_d, mu, _imgSize, _voxelSize, hashTable, viewFrustum_min, viewFrustum_max);
}

__global__ void setToType3(uchar *entriesVisibleType, int *visibleEntryIDs, int noVisibleEntries)
{
	int entryId = threadIdx.x + blockIdx.x * blockDim.x;
	if (entryId > noVisibleEntries - 1) return;
	entriesVisibleType[visibleEntryIDs[entryId]] = 3;
}

__global__ void allocateVoxelBlocksList_device(
		int *voxelAllocationList, int *excessAllocationList,
		ITMHashEntry *hashTable, int noTotalEntries,
		AllocationTempData *allocData,
		uchar *entriesAllocType, uchar *entriesVisibleType,
		Vector4s *blockCoords
) {
	int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (targetIdx > noTotalEntries - 1) return;

	int vbaIdx, exlIdx;

	switch (entriesAllocType[targetIdx])
	{
		case 0: // TODO(andrei): Could we please use constants/enums/defines for these values?
			// 0 == Invisible block.
		break;

	case 1:
		// 1 == block visible and needs allocation, fits in the ordered list.
		vbaIdx = atomicSub(&allocData->noAllocatedVoxelEntries, 1);

		if (vbaIdx >= 0) //there is room in the voxel block array
		{
			Vector4s pt_block_all = blockCoords[targetIdx];

			ITMHashEntry hashEntry;
			hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
			hashEntry.ptr = voxelAllocationList[vbaIdx];
			hashEntry.offset = 0;

			hashTable[targetIdx] = hashEntry;
		}
		else
		{
			// TODO(andrei): Handle this better.
			printf("WARNING: No more room in VBA! vbaIdx became %d.\n", vbaIdx);
			printf("exlIdx is %d.\n", allocData->noAllocatedExcessEntries);
		}
		break;

	case 2:
		// 2 == block visible and needs allocation in the excess list
		vbaIdx = atomicSub(&allocData->noAllocatedVoxelEntries, 1);
		exlIdx = atomicSub(&allocData->noAllocatedExcessEntries, 1);

		if (vbaIdx >= 0 && exlIdx >= 0) //there is room in the voxel block array and excess list
		{
			Vector4s pt_block_all = blockCoords[targetIdx];

			ITMHashEntry hashEntry;
			hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
			hashEntry.ptr = voxelAllocationList[vbaIdx];
			hashEntry.offset = 0;

			int exlOffset = excessAllocationList[exlIdx];

			hashTable[targetIdx].offset = exlOffset + 1; //connect to child

			hashTable[SDF_BUCKET_NUM + exlOffset] = hashEntry; //add child to the excess list

			entriesVisibleType[SDF_BUCKET_NUM + exlOffset] = 1; //make child visible
		}
		else
		{
			// TODO(andrei): Handle this better.
			if (vbaIdx >= 0)
			{
				printf("WARNING: Could not allocate in excess list! There was still room in the main VBA, "
						   "but exlIdx = %d! Consider increasing the overall hash table size, or at least the "
						   "bucket size.\n", exlIdx);
			}
			else if(exlIdx)
			{
				printf("WARNING: Tried to allocate in excess list, but failed because the main VBA is "
							 "full. vbaIdx = %d\n", vbaIdx);
			}
			else
			{
				printf("WARNING: No more room in VBA or in the excess list! vbaIdx became %d.\n", vbaIdx);
				printf("exlIdx is %d.\n", exlIdx);
			}
		}
		break;

		default:
			printf("Unexpected alloc type: %d\n", static_cast<int>(entriesAllocType[targetIdx]));

		break;
	}
}

__global__ void reAllocateSwappedOutVoxelBlocks_device(int *voxelAllocationList, ITMHashEntry *hashTable, int noTotalEntries,
	AllocationTempData *allocData, /*int *noAllocatedVoxelEntries,*/ uchar *entriesVisibleType)
{
	int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (targetIdx > noTotalEntries - 1) return;

	int vbaIdx;
	int hashEntry_ptr = hashTable[targetIdx].ptr;

	if (entriesVisibleType[targetIdx] > 0 && hashEntry_ptr == -1) //it is visible and has been previously allocated inside the hash, but deallocated from VBA
	{
		vbaIdx = atomicSub(&allocData->noAllocatedVoxelEntries, 1);
		if (vbaIdx >= 0) hashTable[targetIdx].ptr = voxelAllocationList[vbaIdx];
	}
}

template<bool useSwapping>
__global__ void buildVisibleList_device(
		ITMHashEntry *hashTable,
		ITMHashSwapState *swapStates,
		int noTotalEntries,
		int *visibleEntryIDs,
		AllocationTempData *allocData,
		uchar *entriesVisibleType,
		Matrix4f M_d,
		Vector4f projParams_d,
		Vector2i depthImgSize,
		float voxelSize
) {
	int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (targetIdx > noTotalEntries - 1) return;

	__shared__ bool shouldPrefix;
	shouldPrefix = false;
	__syncthreads();

	unsigned char hashVisibleType = entriesVisibleType[targetIdx];
	const ITMHashEntry & hashEntry = hashTable[targetIdx];

	if (hashVisibleType == 3)
	{
		bool isVisibleEnlarged, isVisible;

		if (useSwapping)
		{
			checkBlockVisibility<true>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
			if (!isVisibleEnlarged) hashVisibleType = 0;
		} else {
			checkBlockVisibility<false>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
			if (!isVisible) hashVisibleType = 0;
		}
		entriesVisibleType[targetIdx] = hashVisibleType;
	}

	if (hashVisibleType > 0) shouldPrefix = true;

	if (useSwapping)
	{
		if (hashVisibleType > 0 && swapStates[targetIdx].state != 2) swapStates[targetIdx].state = 1;
	}

	__syncthreads();

	// Computes the correct offsets for the visible blocks in `visibleEntryIDs`.
	if (shouldPrefix)
	{
		int offset = computePrefixSum_device<int>(hashVisibleType > 0,
												  &allocData->noVisibleEntries,
												  blockDim.x * blockDim.y,
												  threadIdx.x);
		if (offset != -1) visibleEntryIDs[offset] = targetIdx;
	}

#if 0
	// "active list": blocks that have new information from depth image
	// currently not used...
	__syncthreads();

	if (shouldPrefix)
	{
		int offset = computePrefixSum_device<int>(hashVisibleType == 1, noActiveEntries, blockDim.x * blockDim.y, threadIdx.x);
		if (offset != -1) activeEntryIDs[offset] = targetIdx;
	}
#endif
}

// TODO(andrei): Move to ITMCUDAUtils
/// \brief Naive reduction limited to a single block, useful for, e.g., quickly counting things.
/// Mutates `input`. The result will be left in `input[0]`.
template<typename T>
__device__
void blockReduce(T *input, int count, int localId) {
	if (localId >= count) {
		return;
	}

	for(unsigned int s = count / 2; s > 0; s >>= 1) {
		if (localId < s) {
			int neighborId = localId + s;
			input[localId] = input[localId] + input[neighborId];
		}

		__syncthreads();
	}
}

template<class TVoxel>
__global__
void decay_device(TVoxel *localVBA,
				  const ITMHashEntry *hashTable,
				  int *visibleEntryIDs,
				  int maxWeight,
				  int *outBlocksToDeallocate,
				  int *toDeallocateCount,
				  int maxBlocksToDeallocate
) {
	// Note: there are no range checks because we launch exactly as many threads as we need.

	int entryId = visibleEntryIDs[blockIdx.x];
	const ITMHashEntry &currentHashEntry = hashTable[entryId];
	static const int voxelsPerBlock = SDF_BLOCK_SIZE3;

	if (currentHashEntry.ptr < 0) {
		return;
	}

	// The global position of the voxel block.
	Vector3i globalPos = currentHashEntry.pos.toInt() * SDF_BLOCK_SIZE;
	TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * voxelsPerBlock]);

	int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;

	// The local offset of the voxel in the current block.
	int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	bool emptyVoxel = false;

	if (localVoxelBlock[locId].w_depth <= maxWeight && localVoxelBlock[locId].w_depth > 0) {
		localVoxelBlock[locId].reset();
		emptyVoxel = true;
	}

	if (localVoxelBlock[locId].w_depth == 0) {
		emptyVoxel = true;
	}

	// Prepare for counting the number of empty voxels in each block.
	__shared__ int countBuffer[voxelsPerBlock];
	countBuffer[locId] = static_cast<int>(emptyVoxel);
	__syncthreads();

	// Block-level sum for counting non-empty voxels in this block.
	blockReduce(countBuffer, voxelsPerBlock, locId);
	__syncthreads();

	int emptyVoxels = countBuffer[0];

	bool emptyBlock = (emptyVoxels == voxelsPerBlock);
	if (locId == 0 && emptyBlock) {
		// TODO-LOW(andrei): Use a proper scan & compact pattern for performance.
		int offset = atomicAdd(toDeallocateCount, 1);
		if (offset < maxBlocksToDeallocate) {
			outBlocksToDeallocate[offset] = entryId;
		}
	}
}

template<class TVoxel>
__global__ void freeBlocks_device(
		int *voxelAllocationList,
		int *lastFreeBlockId,
		ITMHashEntry *hashTable,
		int *blockIdsToCleanup,
		int blockIdsCount
) {
	int listIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (listIdx >= blockIdsCount) {
		return;
	}

	int entryDestId = blockIdsToCleanup[listIdx];

	// Note that we may take a small perf hit in the event that there's a collision on the block
	// we're removing. We may remove the block in the main VBA but if there's a second element in
	// that bucket residing in the excess list, we're not moving it back, even though we could.

	// `noAllocatedVoxelEntries` gets initialized with the last free block ID, and gets incremented
	// here, for every entry which was deleted.

	// TODO-LOW(andrei): Given that we're relying A LOT on this atomic, perhaps it could be worth it
	// to change this to a scan & compact type of operation. Sadly we can't "cheat" and use a single
	// thread block because usually there's 1-2k blocks per frame that need to be freed.
	int vbaIdx = atomicAdd(&lastFreeBlockId[0], 1);
	// Put the address of the newly freed block onto the free list.
	voxelAllocationList[vbaIdx] = hashTable[entryDestId].ptr;

	// `ptr == -2` means the entry has been cleared out completely.
	hashTable[entryDestId].ptr = -2;
}



template class ITMLib::Engine::ITMSceneReconstructionEngine_CUDA<ITMVoxel, ITMVoxelIndex>;

