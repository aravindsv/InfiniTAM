// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine_CUDA.h"
#include "ITMCUDAUtils.h"
#include "../../DeviceAgnostic/ITMSceneReconstructionEngine.h"
#include "../../../Objects/ITMRenderState_VH.h"
#include "../../../ITMLib.h"

struct AllocationTempData {
	int noAllocatedVoxelEntries;
	int noAllocatedExcessEntries;
	int noVisibleBlocks;
};

using namespace ITMLib::Engine;

template<class TVoxel, bool stopMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *localVBA, const ITMHashEntry *hashTable, Vector3i *visibleBlocks,
	const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, Vector2i imgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d, 
	Vector4f projParams_rgb, float _voxelSize, float mu, int maxW);

template<class TVoxel, bool stopMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *voxelArray, const ITMPlainVoxelArray::ITMVoxelArrayInfo *arrayInfo,
	const Vector4u *rgb, Vector2i rgbImgSize, const float *depth, Vector2i depthImgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d, 
	Vector4f projParams_rgb, float _voxelSize, float mu, int maxW);

__global__ void buildHashAllocAndVisibleType_device(uchar *entriesAllocType, uchar *entriesVisibleType, Vector4s *blockCoords, const float *depth,
	Matrix4f invM_d, Vector4f projParams_d, float mu, Vector2i _imgSize, float _voxelSize, ITMHashEntry *hashTable, float viewFrustum_min,
	float viewFrustrum_max, int *locks);

__global__ void allocateVoxelBlocksList_device(int *voxelAllocationList,
											   int *excessAllocationList,
											   ITMHashEntry *hashTable,
											   int noTotalEntries,
											   AllocationTempData *allocData,
											   uchar *entriesAllocType,
											   uchar *entriesVisibleType,
											   Vector4s *blockCoords,
											   int currentFrame);

__global__ void reAllocateSwappedOutVoxelBlocks_device(int *voxelAllocationList, ITMHashEntry *hashTable, int noTotalEntries,
	AllocationTempData *allocData, uchar *entriesVisibleType);

__global__ void setToType3(uchar *entriesVisibleType, Vector3i *visibleBlocks, int noVisibleBlocks,
						   ITMHashEntry *hashTable);

template<bool useSwapping>
__global__ void buildVisibleList_device(ITMHashEntry *hashTable, ITMHashSwapState *swapStates, int noTotalBlocks,
	Vector3i *visibleBlocks, AllocationTempData *allocData, uchar *entriesVisibleType,
	Matrix4f M_d, Vector4f projParams_d, Vector2i depthImgSize, float voxelSize);

// TODO(andrei): Redo documentation after you finish implementing this.
/// \brief Erases blocks whose weight is smaller than 'maxWeight', and marks blocks which become
///        empty in the process as pending deallocation in `outBlocksToDeallocate`.
/// \tparam TVoxel The type of voxel representation to operate on (grayscale/color, float/short, etc.)
/// \param localVBA The raw storage where the hash map entries reside.
/// \param hashTable Maps entry IDs to addresses in the local VBA.
/// \param visibleEntryIDs A list of blocks on which to operate (typically, this is the list
///                        containing the visible blocks $k$ frames ago. The size of the list should
///                        be known in advance, and be implicitly range-checked by setting the grid
///                        size's x dimension to it.
template<class TVoxel>
__global__ void decay_device(TVoxel *localVBA,
							 ITMHashEntry *hashTable,
							 Vector3i *visibleEntryIDs,
							 int minAge,
							 int maxWeight,
							 int *voxelAllocationList,
							 int *lastFreeBlockId,
							 int *locks,
							 int currentFrame
);

/// \brief Used to perform voxel decay on all voxels in a volume.
template<class TVoxel>
__global__ void decayFull_device(
		const Vector4s *visibleBlockGlobalPos,
		TVoxel *localVBA,
		ITMHashEntry *hashTable,
		int minAge,
		int maxWeight,
		int *voxelAllocationList,
		int *lastFreeBlockId,
		int *locks,
		int currentFrame);


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
	ITMSafeCall(cudaMalloc((void**)&lastFreeBlockId_device, 1 * sizeof(int)));
	ITMSafeCall(cudaMalloc(&locks_device, SDF_BUCKET_NUM * sizeof(int)));
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
	ITMSafeCall(cudaFree(lastFreeBlockId_device));
	ITMSafeCall(cudaFree(locks_device));
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel,ITMVoxelBlockHash>::ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	totalDecayedBlockCount = 0;
	// Clean up the visible frame queue used in voxel decay.
	while (! frameVisibleBlocks.empty()) {
		delete frameVisibleBlocks.front().blockCoords;
		frameVisibleBlocks.pop();
	}

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
	Vector3i *visibleBlocks = renderState_vh->GetVisibleBlocks();
	uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();

	dim3 cudaBlockSizeHV(16, 16);
	dim3 gridSizeHV((int)ceil((float)depthImgSize.x / (float)cudaBlockSizeHV.x), (int)ceil((float)depthImgSize.y / (float)cudaBlockSizeHV.y));

	dim3 cudaBlockSizeAL(256, 1);
	dim3 gridSizeAL((int)ceil((float)noTotalEntries / (float)cudaBlockSizeAL.x));

	dim3 cudaBlockSizeVS(256, 1);
	dim3 gridSizeVS((int)ceil((float)renderState_vh->noVisibleBlocks / (float)cudaBlockSizeVS.x));

	float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);

	AllocationTempData *tempData = static_cast<AllocationTempData*>(allocationTempData_host);
	tempData->noAllocatedVoxelEntries = scene->localVBA.lastFreeBlockId;
	tempData->noAllocatedExcessEntries = scene->index.GetLastFreeExcessListId();
	tempData->noVisibleBlocks = 0;
	ITMSafeCall(cudaMemcpyAsync(allocationTempData_device, tempData, sizeof(AllocationTempData), cudaMemcpyHostToDevice));

	ITMSafeCall(cudaMemsetAsync(entriesAllocType_device, 0, sizeof(unsigned char)* noTotalEntries));

	if (gridSizeVS.x > 0) {
		// Flags all previously visible blocks accordingly (runs for every element in the
		// visibleEntryIDs list).
		// 0 = invisible (I think)
		// 1 = visible and in memory
		// 2 = visible but swapped out
		// 3 = visible at previous frame and in memory
		//
		// Note: we could have a kernel map from visible keys to visible IDs here, or simply pass
		//       the keys to 'setToType3'.
		//
		printf("setToType3 with grid size %d and block size %d\n", gridSizeVS.x, cudaBlockSizeVS.x);
		setToType3<<<gridSizeVS, cudaBlockSizeVS>>>(
				entriesVisibleType,
				visibleBlocks,
				renderState_vh->noVisibleBlocks,
				hashTable);
	}

	// TODO(andrei): Remove if still unused and unnecessary.
	ITMSafeCall(cudaMemset( locks_device, 0, sizeof(int) * SDF_BUCKET_NUM));

	buildHashAllocAndVisibleType_device << <gridSizeHV, cudaBlockSizeHV >> >(entriesAllocType_device, entriesVisibleType,
		blockCoords_device, depth, invM_d, invProjParams_d, mu, depthImgSize, oneOverVoxelSize, hashTable,
		scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max, locks_device);

	bool useSwapping = scene->useSwapping;
	if (onlyUpdateVisibleList) {
		useSwapping = false;
	}

	if (!onlyUpdateVisibleList)
	{
		allocateVoxelBlocksList_device<<<gridSizeAL, cudaBlockSizeAL>>>(
				voxelAllocationList,
				excessAllocationList, hashTable,
				noTotalEntries,
				(AllocationTempData *) allocationTempData_device,
				entriesAllocType_device,
				entriesVisibleType,
				blockCoords_device,
				frameIdx);
	}

	if (useSwapping) {
		buildVisibleList_device<true> << <gridSizeAL, cudaBlockSizeAL >> >(
				hashTable,
				swapStates,
				noTotalEntries,
				visibleBlocks,
				(AllocationTempData *) allocationTempData_device,
				entriesVisibleType,
				M_d,
				projParams_d,
				depthImgSize,
				voxelSize);
	}
	else {
		buildVisibleList_device<false> << <gridSizeAL, cudaBlockSizeAL >> >(hashTable, swapStates, noTotalEntries, visibleBlocks,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType, M_d, projParams_d, depthImgSize, voxelSize);
	}

	if (useSwapping)
	{
		reAllocateSwappedOutVoxelBlocks_device << <gridSizeAL, cudaBlockSizeAL >> >(voxelAllocationList, hashTable, noTotalEntries,
			(AllocationTempData*)allocationTempData_device, entriesVisibleType);
	}

	ITMSafeCall(cudaMemcpy(tempData, allocationTempData_device, sizeof(AllocationTempData), cudaMemcpyDeviceToHost));
	renderState_vh->noVisibleBlocks = tempData->noVisibleBlocks;
	scene->localVBA.lastFreeBlockId = tempData->noAllocatedVoxelEntries;
	scene->index.SetLastFreeExcessListId(tempData->noAllocatedExcessEntries);

	// visibleEntryIDs is now populated with block IDs which are visible.
	int totalBlockCount = scene->index.getNumAllocatedVoxelBlocks();
	size_t visibleBlockCount = static_cast<size_t>(tempData->noVisibleBlocks);

	size_t visibleBlocksByteCount = visibleBlockCount * sizeof(Vector3i);

  	ORUtils::MemoryBlock<Vector3i> *visibleEntryIDsCopy = nullptr;
	if (visibleBlocksByteCount > 0) {
		// TODO remove debug code here
		ITMSafeCall(cudaDeviceSynchronize());
		ITMSafeCall(cudaGetLastError());

		visibleEntryIDsCopy = new ORUtils::MemoryBlock<Vector3i>(visibleBlocksByteCount, MEMORYDEVICE_CUDA);
		ITMSafeCall(cudaMemcpy(visibleEntryIDsCopy->GetData(MEMORYDEVICE_CUDA),
							   visibleBlocks,
							   visibleBlocksByteCount,
							   cudaMemcpyDeviceToDevice));
	}
	VisibleBlockInfo visibleBlockInfo = {
		visibleBlockCount,
		frameIdx,
		visibleEntryIDsCopy,
	};
	frameVisibleBlocks.push(visibleBlockInfo);
	frameIdx++;

	// This just returns the size of the pre-allocated buffer.
	long allocatedBlocks = scene->index.getNumAllocatedVoxelBlocks();
	// This is the number of blocks we are using out of the chunk that was allocated initially on
	// the GPU (for non-swapping case).
	long usedBlocks = allocatedBlocks - scene->localVBA.lastFreeBlockId;

	long allocatedExcessEntries = SDF_EXCESS_LIST_SIZE;
	long usedExcessEntries = allocatedExcessEntries - tempData->noAllocatedExcessEntries;

	if (usedBlocks > allocatedBlocks) {
		usedBlocks = allocatedBlocks;
	}
	if (usedExcessEntries > allocatedExcessEntries) {
		usedExcessEntries = allocatedExcessEntries;
	}

	// Display some memory stats, useful for debugging mapping failures.
	float percentFree = 100.0f * (1.0f - static_cast<float>(usedBlocks) / allocatedBlocks);
	float allocatedSizeMiB = scene->localVBA.allocatedSize * sizeof(ITMVoxel) / 1024.0f / 1024.0f;
	printf("[Visible: %6d | Used blocks (primary): %8ld/%ld (%.2f%% free)\n"
			" Used excess list slots: %8ld/%ld | Total allocated size: %.2fMiB]\n",
			tempData->noVisibleBlocks,
			usedBlocks,
			allocatedBlocks,
			percentFree,
			usedExcessEntries,
			allocatedExcessEntries,
			allocatedSizeMiB);

//	ITMHashEntry *entries = scene->index.GetEntries();
	if (scene->localVBA.lastFreeBlockId < 0) {
		throw std::runtime_error(
				"Invalid free voxel block ID. InfiniTAM has run out of space in the Voxel Block Array.");
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
	if (renderState_vh->noVisibleBlocks == 0) {
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

//	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
	Vector3i *visibleBlocks = renderState_vh->GetVisibleBlocks();

	dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
	dim3 gridSize(renderState_vh->noVisibleBlocks);

	// These kernels are launched over ALL visible blocks, whose IDs are placed conveniently as the
	// first `renderState_vh->noVisibleEntries` elements of the `visibleEntryIDs` array, which could,
	// in theory, accommodate ALL possible blocks, but usually contains O(10k) blocks.
	if (scene->sceneParams->stopIntegratingAtMaxW) {
		if (trackingState->requiresFullRendering) {
			integrateIntoScene_device<TVoxel, true, false> << < gridSize, cudaBlockSize >> > (
				localVBA, hashTable, visibleBlocks, rgb, rgbImgSize, depth, depthImgSize,
				M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		} else {
			integrateIntoScene_device<TVoxel, true, true> << < gridSize, cudaBlockSize >> > (
				localVBA, hashTable, visibleBlocks, rgb, rgbImgSize, depth, depthImgSize,
				M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		}
	}
	else {
		if (trackingState->requiresFullRendering) {
			// While developing dynslam, this is the version that is run.
			integrateIntoScene_device<TVoxel, false, false> << < gridSize, cudaBlockSize >> > (
					localVBA, hashTable, visibleBlocks, rgb, rgbImgSize, depth, depthImgSize,
						M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		}
		else {
			integrateIntoScene_device<TVoxel, false, true> << < gridSize, cudaBlockSize >> > (
				localVBA, hashTable, visibleBlocks, rgb, rgbImgSize, depth, depthImgSize,
						M_d, M_rgb, projParams_d, projParams_rgb, voxelSize, mu, maxW);
		}
	}
}


template<class TVoxel>
int fullDecay(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
			  int minAge,
			  int maxWeight,
			  int *lastFreeBlockId_device,
			  int *locks_device,
			  int frameIdx
) {
	fprintf(stderr, "WILL now decay ALL voxels in the map...\n");

	dim3 voxelBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
	int *voxelAllocationList = scene->localVBA.GetAllocationList();
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	// TODO(andrei): Don't malloc anything in this method.
	// First, we check every bucket and see if it's allocated, populating each index
	// in `visibleBlockGlobalPos` with the block's position, whereby every element in
	// this array corresponds to a VBA element.
	long sdfLocalBlockNum = scene->index.getNumAllocatedVoxelBlocks();
	int noTotalEntries = scene->index.noTotalEntries;
	Vector4s *visibleBlockGlobalPos_device;
	ITMSafeCall(cudaMalloc((void**)&visibleBlockGlobalPos_device, sdfLocalBlockNum * sizeof(Vector4s)));
	ITMSafeCall(cudaMemset(visibleBlockGlobalPos_device, 0, sizeof(Vector4s) * sdfLocalBlockNum));

	dim3 hashTableVisitBlockSize(256);
	dim3 hashTableVisitGridSize((noTotalEntries - 1) / hashTableVisitBlockSize.x + 1);

	fprintf(stderr, "Launching findAllocatedBlocks with gs.x = %d\n", hashTableVisitGridSize.x);
	fprintf(stderr, "total entries: %d (0x%x)\n", noTotalEntries, noTotalEntries);
	ITMLib::Engine::findAllocatedBlocks<<<hashTableVisitGridSize, hashTableVisitBlockSize>>>(
			visibleBlockGlobalPos_device, hashTable, noTotalEntries
	);
	ITMSafeCall(cudaDeviceSynchronize());
	ITMSafeCall(cudaGetLastError());
	printf("Aight, computed allocated blocks OK...\n");

	// We now know, for every block allocated in the VBA, whether it's in use, and what its
	// global coordinates are.
	dim3 gridSize(sdfLocalBlockNum);
	fprintf(stderr, "Calling decayFull_device...: gs.x = %d, minAge = %d, maxWeight = %d\n",
			gridSize.x, minAge, maxWeight);

	ITMSafeCall(cudaMemset(locks_device, 0, SDF_BUCKET_NUM * sizeof(int)));
	int oldLastFreeBlockId = scene->localVBA.lastFreeBlockId;
	decayFull_device<TVoxel> <<< gridSize, voxelBlockSize >>> (
			visibleBlockGlobalPos_device,
			localVBA,
			hashTable,
			minAge,
			maxWeight,
			voxelAllocationList,
			lastFreeBlockId_device,
			locks_device,
			frameIdx);
	ITMSafeCall(cudaDeviceSynchronize());
	ITMSafeCall(cudaGetLastError());
	printf("decayFull_device went OK\n\n");

	ITMSafeCall(cudaMemcpy(&(scene->localVBA.lastFreeBlockId), lastFreeBlockId_device,
				1 * sizeof(int),
				cudaMemcpyDeviceToHost));
	int freedBlockCount = scene->localVBA.lastFreeBlockId - oldLastFreeBlockId;

	printf("decayFull_device deleted %d blocks\n\n", freedBlockCount);

	ITMSafeCall(cudaFree(visibleBlockGlobalPos_device));
	return freedBlockCount;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash>::Decay(
		ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
		int maxWeight,
		int minAge,
		bool forceAllVoxels
) {
	// TODO(andrei): Refactor this method once the functionality is more or less complete.
//	const bool deallocateEmptyBlocks = false;	// This could be a config param of the recon engine.
	int oldLastFreeBlockId = scene->localVBA.lastFreeBlockId;

	int *voxelAllocationList = scene->localVBA.GetAllocationList();
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	ITMSafeCall(cudaMemcpy(lastFreeBlockId_device, &(scene->localVBA.lastFreeBlockId),
						   1 * sizeof(int),
						   cudaMemcpyHostToDevice));

	dim3 voxelBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);

	if (forceAllVoxels) {
		// TODO(andrei): Make this function obey the 'deallocateEmptryBlocks' flag.
		// TODO(andrei): Remove duplicate functionality for counting freed blocks.
		fullDecay<TVoxel>(scene, minAge, maxWeight, this->lastFreeBlockId_device, locks_device, frameIdx);
	}
	else if (frameVisibleBlocks.size() > minAge) {
		VisibleBlockInfo visible = frameVisibleBlocks.front();
		frameVisibleBlocks.pop();

		printf("Running decay_device on the %lu blocks visible at frame %lu.\n",
			   visible.count,
			   visible.frameIdx);

		// Ensure there are voxels to work with. We can often encounter empty frames when
		// reconstructing individual objects which are too far from the camera for any
		// meaningful depth to be estimated, so there's nothing to do for them.
		if (visible.count > 0) {
			ITMSafeCall(cudaMemset(locks_device, 0, SDF_BUCKET_NUM * sizeof(int)));

			dim3 gridSize(static_cast<uint32_t>(visible.count));
			decay_device<TVoxel> <<< gridSize, voxelBlockSize >>> (
					localVBA,
					hashTable,
					visible.blockCoords->GetData(MEMORYDEVICE_CUDA),
					minAge,
					maxWeight,
					voxelAllocationList,
					lastFreeBlockId_device,
					locks_device,
					frameIdx
			);
			ITMSafeCall(cudaDeviceSynchronize());
			ITMSafeCall(cudaGetLastError());

			// This is important for ensuring ITM "knows" about the freed up blocks in the VBA.
			ITMSafeCall(cudaMemcpy(&(scene->localVBA.lastFreeBlockId), lastFreeBlockId_device,
								   1 * sizeof(int),
								   cudaMemcpyDeviceToHost));

			delete visible.blockCoords;
		}
	}

	int freedBlockCount = scene->localVBA.lastFreeBlockId - oldLastFreeBlockId;
	totalDecayedBlockCount += freedBlockCount;

	// TODO(andrei): New benchmarks once the complete implementation is in place!
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

	if (freedBlockCount > 0) {
		size_t savings = sizeof(TVoxel) * SDF_BLOCK_SIZE3 * freedBlockCount;
		float savingsMb = (savings / 1024.0f / 1024.0f);

		printf("Found %d candidate blocks to deallocate with weight [%d] or below and age [%d]. "
			   "Saved %.2fMb.\n",
			   freedBlockCount,
			   maxWeight,
			   minAge,
			   savingsMb);
	}
	else {
		printf("Decay process found NO voxel blocks to deallocate.\n");
	}
}


template<class TVoxel>
size_t ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash>::GetDecayedBlockCount() {
	return static_cast<size_t>(totalDecayedBlockCount);
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
		ITMScene<TVoxel, ITMPlainVoxelArray>*, int, int, bool
) {
  throw std::runtime_error("Map decay is not supported in conjunction with plain voxel arrays, "
						   "only with voxel block hashing.");
}

template<class TVoxel>
size_t ITMSceneReconstructionEngine_CUDA<TVoxel, ITMPlainVoxelArray>::GetDecayedBlockCount() {
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

// Runs for every block in the visible list => Needs lookup.
template<class TVoxel, bool stopMaxW, bool approximateIntegration>
__global__ void integrateIntoScene_device(TVoxel *localVBA,
										  const ITMHashEntry *hashTable,
										  Vector3i *visibleBlocks,
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
	bool isFound = false;
	int entryId = findBlock(hashTable, visibleBlocks[blockIdx.x], isFound);

	int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
	int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	if (!isFound || entryId < 0) {
      if (locId == 0) {
			printf("FATAL ERROR in integrateIntoScene_device (isFound = %d, entryId = %d, "
				   "blockIdx.x = %d, locId = %d)! | (%d, %d, %d)\n",
				   static_cast<int>(isFound),
				   entryId,
				   blockIdx.x,
				   locId,
				   visibleBlocks[blockIdx.x].x,
				   visibleBlocks[blockIdx.x].y,
				   visibleBlocks[blockIdx.x].z
			);
	  }
		return;
	}

	const ITMHashEntry &currentHashEntry = hashTable[entryId];
	// What error message could we show here for mistakes in integration?
	if (currentHashEntry.ptr < 0) {
		return;
	}

	globalPos = currentHashEntry.pos.toInt() * SDF_BLOCK_SIZE;

	TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3]);

	Vector4f pt_model;

	if (stopMaxW) if (localVoxelBlock[locId].w_depth == maxW) return;
	if (approximateIntegration) if (localVoxelBlock[locId].w_depth != 0) return;

	pt_model.x = (float)(globalPos.x + x) * _voxelSize;
	pt_model.y = (float)(globalPos.y + y) * _voxelSize;
	pt_model.z = (float)(globalPos.z + z) * _voxelSize;
	pt_model.w = 1.0f;

	ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation,TVoxel>::compute(localVoxelBlock[locId], pt_model, M_d, projParams_d, M_rgb, projParams_rgb, mu, maxW, depth, depthImgSize, rgb, rgbImgSize);
}

__global__ void buildHashAllocAndVisibleType_device(
		uchar *entriesAllocType,
		uchar *entriesVisibleType,
		Vector4s *blockCoords,
		const float *depth,
		Matrix4f invM_d,
		Vector4f projParams_d,
		float mu,
		Vector2i _imgSize,
		float _voxelSize,
		ITMHashEntry *hashTable,
		float viewFrustum_min,
		float viewFrustum_max,
		int *locks
) {
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > _imgSize.x - 1 || y > _imgSize.y - 1) return;

	buildHashAllocAndVisibleTypePP(entriesAllocType, entriesVisibleType, x, y, blockCoords, depth, invM_d,
		projParams_d, mu, _imgSize, _voxelSize, hashTable, viewFrustum_min, viewFrustum_max, locks);
}

__global__ void setToType3(uchar *entriesVisibleType, Vector3i *visibleBlocks, int noVisibleBlocks,
						   ITMHashEntry *hashTable)
{
	int entryId = threadIdx.x + blockIdx.x * blockDim.x;
	if (entryId >= noVisibleBlocks) {
		return;
	}

	bool isFound = false;
	int hashIdx = findBlock(hashTable, visibleBlocks[entryId], isFound);

	if (! isFound || hashIdx < 0) {
		printf("FATAL ERROR in setToType3 visibleBlocks[%d]: (isFound = %d, hashIdx = %d"
					   ")! | (%d, %d, %d)\n",
               entryId,
			   static_cast<int>(isFound),
			   hashIdx,
			   visibleBlocks[blockIdx.x].x,
			   visibleBlocks[blockIdx.x].y,
			   visibleBlocks[blockIdx.x].z
		);
		return;
	}
	else {
		entriesVisibleType[hashIdx] = 3;
	}
}

__global__ void allocateVoxelBlocksList_device(
		int *voxelAllocationList, int *excessAllocationList,
		ITMHashEntry *hashTable, int noTotalEntries,
		AllocationTempData *allocData,
		uchar *entriesAllocType, uchar *entriesVisibleType,
		Vector4s *blockCoords,
		int currentFrame
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
			hashEntry.allocatedTime = currentFrame;

			// TODO(andrei): What if there are multiple blocks which think they belong in the
			// ordered list?
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
			hashEntry.allocatedTime = currentFrame;

			int exlOffset = excessAllocationList[exlIdx];

			hashTable[targetIdx].offset = exlOffset + 1; //connect to child

			hashTable[SDF_BUCKET_NUM + exlOffset] = hashEntry; //add child to the excess list

			entriesVisibleType[SDF_BUCKET_NUM + exlOffset] = 1; //make child visible
		}
		else
		{
			// TODO(andrei): Handle this better. We could probably get away with just looking at
			// noAllocatedVoxelEntries and noAllocatedExcessEntries after the kernel completes.
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

// Gets called for every entry in the hash table (ordered + excess) => no lookup.
template<bool useSwapping>
__global__ void buildVisibleList_device(
		ITMHashEntry *hashTable,
		ITMHashSwapState *swapStates,
		int noTotalEntries,
        Vector3i *visibleBlocks,
		AllocationTempData *allocData,
		uchar *entriesVisibleType,
		Matrix4f M_d,
		Vector4f projParams_d,
		Vector2i depthImgSize,
		float voxelSize
) {
	// TODO(andrei): In any case, the 'dummy' bool leads to spaghetti code. Fix that stuff, or, much
	// more likely, get rid of it because it's not helping.
  // Could this cause an error if we need this worker to compute the prefix sum even if its contribution would be 0?
	int targetIdx = threadIdx.x + blockIdx.x * blockDim.x;

	bool dummy = false;
	unsigned char hashVisibleType = 0;

	if (targetIdx > noTotalEntries - 1) {
		dummy = true;
	}

	__shared__ bool shouldPrefix;
	shouldPrefix = false;
	__syncthreads();

	if (! dummy) {
		hashVisibleType = entriesVisibleType[targetIdx];
		const ITMHashEntry & hashEntry = hashTable[targetIdx];

		// i.e., previously seen
		if (hashVisibleType == 3)
		{
			bool isVisibleEnlarged, isVisible;

			if (useSwapping)
			{
				checkBlockVisibility<true>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, 	depthImgSize);
				if (!isVisibleEnlarged) hashVisibleType = 0;
			} else {
				checkBlockVisibility<false>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, 	depthImgSize);
				if (!isVisible) hashVisibleType = 0;
			}
			entriesVisibleType[targetIdx] = hashVisibleType;
		}

		// If we've seen the block last frame, and it's still visible, keep it.
		if (hashVisibleType > 0) shouldPrefix = true;

		if (useSwapping) {
			if (hashVisibleType > 0 && swapStates[targetIdx].state != 2) swapStates[targetIdx].state = 1;
		}
    }

	__syncthreads();

	// Computes the correct offsets for the visible blocks in the output visibleBlocks array.
	if (shouldPrefix)
	{
		int offset = computePrefixSum_device<int>(hashVisibleType > 0,
												  &allocData->noVisibleBlocks,
												  blockDim.x * blockDim.y,
												  threadIdx.x);
		if (offset != -1) {
			// -1 is returned for entries which contribute a 'false' so don't need to be written to
			// the visible list.
			visibleBlocks[offset] = hashTable[targetIdx].pos.toInt();
		}
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

// TODO(andrei): Get rid of code duplication.

/// \brief Deletes a block from the hash table, deallocating its VBA entry.
/// \param hashTable
/// \param blockPos             The position of the block in the voxel grid, i.e., the key.
/// \param locks                Array used for locking in order to prevent data races when
///                             attempting to delete multiple elements with the same key.
/// \param lastFreeBlockId      Index in the voxel allocation list (free list).
/// \param voxelAllocationList  List of free voxels.
///
/// \note Does not support swapping.
template<class TVoxel>
__device__
void deleteBlock(
		ITMHashEntry *hashTable,
		Vector3i blockPos,
		int *locks,
		int *voxelAllocationList,
		int *lastFreeBlockId
) {
	int keyHash = hashIndex(blockPos);
	int status = atomicExch(&locks[keyHash], BUCKET_LOCKED);
	if (status == BUCKET_LOCKED) {
		printf("Contention on bucket of hash value %d. Not going further with deletion of block "
					   "(%d, %d, %d).\n", keyHash, blockPos.x, blockPos.y, blockPos.z);
		return;
	}

	bool isFound = false;
	int outBlockIdx = -1;
	int outPrevBlockIdx = -1;
	findVoxel(hashTable, blockPos, 0, isFound, outBlockIdx, outPrevBlockIdx);

	bool isExcess = (outBlockIdx >= SDF_BUCKET_NUM);
	// Paranoid sanity check
	if (outPrevBlockIdx == -1) {
		if (isExcess) {
			printf("\n[ERROR] Found entity in excess list with no previous element (%d, %d, %d)!\n",
				   blockPos.x,
				   blockPos.y,
				   blockPos.z);
		}
	}
	else {
		if (! isExcess) {
			printf("\n[ERROR] Found entity in bucket list with a previous guy!\n");
		}
	}

	bool hasNext = (hashTable[outBlockIdx].offset >= 1);
	if (isExcess || hasNext) {
		// Even after tons of careful debugging, there still seem to be a few Heisenbugs around
		// causing the wrong blocks to disappear when deleting an element from a bucket with more
		// than one element. However, if our hash table's size is large enough, this happens quite
		// rarely, so we ignore it for now, since we save enough memory by deleting single-block
		// buckets anyway.
		atomicExch(&locks[keyHash], BUCKET_UNLOCKED);
		return;
	}

	// Some paranoid sanity checks. TODO(andrei): Consider adding flag for toggling.
	if (!isFound || outBlockIdx < 0) {
//		if (blockPos.x % 10 == 3) {
			printf("\n\nFATAL ERROR: sanity check failed in 'decay_device' voxel (block) "
						   "found = %d, outBlockIdx = %d (%d, %d, %d) ; %s.\n",
				   static_cast<int>(isFound),
				   outBlockIdx,
				   blockPos.x,
				   blockPos.y,
				   blockPos.z,
				   isExcess ? "excess" : "non-excess"
			);
//		}
		atomicExch(&locks[keyHash], BUCKET_UNLOCKED);
		return;
	}

	if (keyHash % 100 < 40) {
		printf("Will delete block with hash idx %d / %ld (prev = %d, hashVal = %d), offset = %d; (%d, %d, %d); %s\n",
			   outBlockIdx,
			   SDF_BUCKET_NUM,
			   outPrevBlockIdx,
			   keyHash,
			   hashTable[outBlockIdx].offset,
			   blockPos.x,
			   blockPos.y,
			   blockPos.z,
			   isExcess ? "excess" : "non-excess"
		);
	}

	// First, deallocate the VBA slot.
	int freeListIdx = atomicAdd(&lastFreeBlockId[0], 1);
	voxelAllocationList[freeListIdx + 1] = hashTable[outBlockIdx].ptr;

	// TODO(andrei): Update excess freelist! (Should work without doing it but leak memory.)
	// Second, clear out the hash table entry, and do bookkeeping for buckets with more than one element.
	if (outPrevBlockIdx == -1) {
		// In the ordered list
		if (hashTable[outBlockIdx].offset >= 1) {
			// In the ordered list, with a successor.
			long nextIdx = SDF_BUCKET_NUM + hashTable[outBlockIdx].offset - 1;
			hashTable[outBlockIdx] = hashTable[nextIdx];

			// Free up the slot we just copied into the main VBA, in case there's still pointers
			// to it in the visible list from some to-be-decayed frame.
			// [RIP] Not doing this can mean the zombie block gets detected as valid in the future,
			// even though it's in the excess area but nobody is pointing at it.
			hashTable[nextIdx].offset = 0;
			hashTable[nextIdx].ptr = -2;
		}
		else {
			// In the ordered list, and no successor.
			hashTable[outBlockIdx].offset = 0;
			hashTable[outBlockIdx].ptr = -2;
		}
	}
	else {
		// In the excess list with a successor or not.
		hashTable[outPrevBlockIdx].offset = hashTable[outBlockIdx].offset;
		hashTable[outBlockIdx].offset = 0;
		hashTable[outBlockIdx].ptr = -2;
	}

	// TODO(andrei): Remove sanity check.
	// Release the lock.
	int result = atomicExch(&locks[keyHash], BUCKET_UNLOCKED);
	if (result != BUCKET_LOCKED) {
		printf("FATAL LOCK ERROR IN BLOCK DELETION\n");
	}
}


/// \brief Looks up a voxel, determines if it should be decayed, and then automatically deletes the
///        blocks that become empty in the process.
template<class TVoxel>
__device__
void decayVoxel(
		Vector3i blockGridPos,
		int locId,
		TVoxel *localVBA,
		ITMHashEntry *hashTable,
		int minAge,
		int maxWeight,
		int *voxelAllocationList,
		int *lastFreeBlockId,
		int *locks,
		int currentFrame
) {
	bool isFound = false;
	int blockHashIdx = -1;
	int blockPrevHashIdx = -1;

	int voxelIdx = findVoxel(hashTable, blockGridPos, locId, isFound, blockHashIdx, blockPrevHashIdx);

	// For debugging
//	Vector3i intPos = blockGridPos * SDF_BLOCK_SIZE3;
	int hashVal = hashIndex(blockGridPos);

	if (-1 == blockHashIdx) {
		// This happens when we, for instance, have an ID in the visible list whose target gets
		// deleted from the hash table by a previous decay phase.
		if (locId == 0) {
			printf("ERROR: could not find bucket for (%d, %d, %d) @ hash ID %d.\n",
                   blockGridPos.x, blockGridPos.y, blockGridPos.z, hashVal);
		}
		return;
	}

	if (! isFound && locId == 0 && blockIdx.x % 10 == 3) {
		printf("ERROR: voxel not found? WTF.\n");
		return;
	}

	bool emptyVoxel = false;
	bool safeToClear = true;
	int age = currentFrame - hashTable[blockHashIdx].allocatedTime;
	if (age < minAge) {
		// Important corner case: when we had a block in the visible list, but it got deleted in
		// a previous decay pass, and ended up also getting reallocated (and thus the old ID in
		// the visible list was pointing to the wrong thing).
		safeToClear = false;
	}

	if (safeToClear) {
		if (localVBA[voxelIdx].w_depth <= maxWeight && localVBA[voxelIdx].w_depth > 0) {
			localVBA[voxelIdx].reset();
			emptyVoxel = true;
		}

		if (localVBA[voxelIdx].w_depth == 0) {
			emptyVoxel = true;
		}
	}

	// Count the empty voxels in the block, to determine if it's empty

	// TODO(andrei): Try summing all the weights and empty == weightSum < k (==3-10). Niessner et
	// al. do this.
	static const int voxelsPerBlock = SDF_BLOCK_SIZE3;
	__shared__ int countBuffer[voxelsPerBlock];
	countBuffer[locId] = static_cast<int>(emptyVoxel);
	__syncthreads();

	// Block-level sum for counting non-empty voxels in this block.
	blockReduce(countBuffer, voxelsPerBlock, locId);
	__syncthreads();

	int emptyVoxels = countBuffer[0];
	bool emptyBlock = (emptyVoxels == voxelsPerBlock);

	if (locId == 0 && emptyBlock && safeToClear && blockHashIdx < SDF_BUCKET_NUM) {
		deleteBlock<TVoxel>(hashTable,
							hashTable[blockHashIdx].pos.toInt(),
							locks,
							voxelAllocationList,
							lastFreeBlockId);
	}
}

// TODO(andrei): Once your new vect3i-based indexing works, re-enable this.
// This kernel runs per-voxel, just like 'decayFull_device', for every block in the visible list
// (so we need to perform lookups).
template<class TVoxel>
__global__
void decay_device(TVoxel *localVBA,
				  ITMHashEntry *hashTable,
				  Vector3i *visibleBlocks,
				  int minAge,
				  int maxWeight,
				  int *voxelAllocationList,
				  int *lastFreeBlockId,
				  int *locks,
				  int currentFrame
) {
	// Note: there are no range checks because we launch exactly as many threads as we need.
	// The local offset of the voxel in the current block.
	int locId = threadIdx.x + threadIdx.y * SDF_BLOCK_SIZE + threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	bool isFound = false;
	int hashIdx = findBlock(hashTable, visibleBlocks[blockIdx.x], isFound);

	if (!isFound || hashIdx < 0) {
//		if(locId == 0) {
//			printf("FATAL ERROR in decay_device (isFound = %d, hashIdx = %d, "
//						   "blockIdx.x = %d, locId = %d)! | (%d, %d, %d)\n",
//				   static_cast<int>(isFound),
//				   hashIdx,
//				   blockIdx.x,
//				   locId,
//				   visibleBlocks[blockIdx.x].x,
//				   visibleBlocks[blockIdx.x].y,
//				   visibleBlocks[blockIdx.x].z
//			);
//		}
		// Not really fatal; just the visible block was likely deallocated in a previous frame.
		return;
	}

	const ITMHashEntry &currentHashEntry = hashTable[hashIdx];

	if (currentHashEntry.ptr < -1) {
      // Deallocated entry in ordered list.
//		if (locId == 0) {
//			printf("Warning, stumbled upon an invalid ordered hash entry in the visible list.\n");
//		}

		return;
	}

	Vector3i blockGridPos = currentHashEntry.pos.toInt();
	decayVoxel<TVoxel>(blockGridPos, locId, localVBA, hashTable, minAge, maxWeight,
					   voxelAllocationList, lastFreeBlockId, locks, currentFrame);
}


template<class TVoxel>
__global__
void decayFull_device(
		const Vector4s *visibleBlockGlobalPos,
		TVoxel *localVBA,
		ITMHashEntry *hashTable,
		int minAge,
		int maxWeight,
		int *voxelAllocationList,
		int *lastFreeBlockId,
		int *locks,
		int currentFrame
) {
	const int voxelBlockIdx = blockIdx.x;
	const Vector4s blockGridPos_4s = visibleBlockGlobalPos[voxelBlockIdx];

	if (blockGridPos_4s.w == 0) {
		// A zero means no hash table entry points to this block.
		return;
	}

	// Note: this also acts as the key for the voxel block hash.
	const Vector3i blockGridPos = Vector3i(blockGridPos_4s.x, blockGridPos_4s.y, blockGridPos_4s.z);
	int locId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

	decayVoxel<TVoxel>(blockGridPos, locId, localVBA, hashTable, minAge, maxWeight, voxelAllocationList,
					 lastFreeBlockId, locks, currentFrame);
}


template class ITMLib::Engine::ITMSceneReconstructionEngine_CUDA<ITMVoxel, ITMVoxelIndex>;

