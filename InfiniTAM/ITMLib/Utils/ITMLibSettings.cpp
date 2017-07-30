// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMLibSettings.h"

#include <stdio.h>

using namespace ITMLib::Objects;

ITMLibSettings::ITMLibSettings(void)
//	: sceneParams(mu, maxW, voxSize, vFrust_min, vFrust_max, stopIntAtMaxW)
//  It seems that larger mu values lead to denser maps in our case, at the cost of being somewhat
// more sensitive to noise.
// maxW = max # of observations to average, per voxel, before a running average starts getting
//        computed. Decreasing this can seriously lower the memory footprint, but at the cost of quality.
//        Going down to ~25 or 10 should still be quite OK given our fast motion.
// voxSize = voxel size, in meters. Can have a HUGE impact on quality, but making it too small
//           leads to HUGE memory consumption. Moreover, making it extremely small prevents fusion
//           from occurring properly.
// Defaults:
//	: sceneParams(0.02f, 100, 0.0050f, 0.2f, 3.0f, false)
// Work well in most KITTI cases:
//	: sceneParams(0.02f, 10, 0.0035f, 0.1f, 30.0f, false)
// 0.005 also works well for a voxel size, at least for the static map. Cars can be higher-res.
// Current:
// 	Meant to work with metric reconstructions at the correct scale. This means we no longer have
// 	to hack the translation scale when processing the ground truth.
//	: sceneParams(0.3f, 10, 0.035f, 0.1f, 30.0f, false)
: sceneParams(0.30f, 10, 0.035f, 0.1f, 300.0f, false)
// Good, low resolution reconstructions => HUGE scalability.
//: sceneParams(0.90f, 10, 0.1f, 0.1f, 300.0f, false)
{
	/// depth threashold for the ICP tracker
	depthTrackerICPThreshold = 0.1f * 0.1f;

	/// For ITMDepthTracker: ICP iteration termination threshold
	depthTrackerTerminationThreshold = 1e-3f;

	/// skips every other point when using the colour tracker
	skipPoints = true;

#ifndef COMPILE_WITHOUT_CUDA
	deviceType = DEVICE_CUDA;
#else
#ifdef COMPILE_WITH_METAL
	deviceType = DEVICE_METAL;
#else
	deviceType = DEVICE_CPU;
#endif
#endif

	//deviceType = DEVICE_CPU;

	/// enables or disables swapping. HERE BE DRAGONS: It should work, but requires more testing
	/// Enabling the swapping interferes with the free-roam visualization.
	useSwapping = false;
//	useSwapping = true;

	if (useSwapping) {
		throw std::runtime_error("DynSLAM is untested with swapping emabled.");
	}

	/// enables or disables approximate raycast
	useApproximateRaycast = false;

	/// enable or disable bilateral depth filtering;
	/// When used with stereo depth maps, it seems to increase reconstruction quality.
	useBilateralFilter = false;
//	useBilateralFilter = true;

//	trackerType = TRACKER_COLOR;
//	trackerType = TRACKER_ICP;
//	trackerType = TRACKER_REN;
	//trackerType = TRACKER_IMU;
//	trackerType = TRACKER_WICP;
  trackerType = TRACKER_GROUND_TRUTH;

	/// model the sensor noise as the weight for weighted ICP
	modelSensorNoise = false;
	if (trackerType == TRACKER_WICP) modelSensorNoise = true;

	// builds the tracking regime. level 0 is full resolution
	if (trackerType == TRACKER_IMU)
	{
		noHierarchyLevels = 2;
		trackingRegime = new TrackerIterationType[noHierarchyLevels];

		trackingRegime[0] = TRACKER_ITERATION_BOTH;
		trackingRegime[1] = TRACKER_ITERATION_TRANSLATION;
	    //trackingRegime[2] = TRACKER_ITERATION_TRANSLATION;
	}
	else
	{
		noHierarchyLevels = 5;
//		noHierarchyLevels = 1;
		trackingRegime = new TrackerIterationType[noHierarchyLevels];

		trackingRegime[0] = TRACKER_ITERATION_BOTH;
		trackingRegime[1] = TRACKER_ITERATION_BOTH;
		trackingRegime[2] = TRACKER_ITERATION_ROTATION;
		trackingRegime[3] = TRACKER_ITERATION_ROTATION;
		trackingRegime[4] = TRACKER_ITERATION_ROTATION;
	}

	if (trackerType == TRACKER_REN) {
		noICPRunTillLevel = 1;
	}
	else {
		noICPRunTillLevel = 0;
	}

	if ((trackerType == TRACKER_COLOR) && (!ITMVoxel::hasColorInformation)) {
		printf("Error: Color tracker requires a voxel type with color information!\n");
	}

	// When this tracker is used, these parameters get set on engine creation.
	groundTruthPoseFpath = "";
	groundTruthPoseOffset = 0;

	sdfLocalBlockNum = 0xC0000; 		// Original: 0x40000
}

ITMLibSettings::~ITMLibSettings()
{
	delete[] trackingRegime;
}
