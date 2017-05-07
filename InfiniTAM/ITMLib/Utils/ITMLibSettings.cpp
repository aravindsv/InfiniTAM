// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#include "ITMLibSettings.h"

#include <stdio.h>

using namespace ITMLib::Objects;

ITMLibSettings::ITMLibSettings(void)
	: sceneParams(0.02f, 100, 0.005f, 0.2f, 3.0f, false)
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
	useSwapping = false;

	/// enables or disables approximate raycast
	useApproximateRaycast = false;

	/// enable or disable bilateral depth filtering;
	useBilateralFilter = false;

	// TODO(andrei): Original was just 'TRACKER_ICP'. Should be able to set
	// this using a command line flag. If we were to add support for tracking
	// using, e.g., FOVIS or libviso, or "ground truth data" (e.g., dumped by a
	// tool running libviso into a txt file in advance), I'm guessing it would
	// be an addition to this enum.
//	trackerType = TRACKER_COLOR;
//	trackerType = TRACKER_ICP;
//	trackerType = TRACKER_REN;
	//trackerType = TRACKER_IMU;
//	trackerType = TRACKER_WICP;
  trackerType = TRACKER_GROUND_TRUTH;

	/// model the sensor noise as  the weight for weighted ICP
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

  // TODO(andrei): Consider setting this via the command line.
	groundTruthPostFpath = "/home/andrei/datasets/kitti/2011_09_26"
			"/2011_09_26_drive_0019_sync/oxts/";
}

ITMLibSettings::~ITMLibSettings()
{
	delete[] trackingRegime;
}
