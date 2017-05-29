// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#include <memory>
#include "../Objects/ITMSceneParams.h"
#include "../Engine/ITMTracker.h"

namespace ITMLib
{
	namespace Objects
	{
		class ITMLibSettings
		{
		public:
			/// The device used to run the DeviceAgnostic code
			typedef enum {
				DEVICE_CPU,
				DEVICE_CUDA,
				DEVICE_METAL
			} DeviceType;

			/// Select the type of device to use
			DeviceType deviceType;

			/// Enables swapping between host and device.
			bool useSwapping;

			bool useApproximateRaycast;

			bool useBilateralFilter;

			bool modelSensorNoise;

			/// Tracker types
			typedef enum {
				//! Identifies a tracker based on colour image
				TRACKER_COLOR,
				//! Identifies a tracker based on depth image
				TRACKER_ICP,
				//! Identifies a tracker based on depth image (Ren et al, 2012)
				TRACKER_REN,
				//! Identifies a tracker based on depth image and IMU measurement
				TRACKER_IMU,
				//! Identifies a tracker that use weighted ICP only on depth image
				TRACKER_WICP,
				//! Identifies a tracker which uses a set of known poses.
				TRACKER_GROUND_TRUTH
			} TrackerType;

			/// Select the type of tracker to use
			TrackerType trackerType;

			/// The tracking regime used by the tracking controller
			/// TODO(andrei): Handle this correctly on copy!
			TrackerIterationType *trackingRegime;

			/// The number of levels in the trackingRegime
			int noHierarchyLevels;
			
			/// Run ICP till # Hierarchy level, then switch to ITMRenTracker for local refinement.
			int noICPRunTillLevel;

			/// For ITMColorTracker: skip every other point in energy function evaluation.
			bool skipPoints;

			/// For ITMDepthTracker: ICP distance threshold
			float depthTrackerICPThreshold;

			/// For ITMDepthTracker: ICP iteration termination threshold
			float depthTrackerTerminationThreshold;

			/// Further, scene specific parameters such as voxel size
			ITMLib::Objects::ITMSceneParams sceneParams;

			// For ITMGroundTruthTracker: The location of the ground truth pose
			// information folder. Currently, only the OxTS format is supported (the
			// one in which the KITTI dataset ground truth pose information is
			// provided).
			std::string groundTruthPoseFpath;

		  /// \brief The number of voxel blocks stored on the GPU.
			long sdfLocalBlockNum = kDefaultSdfLocalBlockNum;

			// Whether to create all the things required for marching cubes and mesh extraction.
			// - uses additional memory (lots!)
			bool createMeshingEngine = true;

			ITMLibSettings(void);
			~ITMLibSettings(void);

			// Suppress the default copy constructor and assignment operator
			// Re-enabled for DynSLAM, since we need to pass modified copies of the settings object
			// from the static environment reconstructor to the object instance reconstructors.
//			ITMLibSettings(const ITMLibSettings&);
//			ITMLibSettings& operator=(const ITMLibSettings&);
		};
	}
}
