

#ifndef INFINITAM_PRECOMPUTEDSEGMENTATIONPROVIDER_H
#define INFINITAM_PRECOMPUTEDSEGMENTATIONPROVIDER_H

#include "SegmentationProvider.h"

namespace InstRecLib {

	/// \brief Reads pre-existing frame segmentations from the disk, instead of computing them on-the-fly.
	class PrecomputedSegmentationProvider : public SegmentationProvider {

	private:
		std::string segFolder;
		int frameIdx = 0;

	public:

		PrecomputedSegmentationProvider() : SegmentationProvider() {
			printf("Initializing pre-computed segmentation provider.\n");

			segFolder = "/home/andrei/datasets/kitti/odometry-dataset/sequences/06/seg_image_02/mnc";
		}

		void SegmentFrame(ITMLib::Objects::ITMView *view) override;


	};
}


#endif //INFINITAM_PRECOMPUTEDSEGMENTATIONPROVIDER_H
