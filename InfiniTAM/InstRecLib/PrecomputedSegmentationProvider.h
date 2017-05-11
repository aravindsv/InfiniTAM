

#ifndef INFINITAM_PRECOMPUTEDSEGMENTATIONPROVIDER_H
#define INFINITAM_PRECOMPUTEDSEGMENTATIONPROVIDER_H

#include "SegmentationProvider.h"

namespace InstRecLib {

	/// \brief Reads pre-existing frame segmentations from the disk, instead of computing them on-the-fly.
	class PrecomputedSegmentationProvider : public SegmentationProvider {

	private:
		std::string segFolder_;
		int frameIdx_ = 0;
		ITMUChar4Image *lastSegPreview_;

	protected:
		void ReadInstanceInfo(const std::string& base_img_fpath);

	public:

		PrecomputedSegmentationProvider(const std::string &segFolder) : segFolder_(segFolder) {
			printf("Initializing pre-computed segmentation provider.\n");

			lastSegPreview_ = new ITMUChar4Image(true, false);
		}

		~PrecomputedSegmentationProvider() override {
			delete lastSegPreview_;
		}

		void SegmentFrame(ITMLib::Objects::ITMView *view) override;

		const ITMUChar4Image *GetSegResult() const override;
		ITMUChar4Image *GetSegResult() override;
	};
}


#endif //INFINITAM_PRECOMPUTEDSEGMENTATIONPROVIDER_H
