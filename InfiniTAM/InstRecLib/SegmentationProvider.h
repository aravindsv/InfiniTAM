

#ifndef INFINITAM_SEGMENTATIONPROVIDER_H
#define INFINITAM_SEGMENTATIONPROVIDER_H

#include "../ITMLib/Objects/ITMView.h"

namespace InstRecLib {

	/// \brief Performs semantic segmentation on input frames.
	class SegmentationProvider {
	private:

	public:
		SegmentationProvider() { };
		virtual ~SegmentationProvider() { };

		virtual void SegmentFrame(ITMLib::Objects::ITMView *view) = 0;
	};

}


#endif //INFINITAM_SEGMENTATIONPROVIDER_H
