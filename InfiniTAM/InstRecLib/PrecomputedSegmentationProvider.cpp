

#include "PrecomputedSegmentationProvider.h"
#include "../Utils/FileUtils.h"

#include <cstdlib>
#include <sstream>

namespace InstRecLib {

	using namespace std;

	void PrecomputedSegmentationProvider::SegmentFrame(ITMLib::Objects::ITMView *view) {
		stringstream ss;
		ss << this->segFolder_ << "/" << "cls_" << setfill('0') << setw(6) << this->frameIdx_ << ".png";
		ReadImageFromFile(lastSegPreview_, ss.str().c_str());

		this->frameIdx_++;
	}

	const ORUtils::Image<Vector4u> *PrecomputedSegmentationProvider::GetSegResult() const {
		return this->lastSegPreview_;
	}

	ORUtils::Image<Vector4u> *PrecomputedSegmentationProvider::GetSegResult() {
		return this->lastSegPreview_;
	}
}
