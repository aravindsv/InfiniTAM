

#include "PrecomputedSegmentationProvider.h"
#include "../Utils/FileUtils.h"

#include <cstdlib>
#include <sstream>

namespace InstRecLib {

	using namespace std;

	void PrecomputedSegmentationProvider::SegmentFrame(ITMLib::Objects::ITMView *view) {
		printf("Will pretend to segment frame %d now...\n", this->frameIdx);

		ITMUChar4Image *img = new ITMUChar4Image(true, false);

		stringstream ss;
		ss << this->segFolder << "/" << "final_" << setfill('0') << setw(6) << this->frameIdx << ".png";
		ReadImageFromFile(img, ss.str().c_str());

		this->frameIdx++;
		delete img;   // TODO(andrei): Reuse buffer, of course.
	}
}
