
#include "InstanceSegmentationResult.h"

#include <iomanip>

namespace InstRecLib {
	namespace Segmentation {
		using namespace std;

		string InstanceDetection::getClassName() const {
			return parent_result->segmentation_dataset.labels[class_id];
		}

		ostream& operator<<(ostream& out, const InstanceDetection& detection) {
			out << detection.getClassName() << " at [" << detection.bounding_box[0] << ", "
			    << detection.bounding_box[1] << ", " << detection.bounding_box[2] << ", "
			    << detection.bounding_box[3] << "]. Probability: "
			    << setprecision(4) << setw(6) << detection.class_probability << ".";
			return out;
		}
	}
}

