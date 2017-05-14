
#include "InstanceSegmentationResult.h"

#include <iomanip>

namespace InstRecLib {
	namespace Segmentation {
		using namespace std;

		string InstanceDetection::getClassName() const {
			return parent_result->segmentation_dataset.labels[class_id];
		}

		ostream& operator<<(ostream& out, const InstanceDetection& detection) {
			out << detection.getClassName() << " at " << detection.GetBoundingBox() << ". "
			    << "Probability: " << setprecision(4) << setw(6) << detection.class_probability << ".";
			return out;
		}
	}
}

