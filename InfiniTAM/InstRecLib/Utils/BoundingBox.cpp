
#include "BoundingBox.h"

namespace InstRecLib {
	namespace Utils {
		std::ostream& operator<<(std::ostream& out, const BoundingBox& bounding_box) {
			out << "[ (" << bounding_box.r.x0 << ", " << bounding_box.r.y0 << "), "
				           << bounding_box.r.x1 << ", " << bounding_box.r.y1 << ") ]";
			return out;
		}
	}
}

