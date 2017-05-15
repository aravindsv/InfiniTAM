

#include "Utils/BoundingBox.h"
#include "InstanceTracker.h"
#include "Track.h"

#include <iomanip>
#include <sstream>

namespace InstRecLib {
	namespace Reconstruction {
		using namespace std;
		using namespace InstRecLib::Utils;

		float Track::ScoreMatch(const TrackFrame &new_frame) const {
			// TODO(andrei): Use fine mask, not just bounding box.
			// TODO(andrei): Ensure this is modular enough to allow many different matching strategies.
			// TODO(andrei): Take time into account---if I overlap perfectly but with a very old track,
			// the score should probably be discounted.

			// We don't want to accidentally add multiple segments from the same frame to the same track.
			// This is not 100% elegant, but it works.
			if (new_frame.frame_idx == this->GetEndTime()) {
				return 0.0f;
			}

			assert(!this->frames_.empty() && "A track with no frames cannot exist.");
			const TrackFrame &latest_frame = this->frames_[ this->frames_.size() - 1];

			BoundingBox new_bbox = new_frame.instance_view.GetInstanceDetection().GetBoundingBox();
			BoundingBox last_bbox = latest_frame.instance_view.GetInstanceDetection().GetBoundingBox();
			int min_area = std::min(new_bbox.GetArea(), last_bbox.GetArea());
			int overlap_area = last_bbox.IntersectWith(new_bbox).GetArea();

			// If the overlap completely covers one of the frames, then it's considered perfect.
			// Otherwise,	frames which only partially intersect get smaller scores, and frames which don't
			// intersect at all get a score of 0.0.
			return static_cast<float>(overlap_area) / min_area;
		}

		string Track::GetAsciiArt() const {
			stringstream out;
			out << "[";
			int idx = 0;
			for(const TrackFrame& frame : frames_) {
				while(idx < frame.frame_idx) {
					out << "   ";
					++idx;
				}
				out << setw(3) << frame.frame_idx;
				++idx;
			}
			out << "]";

			return out.str();
		}
	}
}