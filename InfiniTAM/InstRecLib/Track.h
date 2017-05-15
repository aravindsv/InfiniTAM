

#ifndef INFINITAM_TRACK_H
#define INFINITAM_TRACK_H


#include <vector>
#include "InstanceSegmentationResult.h"

namespace InstRecLib {
	namespace Reconstruction {

		/// \brief One frame of an instance track (InstRecLib::Reconstruction::Track).
		struct TrackFrame {
			int frame_idx;
			InstRecLib::Segmentation::InstanceDetection detection;

			// Only active if inside a Track. TODO(andrei): Implement.
			TrackFrame* previous;
			TrackFrame* next;

		public:
			TrackFrame(int frame_idx, const InstRecLib::Segmentation::InstanceDetection &detection)
					: frame_idx(frame_idx), detection(detection), previous(nullptr), next(nullptr) {}
		};

		/// \brief A detected object's track through multiple frames.
		/// Modeled as a series of detections, contained in the 'frames' field. Note that there can
		/// be gaps in this list, due to frames where this particular object was not detected.
		class Track {
		public:
			Track() {}

			virtual ~Track() {}

			/// \brief Evaluates how well this new frame would fit the existing track.
			/// \returns A goodness score between 0 and 1, where 0 means the new frame would not match
			/// this track at all, and 1 would be a perfect match.
			float ScoreMatch(const TrackFrame &new_frame) const;

			void AddFrame(const TrackFrame &new_frame) {
				frames_.push_back(new_frame);
			}

			size_t GetSize() const {
				return frames_.size();
			}

			TrackFrame &GetLastFrame() {
				return frames_.back();
			}

			const TrackFrame &GetLastFrame() const {
				return frames_.back();
			}

			int GetStartTime() const {
				return frames_.front().frame_idx;
			}

			int GetEndTime() const {
				return frames_.back().frame_idx;
			}

			/// \brief Draws a visual representation of this feature track.
			std::string GetAsciiArt() const;

		private:
			std::vector<TrackFrame> frames_;
		};
	}
}


#endif //INFINITAM_TRACK_H
