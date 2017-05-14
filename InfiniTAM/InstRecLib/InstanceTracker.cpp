

#include "InstanceTracker.h"

#include <iostream>

namespace InstRecLib {
	namespace Reconstruction {

		using namespace std;
		using namespace InstRecLib::Segmentation;

		float Track::score_match(const TrackFrame &new_frame) {
			// TODO(andrei): Use fine mask, not just bounding box.
			return 0;
		}

		void InstanceTracker::ProcessChunks(int frame_idx, const DetectionVector &new_detections) {
			cout << "Frame [" << frame_idx << "]. Processing " << new_detections.size()
			     << " new detections." << endl;

			list<TrackFrame> new_track_frames;
			for(const InstanceDetection& detection : new_detections) {
				new_track_frames.emplace_back(frame_idx, detection);
			}

			// 1. Find a matching track OR a matching "stray" frame.
			this->AssignToTracks(new_track_frames);

			// 2. For leftover detections, put them into the unassigned list.
			cout << new_track_frames.size() << " new unassigned frames." << endl;
			unassigned_frames_.insert(unassigned_frames_.end(), new_track_frames.begin(), new_track_frames.end());
//			for(auto it = new_track_frames.begin(); it != new_track_frames.end(); ++it) {
//
//
//			}

			// 3. Iterate through tracks, find ``expired'' ones, and discard them.
			// TODO(andrei): Once the rest of the steps work.
		}
	}
}

