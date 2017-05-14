

#include "InstanceTracker.h"

#include <cassert>
#include <iostream>

namespace InstRecLib {
	namespace Reconstruction {

		using namespace std;
		using namespace InstRecLib::Segmentation;

		float Track::ScoreMatch(const TrackFrame &new_frame) const {
			// TODO(andrei): Use fine mask, not just bounding box.
			// TODO(andrei): Ensure this is modular enough to allow many different matching strategies.
			// TODO(andrei): Take time into account---if I overlap perfectly with a very old track, the
			// score should probably be discounted.

			assert(!this->frames.empty() && "A track with no frames cannot exist.");
			const TrackFrame& latest_frame = this->frames[this->frames.size() - 1];

			int min_area = min(new_frame.detection.GetBoundingBox().GetArea(),
			                   latest_frame.detection.GetBoundingBox().GetArea());
			int overlap_area = latest_frame.detection.GetBoundingBox().IntersectWith(new_frame.detection.GetBoundingBox()).GetArea();

			// If the overlap completely covers one of the frames, then it's considered perfect.
			// Otherwise,	frames which only partially intersect get smaller scores, and frames which don't
			// intersect at all get a score of 0.0.
			return static_cast<float>(overlap_area) / min_area;
		}

		void InstanceTracker::ProcessChunks(int frame_idx, const DetectionVector &new_detections) {
			// TODO(andrei): also accept actual DATA!!
			cout << "Frame [" << frame_idx << "]. Processing " << new_detections.size()
			     << " new detections." << endl;

			list<TrackFrame> new_track_frames;
			for(const InstanceDetection& detection : new_detections) {
				new_track_frames.emplace_back(frame_idx, detection);
			}

			// 1. Find a matching track.
			this->AssignToTracks(new_track_frames);

			// 2. For leftover detections, put them into new, single-frame, trackes.
			cout << new_track_frames.size() << " new unassigned frames." << endl;
			for (const TrackFrame &track_frame : new_track_frames) {
				Track new_track;
				new_track.AddFrame(track_frame);
				this->active_tracks_.push_back(new_track);
			}

			cout << "We now have " << this->active_tracks_.size() << " active tracks." << endl;

			// 3. Iterate through tracks, find ``expired'' ones, and discard them.
			this->PruneTracks(frame_idx);
		}

		void InstanceTracker::PruneTracks(int current_frame_idx) {
			for(auto it = active_tracks_.begin(); it != active_tracks_.end(); ++it) {
				int last_active = it->GetLastFrame().frame_idx;
				int frame_delta = current_frame_idx - last_active;

				if (frame_delta > inactive_frame_threshold_) {
					cout << "Should trim old track..." << endl;
					// TODO(andrei): Actually trim it.
				}

			}
		}
	}
}

