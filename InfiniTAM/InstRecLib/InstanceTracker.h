

#ifndef INFINITAM_INSTANCETRACKER_H
#define INFINITAM_INSTANCETRACKER_H


#include <list>
#include <vector>
#include <cassert>

#include "InstanceSegmentationResult.h"

// TODO(andrei): Consider splitting this file up into multiple class files.
namespace InstRecLib {
	namespace Reconstruction {

		const float kTrackScoreThreshold = 0.25f;

		/// \brief One frame of an instance track.
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

		class Track {
		private:
			std::vector<TrackFrame> frames;

		public:
			/// \brief Evaluates how well this new frame would fit the existing track.
			/// \returns A goodness score between 0 and 1, where 0 means the new frame would not match
			/// this track at all, and 1 would be a perfect match.
			float ScoreMatch(const TrackFrame& new_frame) const;

			void AddFrame(const TrackFrame& new_frame) {
				frames.push_back(new_frame);
			}

			size_t GetSize() const {
				return frames.size();
			}

			TrackFrame& GetLastFrame() {
				return frames.back();
			}

			const TrackFrame& GetLastFrame() const {
				return frames.back();
			}
		};

		using DetectionVector = std::vector<InstRecLib::Segmentation::InstanceDetection>;
		using DetectionList = std::list<InstRecLib::Segmentation::InstanceDetection>;

		/// \brief Tracks instances over time by associating multiple isolated detections.
		// TODO(andrei): Once complete, refactor this into an interface + MaxOverlapTracker.
		class InstanceTracker {
		private:
			/// \brief The tracks of the objects being currently tracked.
			std::vector<Track> active_tracks_;

			/// \brief Frames which are currently not assigned to any track.
			// Not used, since right now we just create 1-frame tracks for unmatched frames.
			// TODO(andrei): Remove if not needed.
//			std::list<TrackFrame> unassigned_frames_;

			/// \brief The maximum age of the latest frame in a track, before it is discarded.
			/// The higher this is, the more tracks are held in memory.
			int inactive_frame_threshold_;

		protected:
			static constexpr std::pair<Track*, float> kNoBestTrack = std::pair<Track*, float>(nullptr, 0.0f);

			/// \brief Finds the most likely track for the given frame, if it exists.
			/// \return The best matching track, and the [0-1] match quality score. If no tracks are
			/// available, kNoTrack is returned.
			std::pair<Track*, float> FindBestTrack(const TrackFrame& track_frame) {
				if (active_tracks_.empty()) {
					return kNoBestTrack;
				}

				float best_score = -1.0f;
				Track *best_track = nullptr;

				for (Track& track : active_tracks_) {
					float score = track.ScoreMatch(track_frame);
					if (score > best_score) {
						best_score = score;
						best_track = &track;
					}
				}

				assert(best_score >= 0.0f);
				assert(best_track != nullptr);
				return std::pair<Track*, float>(best_track, best_score);
			};

			/// \brief Assign the detections to the best matching tracks.
			/// \note Mutates the `new_detections` input list, removing the matched detections.
			void AssignToTracks(std::list<TrackFrame>& new_detections) {
				using namespace std;
				for(auto it = new_detections.begin(); it != new_detections.end(); ++it) {
					pair<Track*, float> match = FindBestTrack(*it);
					Track* track = match.first;
					float score = match.second;

					if (score > kTrackScoreThreshold) {
						cout << "Found a match based on overlap with score " << score << "." << endl;
						cout << "Adding it to track of length " << track->GetSize() << "." << endl;
						track->AddFrame(*it);
						it = new_detections.erase(it);
					}
					else {
						cout << "Best score was: " << score << ", below the threshold. Will create new track." << endl;
					}
				}
			};

			/// \brief Removes tracks which have not been active in the past k frames.
			void PruneTracks(int current_frame_idx);

		public:
			/// \brief Associates the new detections with existing tracks, or creates new ones.
			/// \param new_detections The instances detected in the current frame).
			void ProcessChunks(int frame_idx, const DetectionVector& new_detections);

			InstanceTracker() : active_tracks_(std::vector<Track>()), inactive_frame_threshold_(5) {
				std::cout << "Initialized stuff in the Instance Tracker!" << std::endl;
			}
		};

	}
}



#endif //INFINITAM_INSTANCETRACKER_H
