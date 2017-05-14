

#include "InstanceReconstructor.h"

#include <vector>

namespace InstRecLib {
	namespace Reconstruction {
		using namespace std;
		using namespace InstRecLib::Segmentation;
		using namespace InstRecLib::Utils;
		using namespace ITMLib::Objects;

		// TODO(andrei): Implement this in CUDA. It should be easy.
		void processSihlouette_CPU(
				Vector4u* sourceRGB,
				float* sourceDepth,
				Vector4u* destRGB,
				float* destDepth,
				Vector2i sourceDims,
				const InstanceDetection& detection
		) {
			// Blanks out the detection's silhouette in the 'source' frames, and writes its pixels into
			// the output frames.
			// Initially, the dest frames will be the same size as the source ones, but this is wasteful
			// in terms of memory: we should use bbox+1-sized buffers in the future, since most
			// silhouettes are relatively small wrt the size of the whole frame.
			//
			// Moreover, we should be able to pass in several output buffer addresses and a list of
			// detections to the CUDA kernel, and do all the ``splitting up'' work in one kernel call. We
			// may need to add support for the adaptive-size output buffers, since otherwise writing to
			// e.g., 5-6 output buffers may end up using up way too much GPU memory.

			int frame_width = sourceDims[0];
			int frame_height = sourceDims[1];
			const BoundingBox& bbox = detection.GetBoundingBox();

			int box_width = bbox.GetWidth();
			int box_height = bbox.GetHeight();

			memset(destRGB, 0, frame_width * frame_height * sizeof(Vector4u));
			memset(destDepth, 0, frame_width * frame_height * sizeof(float));

			for(int row = 0; row < box_height; ++row) {
				for(int col = 0; col < box_width; ++col) {
					int frame_row = row + bbox.r.y0;
					int frame_col = col + bbox.r.x0;
					// TODO(andrei): Are the CPU-specific InfiniTAM functions doing this in a nicer way?

					int frame_idx = frame_row * frame_width + frame_col;

					int mask = detection.mask->GetMask()[row][col];
					if (mask == 1) {
						destRGB[frame_idx].r = sourceRGB[frame_idx].r;
						destRGB[frame_idx].g = sourceRGB[frame_idx].g;
						destRGB[frame_idx].b = sourceRGB[frame_idx].b;
						sourceRGB[frame_idx].r = 0;
						sourceRGB[frame_idx].g = 0;
						sourceRGB[frame_idx].b = 0;

						destDepth[frame_idx] = sourceDepth[frame_idx];
						sourceDepth[frame_idx] = 0.0f;
					}
				}
			}
		}

		void InstanceReconstructor::ProcessFrame(
				ITMLib::Objects::ITMView* main_view,
				const Segmentation::InstanceSegmentationResult& segmentation_result
		) {
//			if (! chunk_manager_->hasChunk(fingerprint)) {
//				Vector2i frame_size = main_view->rgb->noDims;
//				// bool use_gpu = main_view->rgb->isAllocated_CUDA; // May need to modify 'MemoryBlock' to
//				// check this, since the field is private.
//				chunk_manager_->createChunk(fingerprint, main_view->calib, frame_size, true);
//			}

			// TODO(andrei): Perform this slicing 100% on the GPU.
			main_view->rgb->UpdateHostFromDevice();
			main_view->depth->UpdateHostFromDevice();
//
			ORUtils::Vector4<unsigned char> *rgb_data_h = main_view->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
			float *depth_data_h = main_view->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

			if (segmentation_result.instance_detections.size() > 0) {
				int idx = 0;
				for(const InstanceDetection& instance_detection : segmentation_result.instance_detections) {
					// At this stage of the project, we only care about cars. In the future, this scheme could
					// be extended to also support other classes, as well as any unknown, but moving, objects.
					if (instance_detection.class_id == kPascalVoc2012.label_to_id.at("car")) {
						// TODO(andrei): Maybe just produce a list of chunks and hand them to the chunktracker?
						// I don't think we need a chunk manager and a chunk tracker.

						const string fingerprint = "car" + to_string(idx++);

						// TODO(andrei): This is NOT GOOD. We don't care about fingerprinting and shit here...
						if (! chunk_manager_->hasChunk(fingerprint)) {
							// TODO(andrei): If it's not too much of a PITA, consider always allocating chunks
							// the exact size of the detection's bounding box.
							chunk_manager_->createChunk(fingerprint, main_view->calib, main_view->rgb->noDims, true);
						}

						auto chunk = chunk_manager_->getChunk(fingerprint);
						auto rgb_segment_h = chunk->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
						auto depth_segment_h = chunk->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

						processSihlouette_CPU(rgb_data_h, depth_data_h, rgb_segment_h, depth_segment_h,
						                      main_view->rgb->noDims, instance_detection);
					}
				}
			}

			// TODO(andrei): Actuall pass ITMViews and shit to the instance tracker!
			vector<InstanceDetection> new_detections;
			new_detections.insert(
					new_detections.end(),
					segmentation_result.instance_detections.begin(),
					segmentation_result.instance_detections.end()
			);

			// Associate this frame's detection(s) with those from previous frames.
			this->instance_tracker_->ProcessChunks(frame_idx_, new_detections);

			main_view->rgb->UpdateDeviceFromHost();
			main_view->depth->UpdateDeviceFromHost();

			frame_idx_++;
		}
	}
}
