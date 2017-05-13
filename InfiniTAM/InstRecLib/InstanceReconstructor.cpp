

#include "InstanceReconstructor.h"

namespace InstRecLib {
	namespace Reconstruction {
		using namespace InstRecLib::Segmentation;

		// TODO(andrei): Implement this in CUDA. It should be easy.
		void processSihlouette_CPU(
				ITMUChar4Image& sourceRGB,
				ITMUCharImage& sourceDepth,
				ITMUChar4Image& destRGB,
				ITMUCharImage& destDepth,
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

			int width = sourceRGB.noDims[0];
			auto sourceRGB_ptr_h = sourceRGB.GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

			// TODO(andrei): implement.
			for(int row = 10; row < 200; ++row) {
				for(int col = 10; col < 550; ++col) {
					auto rgb_data_h_it = sourceRGB_ptr_h + (row * width + col);
					rgb_data_h_it->r = 0;
					rgb_data_h_it->g = 0;
					rgb_data_h_it->b = 0;

					// TODO(andrei): Are the CPU-specific itam functions doing this in a nicer way?
				}
			}
		}

		void InstanceReconstructor::ProcessFrame(
				ITMLib::Objects::ITMView* main_view,
				const Segmentation::InstanceSegmentationResult& segmentation_result
		) {
			std::cout << "Will cut things up and paste them to their own buffers!" << std::endl;

			// For now, we pretend there's only one instance out there, and it has a silly ID.
			const std::string fingerprint = "car";
			if (! chunk_manager_->hasChunk(fingerprint)) {
				Vector2i frame_size = main_view->rgb->noDims;
				// bool use_gpu = main_view->rgb->isAllocated_CUDA; // May need to modify 'MemoryBlock' to
				// check this.
				std::cout << "Generating default experimental chunk..." << std::endl;
				chunk_manager_->createChunk(fingerprint, main_view->calib, frame_size, true);
			}

			// TODO(andrei): Perform this slicing 100% on the GPU.
			main_view->rgb->UpdateHostFromDevice();
			main_view->depth->UpdateHostFromDevice();
//
			ORUtils::Vector4<unsigned char> *rgb_data_h = main_view->rgb->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
			float *depth_data_h = main_view->depth->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

			chunk_manager_->getChunk()
			processSihlouette_CPU(rgb_data_h, depth_data_h, rgb_segment_h, depth_segment_h);

			main_view->rgb->UpdateDeviceFromHost();
			main_view->depth->UpdateDeviceFromHost();
		}
	}
}
