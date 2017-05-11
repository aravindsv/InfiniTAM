

#include "PrecomputedSegmentationProvider.h"
#include "InstanceSegmentationResult.h"

// TODO(andrei): Get rid of this dependency.
#include "../Utils/FileUtils.h"

#include <cstdlib>
#include <sstream>

namespace InstRecLib {
	namespace Segmentation {

		using namespace std;

		vector<InstanceDetection> PrecomputedSegmentationProvider::ReadInstanceInfo(
				const std::string &base_img_fpath
		) {
			// Loop through all possible instance detection dumps for this frame.
			//
			// They are saved by the pre-segmentation tool as:
			//     '${base_img_fpath}.${instance_idx}.{result,mask}.txt'.
			//
			// The result file is one line with the format "[x1 y1 x2 y2 junk], probability, class". The
			// first part represents the bounding box of the detection.
			//
			// The mask file is a numpy text file containing the saved (boolean) mask created by the neural
			// network. Its size is exactly the size of the bounding box.

			int instance_idx = 0;
			vector<InstanceDetection> detections;
			while (true) {
				stringstream result_fpath;
				result_fpath << base_img_fpath << "." << setfill('0') << setw(4) << instance_idx
				             << ".result.txt";

				stringstream mask_fpath;
				mask_fpath << base_img_fpath << "." << setfill('0') << setw(4) << instance_idx
				           << ".mask.txt";

				ifstream result_in(result_fpath.str());
				ifstream mask_in(mask_fpath.str());
				if (!(result_in.is_open() && mask_in.is_open())) {
					break;
				}

				string result;
				getline(result_in, result);

				// Because screw C++'s string IO. What were they thinking??
				int bbox[4];
				float class_probability;
				int class_id;
				sscanf(result.c_str(), "[%d %d %d %d %*d], %f, %d",
				       &bbox[0], &bbox[1], &bbox[2], &bbox[3], &class_probability, &class_id);

//				cout << "Found: " << dataset_used.labels[class_id - 1] << " at ["
//				     << bbox[0] << ", " << bbox[1] << ", " << bbox[2] << ", " << bbox[3] << "]." << endl;

				detections.emplace_back(bbox, class_probability, class_id, nullptr, nullptr);

				instance_idx++;
			}

			return detections;
		}

		shared_ptr<InstanceSegmentationResult> PrecomputedSegmentationProvider::SegmentFrame(
				ITMLib::Objects::ITMView *view
		) {
			stringstream img_ss;
			img_ss << this->segFolder_ << "/" << "cls_" << setfill('0') << setw(6) << this->frameIdx_
			       << ".png";
			const string img_fpath = img_ss.str();
			ReadImageFromFile(lastSegPreview_, img_fpath.c_str());

			stringstream meta_ss;
			meta_ss << this->segFolder_ << "/" << setfill('0') << setw(6) << this->frameIdx_ << ".png";
			vector<InstanceDetection> instance_detections = ReadInstanceInfo(meta_ss.str());

			// We read data off the disk, so we assume this is 0.
			long inference_time_ns = 0L;

			// Experimental blanking-out code. # TODO(andrei): Move to own component.
			ORUtils::Vector4<unsigned char> *rgb_data_d = view->rgb->GetData(
					MemoryDeviceType::MEMORYDEVICE_CUDA);

//		rgb_data += 150;

			ORUtils::Vector4<unsigned char> *rgb_data_h = new ORUtils::Vector4<unsigned char>[10000];
			ORcudaSafeCall(cudaMemcpy(rgb_data_h, rgb_data_d, 1000, cudaMemcpyDeviceToHost));

			// TODO(andrei): Use this buffer to blank things out!

//		auto rgb_data_h_it = rgb_data_h;
//		for(int i = 0; i < 1000; ++i) {
//			if (rgb_data_h->x != 0) {
//				printf("%d: %d %d %d\n", i, rgb_data_h_it->x, rgb_data_h_it->y, rgb_data_h_it->z);
//			}
//
//			rgb_data_h_it++;
//		}

			// TODO(andrei): Upload buffer back to GPU if needed (likely is).

			delete[] rgb_data_h;
			this->frameIdx_++;

			return make_shared<InstanceSegmentationResult>(
					dataset_used,
			    instance_detections,
			    inference_time_ns
			);
		}

		const ORUtils::Image<Vector4u> *PrecomputedSegmentationProvider::GetSegResult() const {
			return this->lastSegPreview_;
		}

		ORUtils::Image<Vector4u> *PrecomputedSegmentationProvider::GetSegResult() {
			return this->lastSegPreview_;
		}
	}
}
