

#ifndef INFINITAM_SEGMENTATIONRESULT_H
#define INFINITAM_SEGMENTATIONRESULT_H

#include "SegmentationDataset.h"

#include <cstring>
#include <iostream>
#include <vector>

namespace InstRecLib {
	namespace Segmentation {

		class InstanceSegmentationResult;

		// TODO(andrei): Consider storing an entire class-conditional distribution in each detection
		// object.

		// TODO(andrei): Should we add inter-frame association information to this class? Or should we
		// pass an 'InstanceSegmentationResult' to another component which will then return instance
		// detections associated with their own reconstruction volumes?
		/*
		 * Yeah, it does make more sense to have a SegmentationAssociationProvider which takes in an
		 * InstanceSegmentationResult with its InstanceDetections, and outputs e.g., a list of
		 * (InstanceDetection, InstanceReconstruction) pairs, which can then be used for volum. Fusion.
		 *
		 * The Association Provider can be just an interface, and it could have multiple implementations
		 * under the hood, such as overlap-based (with configurable size/time/etc. thresholds),
		 * object recognition-based, hybrid, etc.
		 *
		 * Its job is to be able to associate objects detected in a frame with their appropriate 3D
		 * model, or to initialize new models as needed.
		 *
		 * Then, we could cut out the instance info from the current frame, and give these cut-out bits
		 * to the instance rec engine, in their equivalent ProcessFrame method. The only extra necessary
		 * thing is the approximate relative pose of the new frame, perhaps in the viewer's frame. Then
		 * the instance rec would be able to initialize its alignment system, e.g., ICP, accordingly,
		 * based on this initial estimate. We can get this estimate from scene flow.
		 *
		 * Such an instance reconstructor may also need to keep track of the instance's motion (either
		 * in the global frame or the car's frame; we'll see which works best---probably the global
		 * frame, since it seems neater), as well as other things, such as (maybe) the motion history,
		 * allowing us to render nice tracks in the 3D map, etc.
		 */

		/// \brief Describes a single object instance detected semantically in an input frame.
		/// This is a component of InstanceSegmentationResult.
		class InstanceDetection {
		public:
			/// The detection's bounding box, expressed in pixels (x1, y1, x2, y2). Coordinates are
			/// <i>inclusive</i>.
			/// In order to compute, e.g., the box's width, one can simply use
			/// `bounding_box[2] - bounding_box[0] + 1`.
			///
			/// TODO(andrei): Where is (0, 0)?
			int bounding_box[4];

			/// This detection is the highest-likelihood class from a proposal region.
			/// This is its probability. It's usually pretty high (>0.9), but can
			/// still be useful for various sanity checks.
			float class_probability;

			/// Class identifier. Depends on the class labels used by the segmentation pipeline,
			/// specified in the associated `InstanceSegmentationResult`.
			int class_id;

			// TODO(andrei): Proper resource management.
			/// 2D binary array indicating the instance pixels, within the bounding box.
			/// It has the same dimensions as the bounding box.
			uint8_t **mask;

			/// The result object to which this instance detection belongs.
			InstanceSegmentationResult* parent_result;

			std::string getClassName() const;

			/// \brief Initializes the detection with bounding box, class, and mask information.
			/// \note This object takes ownership of `mask`.
			InstanceDetection(int *bounding_box, float class_probability, int class_id, uint8_t **mask,
			                  InstanceSegmentationResult *parent_result)
					: class_probability(class_probability), class_id(class_id), mask(mask),
					  parent_result(parent_result)
			{
				for(size_t i = 0; i < 4; ++i) {
					this->bounding_box[i] = bounding_box[i];
				}
			}

			void Set(const InstanceDetection& rhs) {
				// TODO(andrei): Refactor duplicate code.
				// TODO(andrei): Would be faster to wrap the mask properly in a shared ptr.
				for(size_t i = 0; i < 4; ++i) {
					this->bounding_box[i] = rhs.bounding_box[i];
				}

				this->class_probability = rhs.class_probability;
				this->class_id = rhs.class_id;
				this->parent_result = rhs.parent_result;

				int height = bounding_box[3] - bounding_box[1] + 1;
				int width = bounding_box[2] - bounding_box[0] + 1;
				this->mask = new uint8_t*[height];
				for(int i = 0; i < height; ++i) {
					this->mask[i] = new uint8_t[width];
					std::memcpy(this->mask[i], rhs.mask[i], width * sizeof(uint8_t));
					// TODO(andrei): Actually copy stuff.
				}

			}

			InstanceDetection(const InstanceDetection &rhs) {
				this->Set(rhs);
			}

			InstanceDetection& operator=(const InstanceDetection& rhs) {
				this->Set(rhs);
				return *this;
			}

			virtual ~InstanceDetection() {
				if (mask) {
					// TODO(andrei): Handle the mask in a more modern way!
					int height = bounding_box[3] - bounding_box[1] + 1;
					for (int row = 0; row < height; ++row) {
						if (mask[row]) {
							delete[] mask[row];
						}
					}
					delete[] mask;
				}
			}
		};

		/// \brief Supports pretty-printing segmentation result objects.
		std::ostream& operator<<(std::ostream& out, const InstanceDetection& detection);

		/// \brief The result of performing instance-aware semantic segmentation on an input frame.
		struct InstanceSegmentationResult {

			/// \brief Specifies the dataset metadata, such as the labels, which are used by the segmentation.
			SegmentationDataset segmentation_dataset;

			/// \brief The instances detected in the frame.
			std::vector<InstanceDetection> instance_detections;

			/// \brief The total time it took to produce this result.
			long inference_time_ns;

			InstanceSegmentationResult(const SegmentationDataset &segmentation_dataset,
			                           const std::vector<InstanceDetection> &instance_detections,
			                           long inference_time_ns
			) : segmentation_dataset(segmentation_dataset),
			    instance_detections(instance_detections),
			    inference_time_ns(inference_time_ns)
			{
				// A little bit of plumbing: ensure instance detection objects know their parent.
				for (InstanceDetection& detection : this->instance_detections) {
					detection.parent_result = this;
				}
			}
		};
	}
}


#endif //INFINITAM_SEGMENTATIONRESULT_H
