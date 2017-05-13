

#ifndef INFINITAM_INSTANCERECONSTRUCTOR_H
#define INFINITAM_INSTANCERECONSTRUCTOR_H

#include <memory>

#include "ChunkManager.h"
#include "InstanceSegmentationResult.h"

namespace InstRecLib {
	namespace Reconstruction {

		/// \brief Pipeline component responsible for reconstructing the individual object instances.
		class InstanceReconstructor {

		private:
			std::shared_ptr<ChunkManager> chunk_manager_;

		public:
			InstanceReconstructor() : chunk_manager_(new ChunkManager()){ }

			/// \brief Uses the segmentation result to remove dynamic objects from the main view and save
			/// them to separate buffers, which are then used for individual object reconstruction.
			///
			/// This is the ``meat'' of the reconstruction engine.
			///
			/// \param main_view The original InfiniTAM view of the scene. Gets mutated!
			/// \param segmentation_result The output of the view's semantic segmentation.
			void ProcessFrame(
					ITMLib::Objects::ITMView* main_view,
			    const Segmentation::InstanceSegmentationResult& segmentation_result
			);
		};
	}
}


#endif //INFINITAM_INSTANCERECONSTRUCTOR_H
