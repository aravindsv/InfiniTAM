
#ifndef INFINITAM_ITMGROUNDTRUTHTRACKER_H
#define INFINITAM_ITMGROUNDTRUTHTRACKER_H

#include <iostream>

#include "ITMTracker.h"

namespace ITMLib {
  namespace Engine {

    using namespace ITMLib::Objects;
    using namespace std;


    /**
     * Dummy tracker which relays pose information from a file.
     */
    class ITMGroundTruthTracker : public ITMTracker {

    private:
      const string groundTruthFpath;

    protected:

    public:
      ITMGroundTruthTracker(const string &groundTruthFpath) {
        cout << "Created experimental ground truth-based tracker." << endl;
        cout << "Will read data from: " << groundTruthFpath << endl;

        // TODO read OxTS dump using KITTI toolkit and look at pose info.
      }

      void TrackCamera(ITMTrackingState *trackingState, const ITMView *view) {
        // TODO(andrei): populate the appropriate attribute(s) of trackingState.
        cout << "Ground truth tracking." << endl;

        const float dummy[] = {
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0
        };
        Matrix4f currentM(dummy);
        trackingState->pose_d->SetM(currentM);
      }

      // Note: this doesn't seem to get used much in InfiniTAM. It's just
      // called from 'ITMMainEngine', but none of its implementations
      // currently do anything.
      void UpdateInitialPose(ITMTrackingState *trackingState) {
      }

      virtual ~ITMGroundTruthTracker() {}

    };

  }
}

#endif //INFINITAM_ITMGROUNDTRUTHTRACKER_H
