
#ifndef INFINITAM_ITMGROUNDTRUTHTRACKER_H
#define INFINITAM_ITMGROUNDTRUTHTRACKER_H

#include <iostream>

#include "ITMTracker.h"
#include "../Utils/ITMOxtsIO.h"

namespace ITMLib {
  namespace Engine {

    using namespace ITMLib::Objects;
    using namespace std;


    /**
     * Dummy tracker which relays pose information from a file.
     */
    class ITMGroundTruthTracker : public ITMTracker {

    private:
      int currentFrame = 0;
      vector<Matrix4f> groundTruthPoses;

    protected:

    public:
      ITMGroundTruthTracker(const string &groundTruthFpath) {
        cout << "Created experimental ground truth-based tracker." << endl;
        cout << "Will read data from: " << groundTruthFpath << endl;

        // TODO read OxTS dump using KITTI toolkit and look at pose info.

        vector<OxTSFrame> groundTruthFrames = Objects::readOxtsliteData(groundTruthFpath);
        // TODO(andrei): We probably only care about relative poses, right?
        groundTruthPoses = Objects::oxtsToPoses(groundTruthFrames);
      }

      void TrackCamera(ITMTrackingState *trackingState, const ITMView *view) {
        // TODO(andrei): populate the appropriate attribute(s) of trackingState.
        cout << "Ground truth tracking." << endl;

        this->currentFrame++;

        cout << "Old pose: " << endl;
        cout << trackingState->pose_d->GetM() << endl;

        trackingState->pose_d->SetM(groundTruthPoses[currentFrame]);

        cout << "New pose: " << endl;
        cout << trackingState->pose_d->GetM() << endl;
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
