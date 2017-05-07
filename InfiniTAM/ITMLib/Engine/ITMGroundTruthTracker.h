
#ifndef INFINITAM_ITMGROUNDTRUTHTRACKER_H
#define INFINITAM_ITMGROUNDTRUTHTRACKER_H

#include <iostream>
#include <fstream>

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

      vector<Vector3f> groundTruthTrans;
      vector<Matrix3f> groundTruthRots;

	    ifstream myfile;    // the input stream for Xinyuan's code (TODO(andrei): pre-read this.)

      // TODO(andrei): Move this shit out of here.
      vector<Matrix4f> readKittiOdometryPoses(const std::string &fpath) {
        ifstream fin(fpath.c_str());
	      if (! fin.is_open()) {
          throw runtime_error("Could not open odometry file.");
        }

	      cout << "Loading odometry ground truth from file: " << fpath << endl;

        vector<Matrix4f> poses;
        while (! fin.eof()) {
	        // This matrix takes a point in the ith coordinate system, and projects it
	        // into the first (=0th, or world) coordinate system.
	        // Confirmed: in sequence with car moving forward, Z+ is forward (see ground
	        // truth for Kitti odometry sequence 6).
	        //
	        // When using built-in InfiniTAM ICP-based odometry, displaying the M matrix
	        // shows a constantly-decreasing Z when moving forward. Positive X seems associated
	        // with right.
	        // The M matrix in InfiniTAM is a modelview matrix, so it transforms points from
	        // the world coordinates, to camera coordinates.

	        // Sequence 06: When reading the pose matrix ground truth and setting the
	        // value of M to its value, the camera seems to be moving backwards and to the
	        // right, instead of forwards, and to the right.

          Matrix4f pose;
          fin >> pose.m00 >> pose.m10 >> pose.m20 >> pose.m30
              >> pose.m01 >> pose.m11 >> pose.m21 >> pose.m31
              >> pose.m02 >> pose.m12 >> pose.m22 >> pose.m32;
	        pose.m03 = pose.m13 = pose.m23 = 0.0f;
          pose.m33 = 1.0f;
	        poses.push_back(pose);
        }

        return poses;
      }

	    // Taken from 'ITMPose' to aid with debugging.
	    Matrix3f getRot(const Matrix4f M) {
		    Matrix3f R;
		    R.m[0 + 3*0] = M.m[0 + 4*0]; R.m[1 + 3*0] = M.m[1 + 4*0]; R.m[2 + 3*0] = M.m[2 + 4*0];
		    R.m[0 + 3*1] = M.m[0 + 4*1]; R.m[1 + 3*1] = M.m[1 + 4*1]; R.m[2 + 3*1] = M.m[2 + 4*1];
		    R.m[0 + 3*2] = M.m[0 + 4*2]; R.m[1 + 3*2] = M.m[1 + 4*2]; R.m[2 + 3*2] = M.m[2 + 4*2];
		    return R;
	    }

    protected:

    public:
      ITMGroundTruthTracker(const string &groundTruthFpath) {
        cout << "Created experimental ground truth-based tracker." << endl;
//        cout << "Will read data from: " << groundTruthFpath << endl;

        // TODO read OxTS dump using KITTI toolkit and look at pose info.

//        vector<OxTSFrame> groundTruthFrames = Objects::readOxtsliteData(groundTruthFpath);
        // TODO(andrei): We probably only care about relative poses, right?
//        groundTruthPoses = Objects::oxtsToPoses(groundTruthFrames, groundTruthTrans, groundTruthRots);
//	      groundTruthPoses = readKittiOdometryPoses("/home/andrei/datasets/kitti/odometry-dataset/poses/06.txt");
	      myfile = ifstream("/home/andrei/datasets/kitti/odometry-dataset/poses/06.txt");
      }

	    // Andrei's shitty slav version.
//      void TrackCamera(ITMTrackingState *trackingState, const ITMView *view) {
//
//				Matrix4f M = groundTruthPoses[currentFrame];
//	      Matrix3f R = getRot(M);
//	      cout << "Ground truth pose matrix: " << endl;
//	      prettyPrint(cout, M);
//	      cout << "Rotation of the pose matrix: " << endl << R << endl;
//
//	      if (fabs(R.det() - 1.0f) > 0.001) {
//		      cerr << "WARNING: Detected left-handed rotation matrix!!!" << endl;
//		      cout << "det(R) = " << R.det() << endl;
//	      }
//
//	      Matrix4f invM;
//	      M.inv(invM);
//	      cout << "Tracking normalization coef: " << invM.m33 << endl;
//
//	      // TODO(andrei): Figure out how to correct the pose for proper integration.
//
//	      // The sane thing *seems* to be to set the inverse of M to the GT pose we read,
//	      // since that *seems* to be the convention InfiniTAM is using.
//
//	      trackingState->pose_d->SetM(invM);
////	      trackingState->pose_d->Coerce();
//
//	      cout << "New pose in trackingState:" << endl;
//	      prettyPrint(cout, trackingState->pose_d->GetM());
//	      cout << endl;
//
//	      this->currentFrame++;
//      }

      // Xiyuan's glorious chinese version.
      void TrackCamera(ITMTrackingState *trackingState, const ITMView *view) {
	      std::cout<<"Tracking Pose"<<std::endl<<trackingState->pose_d->GetInvM()<<std::endl;
	      double poseValue[3][4];
	      Matrix4f currentPose = * (new Matrix4f());
	      Matrix4f kinectTrans = Matrix4f();
	      float kinectTransVal[16] = {1.0, 0.0 , 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
	      for(int i=0;i<16;i++)
	      {
		      kinectTrans.m[i] = kinectTransVal[i];
	      }
	      Matrix4f kinectTransInv = Matrix4f();
	      kinectTrans.inv(kinectTransInv);
	      std::cout<<"inverse kinect pose"<<kinectTransInv<<std::endl;

	      Matrix4f transPose = Matrix4f();
	      Matrix4f invPose = Matrix4f();
	      Matrix4f inversePose = Matrix4f();
	      Matrix4f transInvPose = Matrix4f();
	      std::cout<<"GroundTruth Tracking Pose"<<std::endl;
	      for(int i=0;i<3;i++)
	      {
		      for(int j=0;j<4;j++)
		      {
			      myfile>>poseValue[i][j];
			      currentPose.m[i*4+j]= poseValue[i][j];
			      std::cout<<poseValue[i][j]<<" ";
		      }
		      std::cout<<std::endl;
	      }
	      currentPose.m[12] = 0.0;
	      currentPose.m[13] = 0.0;
	      currentPose.m[14] = 0.0;
	      currentPose.m[15] = 1.0;

//    transPose = kinectTrans* currentPose * kinectTransInv;
	      transPose = currentPose;


//    for(int i=0;i<3;i++)
//    {
//        for(int j=0;j<3;j++)
//        {
//            inversePose.m[4*j+i] = transPose.m[4*i+j];
//        }
//    }
//
//    for(int i=0;i<3;i++)
//    {
//        inversePose.m[4*i+3] = 0.0;
//        for(int j=0;j<3;j++)
//        {
//            inversePose.m[4*i+3] = inversePose.m[4*i+3] - transPose.m[4*i+3]*inversePose.m[4*i+j];
//        }
//    }
//    inversePose.m[12] = 0.0;
//    inversePose.m[13] = 0.0;
//    inversePose.m[14] = 0.0;
//    inversePose.m[15] = 1.0;
//

	      transPose.inv(invPose);
	      std::cout<<invPose<<std::endl;
	      for(int i=0;i<4;i++)
	      {
		      for(int j=0;j<4;j++)
		      {
			      transInvPose.m[4*i+j] = invPose.m[4*j+i];
		      }
	      }
	      trackingState->pose_d->SetM(transInvPose);
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
