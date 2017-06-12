
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "ITMOxtsIO.h"
#include "../../../../DynSLAM/Utils.h"

using namespace std;

namespace ITMLib {
  // TODO(andrei): Better namespace name?
  namespace Objects {

    void readTimestampWithNanoseconds(
        const string &input,
        tm *time,
        long *nanosecond
    ) {
      int year, month, day, hour, minute, second;
      sscanf(input.c_str(), "%d-%d-%d %d:%d:%d.%ld", &year, &month, &day, &hour,
             &minute, &second, nanosecond);
      time->tm_year = year;
      time->tm_mon = month - 1;
      time->tm_mday = day;
      time->tm_hour = hour;
      time->tm_min = minute;
      time->tm_sec = second;
    }

    vector<tm> readTimestamps(const string& dir) {
      const string timestampFpath = dir + "/timestamps.txt";
      ifstream fin(timestampFpath.c_str());
      vector<tm> timestamps;

      string line;
      while(getline(fin, line)) {
        tm timestamp;
        long ns;     // We ignore this for now.
        readTimestampWithNanoseconds(line, &timestamp, &ns);
        timestamps.push_back(timestamp);
      }

      return timestamps;
    }

    /// \brief Customized matrix printing useful for debugging.
    void prettyPrint(ostream &out, const Matrix4f& m) {
	    out << "[";
      for (size_t row = 0; row < 4; ++row) {
	      if (row > 0) {
		      out << " ";
	      }
	      out << "[";
        for (size_t col = 0; col < 4; ++col) {
          out << setw(6) << setprecision(4) << fixed << right << m.m[col * 4 + row];
	        if (col < 3) {
		        out << ", ";
	        }
        }
        out << "]" << endl;
      }
    }

    Matrix3d RotFromYaw(double yaw) {
      return Matrix3d(
          cos(yaw), -sin(yaw),             0,
          sin(yaw),  cos(yaw),             0,
          0,                0,             1
      ).t();
    }

    Matrix3d RotFromPitch(double pitch) {
      return Matrix3d(
          cos(pitch),          0,   sin(pitch),
          0,        1,         0,
          -sin(pitch),          0,   cos(pitch)
      ).t();
    }

    Matrix3d RotFromRoll(double roll) {
      return Matrix3d(
          1,          0,           0,
          0,  cos(roll),  -sin(roll),
          0,  sin(roll),   cos(roll)
      ).t();
    }

    Matrix3d RotFromEuler(double yaw, double pitch, double roll) {
      return RotFromYaw(yaw) * RotFromPitch(pitch) * RotFromRoll(roll);
    }

    Matrix4d TransformFromRotTrans(const Matrix3d &rot, const Vector3d &translation) {
      Matrix4d transform;
      for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
          transform(x, y) = rot(x, y);
        }
      }

      transform(3, 0) = translation[0];
      transform(3, 1) = translation[1];
      transform(3, 2) = translation[2];

      transform(0, 3) = 0;
      transform(1, 3) = 0;
      transform(2, 3) = 0;
      transform(3, 3) = 1;

      return transform;
    }

  /// \brief Compute the Mercator scale from the latitude.
    double latToScale(double latitude) {
      return cos(latitude * M_PI / 180.0);
    }

    /// \brief Converts lat/lon coordinates to Mercator coordinates.
    Vector2d latLonToMercator(double latitude, double longitude, double scale) {
      const double EarthRadius = 6378137.0;
      double mx = scale * longitude * M_PI * EarthRadius / 180.0;
      double my = scale * EarthRadius * log(tan( (90 + latitude) * M_PI / 360 ));
      return Vector2d(mx, my);
    }

    OxTSFrame readOxtslite(const string& fpath) {
      ifstream fin(fpath.c_str());
      if (! fin.is_open()) {
        throw runtime_error(dynslam::utils::Format("Could not open pose file [%s].", fpath.c_str()));
      }
      if (fin.bad()) {
        throw runtime_error(dynslam::utils::Format("Could not read pose file [%s].", fpath.c_str()));
      }

      OxTSFrame resultFrame;
      fin >> resultFrame.lat >> resultFrame.lon >> resultFrame.alt
          >> resultFrame.roll >> resultFrame.pitch >> resultFrame.yaw
          >> resultFrame.vn >> resultFrame.ve >> resultFrame.vf
          >> resultFrame.vl >> resultFrame.vu >> resultFrame.ax
          >> resultFrame.ay >> resultFrame.az >> resultFrame.af
          >> resultFrame.al >> resultFrame.au >> resultFrame.wx
          >> resultFrame.wy >> resultFrame.wz >> resultFrame.wf
          >> resultFrame.wl >> resultFrame.wu >> resultFrame.posacc
          >> resultFrame.velacc >> resultFrame.navstat >> resultFrame.numsats
          >> resultFrame.posmode >> resultFrame.velmode >> resultFrame.orimode;
      return resultFrame;
    }

    // TODO(andrei): There are 5+ different helper functions you can extract from this method to
    // make it leaner once it's working...
    Matrix4f poseFromOxts(const OxTSFrame &frame, double scale,
                          const Vector3d &t_0,
                          const Matrix3d &R_w_I0,
                          int frameIdx) {
      Vector2d translation2d = latLonToMercator(frame.lat, frame.lon, scale);
      Vector3d translation(translation2d.x, translation2d.y, frame.alt);

      translation -= t_0;

      cout << "Translation: " << translation << endl;

      // Extract the 3D rotation information from yaw, pitch, and roll.
      // See the OxTS user manual, pg. 71, for more information.
      Matrix3d R_w_Ik = RotFromEuler(frame.yaw, frame.pitch, frame.roll);
      Matrix3d R_I0_w = R_w_I0.t();
      Matrix3d R_I0_Ik = R_w_Ik * R_I0_w;

      Matrix3d R_I0_Ik_neg = R_I0_Ik;
      R_I0_Ik_neg *= -1.0;

      Vector3d t_k = R_I0_Ik_neg * translation;

      cout << "Rotation: " << endl << R_I0_Ik << endl;
      cout << "Rotated translation: " << t_k << endl;

      Matrix4d T_I0_Ik = TransformFromRotTrans(R_I0_Ik, t_k);

      // TODO(andrei): Read this from a file.
      // TODO(andrei): Double-check this:
      // This matrix takes points in the IMU frame, and transforms them into the Velodyne frame.
      const Matrix4d kImuToVeloKitti = Matrix4d(
           9.999976e-01, 7.553071e-04, -2.035826e-03, -8.086759e-01,
          -7.854027e-04, 9.998898e-01, -1.482298e-02,  3.195559e-01,
           2.024406e-03, 1.482454e-02,  9.998881e-01, -7.997231e-01,
                    0.0,          0.0,           0.0,           1.0
      ).t();
      // This matrix takes points in the velodyne frame, and transforms them into the left camera's
      // frame. TODO(andrei): Color or gray? Does it matter?
      const Matrix4d kVeloToCamKitti = Matrix4d(
           7.027555e-03, -9.999753e-01,  2.599616e-05, -7.137748e-03,    //  0 -1  0  0.0
          -2.254837e-03, -4.184312e-05, -9.999975e-01, -7.482656e-02,    //  0  0 -1  0.0
           9.999728e-01,  7.027479e-03, -2.255075e-03, -3.336324e-01,    //  1  0  0 -0.3
                    0.0,           0.0,           0.0,           1.0
      ).t();
      const Matrix4d kVeloToCamKitti2 = Matrix4d(
        -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
        -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
         9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
         0, 0, 0, 1
      ).t();

      Matrix4d T_IMU_Cam = kVeloToCamKitti * kImuToVeloKitti;
      Matrix4d T_IMU_Cam_inv;
      if (! T_IMU_Cam.inv(T_IMU_Cam_inv)) {
        throw runtime_error("Ill-posed inversion.");
      }

      // T_IMU_Cam = T_I*_Cam* = T_Ik_Camk, since the transform between the IMU and the camera is
      // always the same, since they're both on the car.
      Matrix4d T_Cam0_Camk = T_IMU_Cam * T_I0_Ik * T_IMU_Cam_inv;

      Matrix4d R_rect_00 = Matrix4d(
        9.999280e-01, 8.085985e-03, -8.866797e-03, 0,
        -8.123205e-03, 9.999583e-01, -4.169750e-03, 0,
        8.832711e-03, 4.241477e-03, 9.999520e-01, 0,
        0, 0, 0, 1
      );

//      Matrix4d tinv;
//      transform.inv(tinv);

      // This IMHO looks like the correct way, but doesn't work
      // From the dataset devkit docs:
      // To transform a point X from GPS/IMU coordinates to the image plane:
      //   Y = P_rect_xx * R_rect_00 * (R|T)_velo_to_cam * (R|T)_imu_to_velo * X
      // Since we're not projecting points to 2D, what we basically care about is the following
      //   Y_cam = R_rect_00 * velo2cam * imuToVelo * X;
      // So the transform we're interested in would be:
      //   Y_cam = transformCam * X
      //   transformCam = R_rect_00 * velo2cam * imuToVelo;
//      Matrix4d transformCam = R_rect_00 * kVeloToCamKitti * kImuToVeloKitti * transform;
//      Matrix4d transformCam = R_rect_00 * kVeloToCamKitti * kImuToVeloKitti * tinv;

      Matrix4f transformCam_f;
      for(int x = 0; x < 4; ++x) {
        for(int y = 0; y < 4; ++y) {
          transformCam_f.at(x, y) = static_cast<float>(T_Cam0_Camk.at(x, y));
        }
      }
      return transformCam_f;
    }

    vector<Matrix4f> oxtsToPoses(const vector<OxTSFrame>& oxtsFrames,
                                 vector<Vector3f>& trans,
                                 vector<Matrix3f>& rots
    ) {
      double scale = latToScale(oxtsFrames[0].lat);
      vector<Matrix4f> poses;

      // TODO(andrei): Ensure matrix initialization order is correct. The
      // Matrix classes used here are COLUMN major!!
      // TODO(andrei): Ensure precision is consistent.

      const OxTSFrame &first_frame = oxtsFrames[0];

      Vector2d translation2d = latLonToMercator(first_frame.lat, first_frame.lon, scale);
      Vector3d t_0(translation2d.x, translation2d.y, first_frame.alt);
      Matrix3d R_w_I0 = RotFromEuler(first_frame.yaw, first_frame.pitch, first_frame.roll);
      Matrix4f first_pose = poseFromOxts(first_frame, scale, t_0, R_w_I0, 0);

      Matrix4f tr_0_inv;
      if (!first_pose.inv(tr_0_inv)) {
        throw runtime_error("Ill-posed transform matrix inversion");
      }

      for (size_t frameIdx = 1; frameIdx < oxtsFrames.size(); ++frameIdx) {
        const OxTSFrame &frame = oxtsFrames[frameIdx];
        Matrix4f pose = poseFromOxts(frame, scale, t_0, R_w_I0, frameIdx);
        Matrix4f newPose = tr_0_inv * pose;
//        pose.inv(tr_0_inv);    // TODO(andrei): Rename this to make it clearer.
        poses.push_back(newPose);

        // TODO(andrei): Dump converted GPS/IMU data and plot it in python as a sanity check.

        if (frameIdx < 10) {
          cout << "New transform (relative to first): " << endl
               << newPose << endl;
        }

      }

      // TODO(andrei): We may actually just require incremental poses, not
      // absolute ones. Documnent the output very clearly either way!

      return poses;
    }

    /// \brief Reads a set of ground truth OxTS IMU/GPU unit from a directory.
    /// \note This function, together with its related utilities, are based on the KITTI devkit
    /// authored by Prof. Andreas Geiger.
    vector<OxTSFrame> readOxtsliteData(const string& dir) {
      // TODO(andrei): In the future, consider using the C++14 filesystem API
      // to make this code cleaner and more portable.
      vector<OxTSFrame> res;

      auto timestamps = readTimestamps(dir);
//      cout << "Read " << timestamps.size() << " timestamps." << endl;

      for(size_t i = 0; i < timestamps.size(); ++i) {
        stringstream ss;
        ss << dir << "/data/" << setw(10) << setfill('0') << i << ".txt";
//        cout << ss.str() << endl;

        // TODO(andrei): Should we expect missing data? Does that occur in
        // KITTI?
        res.push_back(readOxtslite(ss.str().c_str()));
      }

      return res;
    }

  }
}

