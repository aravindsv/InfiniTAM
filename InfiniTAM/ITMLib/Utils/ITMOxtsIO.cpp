
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
    Matrix4f poseFromOxts(const OxTSFrame &frame, double scale, int frameIdx) {
      Vector2d translation2d = latLonToMercator(frame.lat, frame.lon, scale);
      Vector3d translation(translation2d.x, translation2d.y, frame.alt);

      // Hacky centering; TODO be neater
//      translation -= Vector3d(612608, 4118210, 113.112);

      // Extract the 3D rotation information from yaw, pitch, and roll.
      // See the OxTS user manual, pg. 71, for more information.
      double roll  = frame.roll;
      double pitch = frame.pitch;
      double yaw   = frame.yaw;

      if (frameIdx < 10) {
//        cout << "Roll, pitch, yaw (heading)" << endl;
//        cout << roll << " " << pitch << " " << yaw << endl;
      }

      Matrix3d rotX = Matrix3d(
          1,          0,           0,
          0,  cos(roll),  -sin(roll),
          0,  sin(roll),   cos(roll)
      ).t();
      Matrix3d rotY = Matrix3d(
         cos(pitch),          0,   sin(pitch),
         0,        1,         0,
        -sin(pitch),          0,   cos(pitch)
      ).t();
      Matrix3d rotZ = Matrix3d(
          cos(yaw), -sin(yaw),             0,
          sin(yaw),  cos(yaw),             0,
          0,                0,             1
      ).t();

      Matrix3d rot = rotZ * rotY * rotX;

//        rots.push_back(rot);
//        trans.push_back(translation);

      Matrix4d transform;
      // TODO utility for this (setTransform(Matrix4f&, const Matrix3f&, const Vector3f&)
      for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
          transform(x, y) = rot(x, y);
        }
      }

//      transform(3, 0) = translation[1];
//      transform(3, 1) = translation[2];
//      transform(3, 2) = translation[0];
      transform(3, 0) = translation[0];
      transform(3, 1) = translation[1];
      transform(3, 2) = translation[2];

      // imu-to-velo (almost just a pure translation)
//      R: 9.999976e-01 7.553071e-04 -2.035826e-03
//        -7.854027e-04 9.998898e-01 -1.482298e-02
//         2.024406e-03 1.482454e-02  9.998881e-01
//      T: -8.086759e-01 3.195559e-01 -7.997231e-01
      // velo-to-cam (minor translation, axis XYZ -> Z(-X)(-Y)
      // R: 7.027555e-03 -9.999753e-01 2.599616e-05
      // -2.254837e-03 -4.184312e-05 -9.999975e-01
      // 9.999728e-01 7.027479e-03 -2.255075e-03
//      T: -7.137748e-03 -7.482656e-02 -3.336324e-01

      // TODO(andrei): Read this from a file.
      // Note that we're jumping through this hoop, namely IMU -> Velodyne -> Camera because that's
      // how the calibration matrices are specified for the KITTI dataset.
      const Matrix4d kImuToVeloKitti = Matrix4d(
           9.999976e-01, 7.553071e-04, -2.035826e-03, -8.086759e-01,
          -7.854027e-04, 9.998898e-01, -1.482298e-02,  3.195559e-01,
           2.024406e-03, 1.482454e-02,  9.998881e-01, -7.997231e-01,
                    0.0,          0.0,           0.0,           1.0
      ).t();
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

      transform(0, 3) = 0;
      transform(1, 3) = 0;
      transform(2, 3) = 0;
      transform(3, 3) = 1;

      Matrix4d invVeloToCamKitti;
      kVeloToCamKitti.inv(invVeloToCamKitti);
      Matrix4d invVeloToCamKitti2;
      kVeloToCamKitti2.inv(invVeloToCamKitti2);
      Matrix4d invImuToVeloKitti;
      kImuToVeloKitti.inv(invImuToVeloKitti);

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
      Matrix4d transformCam = R_rect_00 * kVeloToCamKitti * kImuToVeloKitti * transform;
//      Matrix4d transformCam = R_rect_00 * kVeloToCamKitti * kImuToVeloKitti * tinv;

      if (frameIdx < 10) {
//        cout << endl;
//
//        cout << kVeloToCamKitti << endl;
//        cout << kImuToVeloKitti << endl;
        cout << "Read transform (in INU frame):" << endl;
        cout << transform << endl;

//        cout << transformCam << endl;
//        cout << "[" << frameIdx << "] Translation (lat, lon, alt): " << translation << endl;
//        cout << "[" << frameIdx << "] Rotation    (yaw, pit, rol): "
//             << yawf << ", " << pitchf << ", " << rollf << endl;
//
//        cout << "[" << frameIdx << "] Rotation:   (matrix): " << endl << rot << endl;
//        cout << "[" << frameIdx << "] Transform:   " << endl;
//        prettyPrint(cout, transform);
//
//        cout << endl;
      }

      Matrix4f transformCam_f;
      for(int x = 0; x < 4; ++x) {
        for(int y = 0; y < 4; ++y) {
          transformCam_f.at(x, y) = static_cast<float>(transformCam.at(x, y));
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

      // Keeps track of the inverse transform of the initial pose.
      const OxTSFrame &first_frame = oxtsFrames[0];
      Matrix4f first_pose = poseFromOxts(first_frame, scale, 0);

      Matrix4f tr_0_inv;
      if (!first_pose.inv(tr_0_inv)) {
        throw runtime_error("Ill-posed transform matrix inversion");
      }

      for (size_t frameIdx = 1; frameIdx < oxtsFrames.size(); ++frameIdx) {
        const OxTSFrame &frame = oxtsFrames[frameIdx];
        Matrix4f pose = poseFromOxts(frame, scale, frameIdx);
        Matrix4f newPose = tr_0_inv * pose;
//        pose.inv(tr_0_inv);    // TODO(andrei): Rename this to make it clearer.
        poses.push_back(newPose);

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

