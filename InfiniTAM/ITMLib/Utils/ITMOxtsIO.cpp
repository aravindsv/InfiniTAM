
#include <cstdio>
#include "ITMOxtsIO.h"

using namespace std;

namespace ITMLib {
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
      const string ts_fpath = dir + "/timestamps.txt";
      ifstream fin(ts_fpath.c_str());
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


    vector<OxTSFrame> readOxtsliteData(const string& dir) {
      // TODO
      vector<OxTSFrame> res;

      auto timestamps = readTimestamps(dir);
      // then read the actual stuff from 'dir/data'.

      cout << "Read " << timestamps.size() << " timestamps." << endl;

      return res;
    }

  }
}

