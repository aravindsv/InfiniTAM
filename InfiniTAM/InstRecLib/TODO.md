# TODO list for the Instance Reconstruction Library (InstRecLib)

## Overview
 * The project has started as an extension of InfiniTAM, but the goal is to make it generic enough
 so it can operate with any underlying SLAM framework, such as ElasticFusion, Kintinuous, gtsam, etc.
 * In order to achieve this, the code must be written in such a way that it only depends on
 InfiniTAM via adapter classes, which can easily be swapped out to allow working with a different
 base system. These dependencies currently include InfiniTAM-specific image manipulation routines,
 and the ITMView data wrapper.
 
## Tasks

 - [ ] Decide on InstRecLib-specific internal image/data representation. Should we really depend on OpenCV for this? OpenCV is a nasty dependency to have just for a little IO. But, then again, if we end up having to integrate tightly with Caffe, it requires OpenCV anyway...
 - [ ] View (ITMView) adapter class.
 - [ ] Image adapter class.
 