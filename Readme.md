# Panorama #
> CS698U (Computer Vision)
  
* Interest points of an image are located using SIFT of python openCV package and those are matched between two images using FLANN based matcher.

* Ransac algorithm is used to maximize the number of inliers and DLT (Direct Linear Transform) is used to compute Homography.

* Images are stitched using "mix_and_match" method taken from "http://kushalvyas.github.io/stitching.html". 

