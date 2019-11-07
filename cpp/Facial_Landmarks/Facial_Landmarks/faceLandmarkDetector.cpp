#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include "renderFace.h"

using namespace dlib;
using namespace std;

void writeLandmarksToFile(full_object_detection &landmarks, const string &filename)
{
	std::ofstream ofs;
	ofs.open(filename);

	for (int i = 0; i < landmarks.num_parts(); i++)
	{
		ofs << landmarks.part(i).x() << " " << landmarks.part(i).y() << endl;

	}
	ofs.close();
}

int faceLandmarkDetector(int argc, char** argv)
{

	frontal_face_detector faceDetector = get_frontal_face_detector();
	shape_predictor landmarkDetector; // shape_predictor class

	deserialize("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

	string imageFilename("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/39.png");
	cv::Mat im = cv::imread(imageFilename);

	string landmarksBasename("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/results/family");

	cv_image<bgr_pixel> dlibIm(im); // opencv image -> dlib image

	std::vector<rectangle> faceRects = faceDetector(dlibIm);
	std::vector<full_object_detection> landmarksAll;
	//cout << "Number of faces detected: " << faceRects.size() << endl;

	for (int i = 0; i < faceRects.size(); i++)
	{
		full_object_detection landmarks = landmarkDetector(dlibIm, faceRects[i]);

		if (i == 0) cout << "Number of landmarks : " << landmarks.num_parts() << endl;

		landmarksAll.push_back(landmarks);

		renderFace(im, landmarks);

		std::stringstream landmarksFilename;
		landmarksFilename << landmarksBasename << "_" << i << ".txt";
		cout << "Saving landmarks to " << landmarksFilename.str() << endl;
		writeLandmarksToFile(landmarks, landmarksFilename.str());

	}

	string outputFilename("C:/Users/xwen2/Desktop/Computer Vision Projects/Face Landmarks/data/images/results/familyLandmarks.jpg");
	cout << "Saving output image to " << outputFilename << endl;
	cv::imwrite(outputFilename, im);

	resize(im, im, cv::Size(), 0.2, 0.2);
	cv::imshow("Facial Landmark Detector", im);
	cv::waitKey(0);

	return 0;
}

