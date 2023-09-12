#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;

class Webcam {
public:
	Webcam(int id) : 
		id(id), 
		cap(id) {
		if (!cap.isOpened()) {
			std::cout << "Error opening video stream or file\n";
		}
	}

	void test() {		
		while (1) {
			Mat frame = getFrame();
			findDigits(frame);

			if (frame.empty()) {
				break;
			}

			imshow("Frame", frame);

			char c = (char)waitKey(25);
			if (c == 27) {
				break;
			}
		}

		destroyAllWindows();
	}

	std::vector<std::pair<Rect, Mat>> findDigits(Mat& frame) {
		Mat original = frame.clone();
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		GaussianBlur(frame, frame, Size(5, 5), 0);
		//imshow("cFrame", frame);

		adaptiveThreshold(frame, frame, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
		//imshow("oFrame", frame);

		GaussianBlur(frame, frame, Size(5, 5), 0);
		threshold(frame, frame, 100, 255, THRESH_BINARY);

		//GaussianBlur(frame, frame, Size(5, 5), 0);
		//threshold(frame, frame, 1, 255, THRESH_BINARY);

		std::vector<std::vector<Point>> contours;
		findContours(frame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		std::vector<std::vector<Point>> contours_poly(contours.size());
		std::vector<std::pair<Rect, Mat>> digits;

		imshow("Processed", frame);

		for (size_t i = 0; i < contours.size(); i++) {
			approxPolyDP(contours[i], contours_poly[i], 3, true);
			Rect rect = boundingRect(contours_poly[i]);
			if (rect.area() > 300 && rect.width < frame.cols * 0.8 && rect.height < frame.rows * 0.8) {
				if (digits.size() < 10) {
					Mat digit = frame(rect);
					const int space = 4;
					float ratio = std::clamp((float)digit.cols / (float)digit.rows, 0.05f, 1.0f);
					//std::cout << ratio << std::endl;
					resize(digit, digit, Size((int)((28 - 2 * space) * ratio), (28 - 2 * space)));

					int diff = std::clamp<int>((28 - 2 * space) - (int)((28 - 2 * space) * ratio), 0, (28 - 2 * space));
					//std::cout << std::floor(diff / 2.0f) << " " << std::ceil(diff / 2.0f) << std::endl;
					//std::cout << digit.rows << " " << digit.cols << std::endl;

					copyMakeBorder(digit, digit, space, space, std::floor(diff / 2.0f) + space, std::ceil(diff / 2.0f) + space, BORDER_CONSTANT, Scalar(0, 0, 0));
					//std::cout << digit.rows << " " << digit.cols << std::endl;
					
					digits.emplace_back(rect, digit);
				}
			}
		}

		std::sort(digits.begin(), digits.end(), [](const std::pair<Rect, Mat>& a, const std::pair<Rect, Mat>& b) {
			return a.first.x < b.first.x;
		});

		for (auto& bound : digits) {
			rectangle(original, bound.first.tl(), bound.first.br(), Scalar(0, 255, 0), 2);
		}
		imshow("Camera", original);

		Mat previews = Mat::zeros(Size(28 * 10, 28), CV_8UC1);
		for (size_t i = 0; i < digits.size(); i++) {
			//resize(digits[i], digits[i], Size(28, 28));
			//imshow("Digit" + std::to_string(i), digits[i]);
			//waitKey(0);

			digits[i].second.copyTo(previews(Rect(i * 28, 0, 28, 28)));
		}

		imshow("Found digits", previews);

		return digits;
	}

	Mat getFrame() {
		Mat frame;
		cap >> frame;
		return frame;
	}

	~Webcam() {
		cap.release();
	}

private:
	int id;
	VideoCapture cap;
};
