#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <chrono>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "GpuTimer.h"

#define THRESHOLD_VALUE 205

using namespace std;

/*
Functions not implemented or partially implemented
-find_all_symbols
-recognize isolated symbol
*/

int sliding_window_argmax(vector<int>& a, int k)
{
	int cursum = 0, maxsum = 0, maxInd = k;
	for (int i = 0; i < 2 * k + 1; i++)
	{
		maxsum += a[i];
	}
	cursum = maxsum;
	for (int i = 2 * k + 1; i < a.size(); i++)
	{
		cursum -= a[i - 2 * k - 1];
		cursum += a[i];
		if (cursum > maxsum)
		{
			maxInd = i - k;
			maxsum = cursum;
		}
	}
	return maxInd;
}

vector<int> getprojection(unsigned char* image, int numRows, int numCols, char axis = 'X', int startx = 0, int starty = 0, int endx = -1, int endy = -1)
{
	if (endx == -1 || endx >= numCols) endx = numCols - 1;
	if (endy == -1 || endy >= numRows) endy = numRows - 1;

	vector<int> result;
	if (axis == 'X') {
		result.resize(endx - startx + 1, 0);
		for (int i = starty; i <= endy; i++) {
			for (int j = startx; j <= endx; j++) {
				if (image[i * numCols + j] == 0) {
					result[j - startx]++;
				}
			}
		}
	}
	else {
		result.resize(endy - starty + 1, 0);
		for (int i = starty; i <= endy; i++) {
			for (int j = startx; j <= endx; j++) {
				if (image[i * numCols + j] == 0) {
					result[i - starty]++;
				}
			}
		}
	}
	return result;
}

void computeStaff(unsigned char* image, int &staff_thickness, int &staff_spacing, int numRows, int numCols)
{
	vector<int> whitehist(numRows + 1, 0);
	vector<int> blackhist(numRows + 1, 0);
	for (int j = 0; j < numCols; j++)
	{
		for (int i = 0; i < numRows; i++)
		{
			int seqlen = 0;
			while (i < numRows && image[i*numCols + j] > 0)
			{
				seqlen++; i++;
			}
			if (seqlen > 0)
			{
				whitehist[seqlen] += 1;
			}
			seqlen = 0;
			while (i < numRows && image[i*numCols + j] == 0)
			{
				seqlen++; i++;
			}
			if (seqlen > 0)
			{
				blackhist[seqlen] += 1;
			}
		}
	}
	staff_thickness = sliding_window_argmax(blackhist, 1) - 1;
	staff_spacing = sliding_window_argmax(whitehist, 1) + 1;
}

void find_staves(unsigned char *image, vector<vector<int> > &staves, int staff_thickness, int staff_spacing, int numRows, int numCols)
{
	vector<int> staff;
	vector<int> score = getprojection(image, numRows, numCols, 'Y');
	for (int i = 1; i < numRows; i++) {
		score[i] = score[i] + score[i - 1];
	}

	double confidence = 0.8;
	int threshold = numCols * staff_thickness;
	int row = 1;
	while (row < numRows - staff_thickness) {
		if (score[row + staff_thickness] - score[row - 1] > threshold * confidence) {
			staff.push_back(row + staff_thickness / 2);
			row += staff_spacing - 2;
		}
		else {
			row++;
		}
		if (staff.size() == 5) {
			staves.push_back(staff);
			staff.clear();
			confidence = 0.8;
		}
	}
}

vector<vector<int>> segment_by_staves(int numRows, vector<vector<int>> &staves, int staff_thickness, int staff_spacing)
{
	vector<vector<int>> track_bounds;
	for (int i = 0; i < staves.size(); i++)
	{
		int y1 = max(staves[i][0] - 3 * (staff_thickness + staff_spacing), 0);
		int y2 = min(staves[i][(int)staves[i].size() - 1] + 3 * (staff_thickness + staff_spacing), numRows);
		track_bounds.push_back({ y1,y2 });
	}
	return track_bounds;
}

vector<vector<int>> get_interesting_intervals(vector<int> &proj, int threshold)
{
	vector<vector<int>> boundaries;
	int i = 0;
	while (i < proj.size()) {
		if (proj[i] < threshold) i++;
		else {
			vector<int> boundary = { i };
			while (i < proj.size() && proj[i] >= threshold) {
				i++;
			}
			boundary.push_back(i);
			boundaries.push_back(boundary);
		}
	}
	return boundaries;
}

void find_all_symbols(unsigned char* image, vector<vector<cv::Point>> &symbols, int starty, int endy, int staff_thickness, int numRows, int numCols) {
	vector<int>xproj = getprojection(image, numRows, numCols, 'X', 0, starty, -1, endy);

	vector<vector<int>> vertical_boundaries = get_interesting_intervals(xproj, staff_thickness);

	for (int i = 0; i < vertical_boundaries.size(); i++) {
		vector<int> vboundary = vertical_boundaries[i];
		vector<int> yproj = getprojection(image, numRows, numCols, 'Y', vboundary[0], starty, vboundary[1], endy);

		vector<vector<int>> horizontal_boundaries = get_interesting_intervals(yproj, staff_thickness / 2);

		for (int j = 0; j < horizontal_boundaries.size(); j++) {
			vector<int> hboundary = horizontal_boundaries[j];
			symbols.push_back({ cv::Point(vboundary[0], starty + hboundary[0]), cv::Point(vboundary[1], starty + hboundary[1]) });
		}
	}
}

void computeRuns(unsigned char *image, int *dst, char axis, int numRows, int numCols) {
	vector<vector<int>>res(numRows, vector<int>(numCols, 0));
	if (axis == 'X') {
		for (int i = 0; i < numRows; i++) {
			int currentSequence = 0;
			for (int j = 0; j < numCols; j++) {
				if (image[i * numCols + j] == 0) {
					currentSequence++;
				}
				else {
					currentSequence = 0;
				}
				dst[i*numCols + j] = currentSequence;
			}
		}
	}
	else {
		for (int j = 0; j < numCols; j++) {
			int currentSequence = 0;
			for (int i = 0; i < numRows; i++) {
				if (image[i * numCols + j] == 0) {
					currentSequence++;
				}
				else {
					currentSequence = 0;
				}
				dst[i*numCols + j] = currentSequence;
			}
		}

	}
}

void remove_staff(unsigned char *image, vector<int> &staff, int staff_thickness, int staff_spacing, int numRows, int numCols) {
	int* Iv = (int*)malloc(sizeof(int) * numRows * numCols);
	computeRuns(image, Iv, 'Y', numRows, numCols);

	vector<int> istaff;

	for (int i = 0; i < 3; i++) {
		if (staff[0] - i * (staff_spacing + staff_thickness) >= 0) istaff.push_back(staff[0] - i * (staff_spacing + staff_thickness));
	}

	for (int i = 0; i < staff.size(); i++) istaff.push_back(staff[i]);

	for (int i = 1; i <= 3; i++) if (staff.back() + i * (staff_spacing + staff_thickness) < numRows) istaff.push_back(staff.back() + i * (staff_spacing + staff_thickness));

	for (int i = 0; i < istaff.size(); i++) {
		int x = istaff[i] + 1;
		if (x >= numRows) continue;
		bool flag = (i < 3 || i > 8);
		for (int j = 0; j < numCols; j++) {
			if (Iv[x * numCols + j] == 0) continue;

			int x2 = x;
			while (x2 < numRows && Iv[x2 * numCols + j]>0) x2++;
			if (Iv[(x2 - 1) * numCols + j] <= (1 + flag) * staff_thickness + 3) {
				int x1 = x2 - Iv[(x2 - 1) * numCols + j];
				while (x1 < x2) {
					image[x1 * numCols + j] = 1;
					x1++;
				}
			}
		}
	}
	free(Iv);
}

vector<vector<unsigned char>> transformBinaryImageToVector(unsigned char* image, int numRows, int numCols) {
	vector<vector<unsigned char>>res(numRows, vector<unsigned char>(numCols));
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			res[i][j] = image[i * numCols + j];
		}
	}
	return res;
}

vector<vector<vector<unsigned char>>> transformColoredImageToVector(unsigned char* image, int numRows, int numCols) {
	vector<vector<vector<unsigned char>>>res(numRows, vector<vector<unsigned char>>(numCols, vector<unsigned char>(3)));
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			for (int k = 0; k < 3; k++) {
				res[i][j][k] = image[(i * numCols + j) * 3 + k];
			}
		}
	}
	return res;
}

void to_grayscale_and_threshold(unsigned char *src, unsigned char *dst, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int gray_offset = i * width + j;
			int k = 3 * gray_offset;
			dst[gray_offset] = 0.299 * src[k] + 0.587 * src[k + 1] + 0.114 * src[k + 2];
			if (dst[gray_offset] >= THRESHOLD_VALUE) dst[gray_offset] = 1;
			else dst[gray_offset] = 0;
		}
	}
}

void scale(unsigned char *src, int k, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int idx = i * width + j;
			src[idx] *= k;
		}
	}
}

void draw_staff(cv::Mat img, vector<int> &staff, int thickness = 1) {
	for (int y : staff) {
		cv::line(img, cv::Point(0, y), cv::Point(img.cols - 1, y), cv::Scalar(0, 0, 255), thickness);
	}

}

map<string, cv::Mat> load_dictionary() {
	map<string, cv::Mat> dictionary;

	string filename = "templates/*.png";
	vector<cv::String> fn;
	cv::glob(filename, fn, false);

	for (size_t i = 0; i < fn.size(); i++) {
		cv::Mat raw_image = cv::imread(fn[i], cv::IMREAD_COLOR);

		unsigned char* binary_image = (unsigned char *)malloc(sizeof(unsigned char) * raw_image.rows * raw_image.cols);
		to_grayscale_and_threshold(raw_image.data, binary_image, raw_image.rows, raw_image.cols);

		int n = fn[i].size();
		string name = fn[i].substr(10, (n - 14));
		cv::Mat template_image(raw_image.rows, raw_image.cols, CV_8UC1, binary_image);
		dictionary[name] = template_image;
	}

	return dictionary;
}

void _match_and_slide(unsigned char* image, vector<cv::Point> &symbol, unsigned char *mask, int numRows, int numCols, int maskRows, int maskCols, double &score, cv::Point &pos, bool bound = false) {
	/*
	* Given an input binary image I, a bounding rectangle symbol, and a template mask, find the
	* most probably position where this template could be in this symbol by sliding this
	* template accross the symbol and counting the number of matching pixels.
	* A score is assigned equal to the number of matching pixels / number of pixels in the templat
	*/

	// Initialize symbol rectangle bounds and mask rectangle bounds
	cv::Point p1 = symbol[0];
	cv::Point p2 = symbol[1];

	// Initialize score and position variables

	// For most templates, the width and height should be a bit close to the symbol. 
	// However, for empty and filled note templates, these could be found anywhere within
	// the symbol. In order to control this behavior, we use the bound parameter.
	// If bound is set, we do the following:
	// In order to speed things up, only consider symbols whose height and width are close
	// to the height and width of the mask. This avoids trying to slide very small masks
	// over large symbols, which obviously do not match.

	// Find the ratio of symbol width to mask width and ration of symbol height to mask height
	// If these ratios are too large or too small, then don't try proceed.
	double rx = (p2.x - p1.x) * 1.0 / maskCols;
	double ry = (p2.y - p1.y) * 1.0 / maskRows;

	double min_ratio = 0.8;
	double max_ratio = 1.2;

	// If bound is not set, then we proceed normally:
	if (!bound || (bound && (min_ratio <= rx && rx <= max_ratio) && (min_ratio <= ry && ry <= max_ratio))) {
		// Loop over every pixel in the symbol to choose the top left corner (i, j) and try matching.
		// From experimentation, it helps to consider at the least the first pixel, even if
		// the bounds of the template exceed the bounds of the symbol. Usually, they don't exceed
		// exceed them by too much if there really is a match.
		for (int i = p1.y; i < max(p2.y - maskRows, p1.y + 1); i++) {
			for (int j = p1.x; j < max(p2.x - maskCols, p1.x + 1); j++) {
				double tmp = 0;
				for (int x = 0; x < maskRows; x++) {
					for (int y = 0; y < maskCols; y++) {
						if (i + x < numRows && j + y < numCols) {
							tmp += (image[(i + x) * numCols + j + y] == mask[x * maskCols + y]);
						}
					}

					if (tmp > score) {
						score = tmp;
						pos = cv::Point(j + maskCols / 2, i + maskRows / 2); // Pos should be the exact center of where the match takes place
					}

				}
			}
		}
	}
	score /= (maskRows * maskCols);
}

void _match_all(unsigned char* image, vector<cv::Point> &symbol, unsigned char *mask, vector<cv::Point> &pos, int numRows, int numCols, int maskRows, int maskCols, double confidence) {

	cv::Point p1 = symbol[0];
	cv::Point p2 = symbol[1];

	int i = p1.y;
	while (i < max(p2.y - maskRows, p1.y + 1)) {
		// If we do find a match, then we set a flag, so that we know we should jump by one 
		// whole mask_height in the next iteration of i
		bool flag = 0;
		int j = p1.x;
		while (j < max(p2.x - maskCols, p1.x + 1)) {
			double score = 0;
			for (int x = 0; x < maskRows; x++) {
				for (int y = 0; y < maskCols; y++) {
					if (i + x < numRows && j + y < numCols) {
						score += (image[(i + x) * numCols + j + y] == mask[x * maskCols + y]);
					}
				}
			}
			score /= (maskRows * maskCols);
			// Check if the score is above the confidence to add it to the list of possible
			// locations of the mask.
			if (score >= confidence) {
				pos.push_back(cv::Point(j + maskCols / 2, i + maskRows / 2));
				j += maskCols;
				flag = 1;
			}
			else j++;
		}
		if (flag) i += maskRows;
		else i++;
	}
}

void match_symbol(unsigned char* image, vector<cv::Point> &symbol, map<string, cv::Mat> &dictionary, vector<pair<string, cv::Point>> &res, int staff_thickness, int staff_spacing, int numRows, int numCols) {
	double filled_confidence = 0.8;
	double empty_confidence = 0.7;
	double symbol_confidence = 0.6;

	vector<pair<double, int> > scores;
	vector<cv::Point> posv;
	vector<string> names;

	cv::Mat mask;
	double score; cv::Point pos;

	int index = 0;
	for (auto &k : dictionary) {
		string name = k.first;
		if (name == "filled_note" || name == "empty_note") continue;
		mask = k.second;

		_match_and_slide(image, symbol, mask.data, numRows, numCols, mask.rows, mask.cols, score, pos, true);
		scores.push_back(make_pair(score, index));
		posv.push_back(pos);
		names.push_back(name);
		index++;
	}
	sort(scores.begin(), scores.end());

	if (scores.back().first >= symbol_confidence) {
		int idx = scores.back().second;
		res.push_back(make_pair(names[idx], posv[idx]));
		return;
	}

	mask = dictionary["empty_note"];
	_match_and_slide(image, symbol, mask.data, numRows, numCols, mask.rows, mask.cols, score, pos, false);

	if (score >= empty_confidence) {
		res.push_back(make_pair("half_note", pos));
		return;
	}

	posv.clear();
	mask = dictionary["filled_note"];
	_match_all(image, symbol, mask.data, posv, numRows, numCols, mask.rows, mask.cols, filled_confidence);

	if (posv.size() == 1) {
		string name = "quater_note";
		if ((1.0 * mask.cols) / (symbol[1].x - symbol[0].x) < 0.7) name = "eighth_note";
		res.push_back(make_pair(name, pos));
	}
	else {
		for (auto pos : posv) {
			res.push_back(make_pair("eigthth_note", pos));
		}
	}
}

int full_step(int n) {
	int note = (n - 1) % 12;
	if (note == 2 || note == 7) return n + 1;
	else return n + 2;
}

int advance(int n, int n_step) {
	int i = 0;
	while (i < n_step) {
		n = full_step(n);
		i += 1;
	}
	return n;
}

int classify_note(cv::Point note, vector<int> &staff, int staff_thickness, int staff_spacing) {
	int note_increment = (staff_thickness + staff_spacing) / 2;
	int n = 44;
	int delta_n = int(round(((staff.back() + staff_thickness / 2) - note.y) / note_increment));
	return advance(n, delta_n);
}


// ********************** CUDA KERNELS *********************** //

_global_ void to_grayscale_and_threshold_gpu(unsigned char* Pout, unsigned char* Pin, int width, int height, int numChannels)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height) {
		int grayOffset = y * width + x;
		int rgbOffset = grayOffset * numChannels;
		unsigned char r = Pin[rgbOffset];
		unsigned char g = Pin[rgbOffset + 1];
		unsigned char b = Pin[rgbOffset + 2];
		Pout[grayOffset] = 0.299 * r + 0.587 * g + 0.114 * b;
	}
}


_global_ void match_and_slide(unsigned char* image, int numRows, int numCols, int* symbol, unsigned char* mask, int mask_height, int mask_width, unsigned char* output, bool bound) {

	// output is of the size of th bounding box
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x1 = symbol[0], y1 = symbol[1], x2 = symbol[2], y2 = symbol[3];
	int score = 0;
	int pos[2] = { -1,-2 };
	double rx = (x2 - x1 + 0.0) / mask_width, ry = (y2 - y1 + 0.0) / mask_height;
	double min_ratio = 0.8, max_ratio = 1.2;

	if (!bound || (bound && rx >= min_ratio && rx <= max_ratio && ry >= min_ratio && ry <= max_ratio)) {
		int col = x + x1;
		int row = y + y1;
		int colbound, rowbound;
		if (y2 - mask_height >= y1 + 1)rowbound = y2 - mask_height;
		else rowbound = y1 + 1;
		if (x2 - mask_width >= x1 + 1)colbound = x2 - mask_width;
		else colbound = x1 + 1;
		if (col <= colbound && row <= rowbound) {
			int tmp = 0;
			for (int x3 = 0; x3 < mask_width; x3++) {
				for (int y3 = 0; y3 < mask_height; y3++) {
					if (row + y3 < numRows && col + x3 <= numCols)
						tmp += image[(row + y3) * numCols + col + x3] == mask[y3 * mask_width + x3];
				}
			}
			output[y * (y2 - y1 + 1) + x] = tmp;
		}
	}
}

_global_ void match_all(unsigned char* image, int numRows, int numCols, int* symbol, unsigned char* mask, int mask_height, int mask_width, int* output) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x1 = symbol[0], y1 = symbol[1], x2 = symbol[2], y2 = symbol[3];
	int col = x + x1;
	int row = y + y1;
	int colbound, rowbound;
	if (y2 - mask_height >= y1 + 1)rowbound = y2 - mask_height;
	else rowbound = y1 + 1;
	if (x2 - mask_width >= x1 + 1)colbound = x2 - mask_width;
	else colbound = x1 + 1;
	if (col < colbound && row < rowbound) {
		int score = 0;
		for (int x3 = 0; x3 < mask_width; x3++) {
			for (int y3 = 0; y3 < mask_height; y3++) {
				if (row + y3 < numRows && col + x3 <= numCols)
					score += image[(row + y3) * numCols + col + x3] == mask[y3 * mask_width + x3];
			}
		}
		output[y * (y2 - y1 + 1) + x] = score;
	}
}
// ************************ MAIN ********************** //

vector<vector<int>> outputToVector(int* image, int numRows, int numCols) {
	vector<vector<int>>res(numRows, vector<int>(numCols));
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			res[i][j] = image[i * numCols + j];
		}
	}
	return res;
}

vector<vector<int>> callMatchAllKernel(unsigned char* d_image, int numRows, int numCols, vector<cv::Point>symbol, unsigned char* mask, int mask_height, int mask_width) {
	unsigned char* d_mask;
	cudaMalloc((void**)&d_mask, sizeof(unsigned char) * mask_width * mask_height);
	cudaMemcpy(d_mask, mask, sizeof(unsigned char) * mask_height * mask_width, cudaMemcpyHostToDevice);
	int output_width = max(symbol[0].x + 1, symbol[1].x - mask_width);
	int output_height = max(symbol[0].y + 1, symbol[1].y - mask_height);
	int* symb = new int[4]();
	symb[0] = symbol[0].x;
	symb[1] = symbol[0].y;
	symb[2] = symbol[1].x;
	symb[3] = symbol[1].y;
	int* d_symb;
	cudaMalloc((void**)&d_symb, sizeof(int) * 4);
	cudaMemcpy(d_symb, symb, sizeof(int) * 4, cudaMemcpyHostToDevice);
	dim3 dimBlock(16, 16, 1);
	dim3 DimGrid((output_width - 1) / 16 + 1, (output_height - 1) / 16 + 1, 1);
	int* d_output;
	cudaMalloc((void**)&d_output, sizeof(int) * output_width * output_height);
	int* output = (int*)malloc(sizeof(int) * output_width * output_height);
	match_all << <DimGrid, dimBlock >> > (d_image, numRows, numCols, d_symb, d_mask, mask_height, mask_width, d_output);
	cudaMemcpy(output, d_output, sizeof(int) * output_width * output_height, cudaMemcpyDeviceToHost);
	cudaFree(d_symb);
	cudaFree(d_mask);
	cudaFree(d_output);
	return outputToVector(output, output_height, output_width);
}


int main()
{
	string filename = "src/*.png";
	//string filename = "src/bar_keysig.png";
	vector<cv::String> fn;
	cv::glob(filename, fn, false);

	fstream in("templates/conf.txt");
	char c; in >> c;
	while (c != '=') {
		in >> c;
	}
	int target_staff_spacing;
	in >> target_staff_spacing;

	map<string, cv::Mat> dictionary = load_dictionary();

	for (size_t i = 0; i < fn.size(); i++) {
		cout << fn[i] << endl;
		cv::Mat raw_image = cv::imread(fn[i], cv::IMREAD_COLOR);

		int image_height = 400;
		int image_width = round((((float)image_height) / raw_image.rows)  * raw_image.cols);

		cv::Mat resized_image(image_height, image_width, CV_8UC3);
		cv::resize(raw_image, resized_image, resized_image.size(), 0, 0, cv::INTER_AREA);

		image_height = resized_image.rows;
		image_width = resized_image.cols;
		int npixels = resized_image.rows* resized_image.cols;


		// Allocate the host image vectors
		unsigned char *input_image = (unsigned char *)malloc(3 * npixels * sizeof(unsigned char));
		unsigned char *binary_image = (unsigned char *)malloc(npixels * sizeof(unsigned char));
		input_image = resized_image.data;

		// Convert to grayscale
		to_grayscale_and_threshold(input_image, binary_image, image_height, image_width);

		// Compute the staff parameters
		int staff_thickness, staff_spacing;
		computeStaff(binary_image, staff_thickness, staff_spacing, image_height, image_width);

		// Adjusting image size to match templates.
		double r = (1.0 * target_staff_spacing) / staff_spacing;
		image_height *= r;
		image_width = int((((float)image_height) / resized_image.rows)  * resized_image.cols);

		cv::Mat image(image_height, image_width, CV_8UC3);
		cv::resize(resized_image, image, image.size(), 0, 0, cv::INTER_AREA);
		npixels = image.rows* image.cols;

		input_image = (unsigned char*)malloc(3 * npixels * sizeof(unsigned char));
		binary_image = (unsigned char*)realloc(binary_image, npixels * sizeof(unsigned char));
		input_image = image.data;

		// Start

		// Convert to grayscale and threshold


		cout << 1 << endl;
		GpuTimer timer;
		timer.Start();

		to_grayscale_and_threshold(input_image, binary_image, image_height, image_width);

		timer.Stop();
		cout << "(CPU) Time taken to convert input image to grayscale: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		cout << 2 << endl;
		// Compute staff parameters
		timer.Start();

		computeStaff(binary_image, staff_thickness, staff_spacing, image_height, image_width);

		timer.Stop();
		cout << "(CPU) Time taken to compute staff parameters: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		vector<vector<int> > staves;

		cout << 3 << endl;
		// Locate staves
		timer.Start();

		find_staves(binary_image, staves, staff_thickness, staff_spacing, image_height, image_width);

		timer.Stop();
		cout << "(CPU) Time taken to locate staves in image: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;


		cout << 4 << endl;
		// Remove staves
		timer.Start();
		for (auto &staff : staves) {
			//             draw_staff(image, staff, staff_thickness);
			remove_staff(binary_image, staff, staff_thickness, staff_spacing, image_height, image_width);
		}
		timer.Stop();
		cout << "(CPU) Time taken to remove all staves in image: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		vector<vector<int>> track_bounds = segment_by_staves(image_height, staves, staff_thickness, staff_spacing);

		vector<vector<cv::Point>> symbols;
		vector<int> track_number;

		timer.Start();

		for (int i = 0; i < staves.size(); i++) {
			vector<int> &track_bound = track_bounds[i];
			int x = symbols.size();
			find_all_symbols(binary_image, symbols, track_bound[0], track_bound[1], staff_thickness, image_height, image_width);
			int y = symbols.size();
			for (int j = x; j < y; j++) {
				track_number.push_back(i);
			}
		}

		timer.Stop();
		cout << "(CPU) Time taken to bound all symbols with boxes: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		for (int i = 0; i < symbols.size(); i++) {
			cv::Point p1 = symbols[i][0];
			cv::Point p2 = symbols[i][1];
			cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);
		}

		vector<pair<string, cv::Point> > res;
		vector<vector<string>> char_sequence(staves.size());

		timer.Start();

		for (int i = 0; i < symbols.size(); i++) {
			int x = res.size();
			match_symbol(binary_image, symbols[i], dictionary, res, staff_thickness, staff_spacing, image_height, image_width);
			int y = res.size();
			for (int j = x; j < y; j++) {
				string name = res[j].first;
				if (name.find("note") != string::npos) {
					name = name + '.' + to_string(classify_note(res[j].second, staves[track_number[i]], staff_thickness, staff_spacing));
				}
				char_sequence[track_number[i]].push_back(name);
			}
		}

		timer.Stop();
		cout << "(CPU) Time taken to classify all symbols: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		for (int i = 0; i < res.size(); i++) {
			cv::Point p1 = res[i].second;
			cv::putText(image, res[i].first, p1, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 0, 255));
		}

		ofstream of("out.txt");
		for (int t = 0; t < staves.size(); t++) {
			of << "Track" << t << ": ";
			for (int i = 0; i < char_sequence[t].size(); i++) {
				of << char_sequence[t][i] << " ";
			}of << endl;
		}

		scale(binary_image, 255, image_height, image_width);

		cv::Mat result(image_height, image_width, CV_8UC1, binary_image);
		cv::imshow("Original", image);
		cv::imshow("Result", result);
		cv::waitKey(0);

		free(binary_image);
	}
	return 0;

}