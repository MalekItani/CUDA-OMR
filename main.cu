#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <chrono>
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

_global_ void colorToGrayscaleandThresoldKernel(unsigned char* Pout, unsigned char* Pin, int width, int height, int numChannels)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height) {
		int grayOffset = y * width + x;
		int rgbOffset = grayOffset * numChannels;
		unsigned char r = Pin[rgbOffset];
		unsigned char g = Pin[rgbOffset + 1];
		unsigned char b = Pin[rgbOffset + 2];
		Pout[grayOffset] = 0.299f * r + 0.587f * g + 0.114f * b;
		if (Pout[grayOffset] >= THRESHOLD_VALUE) Pout[grayOffset] = 1;
		else Pout[grayOffset] = 0;
	}
}


_global_ void match_all(unsigned char* image, int numRows, int numCols, int* symbol, unsigned char* mask, int mask_height, int mask_width, int* output, int output_width, int output_height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x1 = symbol[0], y1 = symbol[1];
	int col = x + x1;
	int row = y + y1;
	if (x < output_width && y < output_height) {
		int score = 0;
		for (int x3 = 0; x3 < mask_width; x3++) {
			for (int y3 = 0; y3 < mask_height; y3++) {
				if (row + y3 < numRows && col + x3 <= numCols)
					score += image[(row + y3) * numCols + col + x3] == mask[y3 * mask_width + x3];
			}
		}
		output[y * (output_width)+x] = score;
	}
}


int sliding_window_argmax(vector<int>& a, int k)
{
	/*
    Given a 1D array a, computes the index of maximum value of the sum of elements in
    a window of size k this array.
    This is done by first computing the partial sum of a, then taking the maximum value
    or partial_sum_a[i+k] - partial_sum_a[i-1];
    */
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
	/*
    Returns the projection of an image (count of black pixels) from start to end along a specified axis.
    We can also compute the projection for a specific subrectangle of the image by specifying startx, stary, endx, endy.
    */
   	// If the values for end or endy are not set, then we assume them to be the last column or last row in the image.
	if (endx == -1 || endx >= numCols) endx = numCols - 1;
	if (endy == -1 || endy >= numRows) endy = numRows - 1;

	// Loop over the matrix, and if the projection is along X, then:
    // result[i] = number of black pixels in (startx + i)-th column
    // Loop over the matrix, and if the projection is along Y, then:
    // result[i] = number of black pixels in (starty + i)-th row
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

void computeStaff(unsigned char* image, int& staff_thickness, int& staff_spacing, int numRows, int numCols)
{
	/*
    Extract staff parameters from input image.
    Staff parameters are staff width and staff spacing.
    This is done by creating a histogram of consecutive black an white pixels, and 
    taking the lengths of black pixels which occur most as staff thickness, and lengths
    of white pixels which occur most as staff spacing.
    This algorithm is described in Optical Music Recognition using Projections.
    */
    // Initialize histograms
	vector<int> whitehist(numRows + 1, 0);
	vector<int> blackhist(numRows + 1, 0);
	// Loop over columns
	for (int j = 0; j < numCols; j++)
	{
		for (int i = 0; i < numRows; i++)
		{
			// Compute length of consecutive sequence of white pixels
			int seqlen = 0;
			while (i < numRows && image[i * numCols + j] > 0)
			{
				seqlen++; i++;
			}
			if (seqlen > 0)
			{
				whitehist[seqlen] += 1;
			}
			// Compute length of consecutive sequence of black pixels
			seqlen = 0;
			while (i < numRows && image[i * numCols + j] == 0)
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

void find_staves(unsigned char* image, vector<vector<int> >& staves, int staff_thickness, int staff_spacing, int numRows, int numCols)
{
	/*
    Given a binary image I, locates the staves in the image.
    A staff is a list of 5 staff y-positions.
    */
	vector<int> staff;

	// Define the score[i] of a row i as the number of black pixels along all columns of all rows
    // that are at most staff_thickness below row i. In other words, score[i] = number of black pixels
    // in a submatrix whose top left corner is (0, i) and whose bottom right corner is (numRows-1, i+staff_thickness)
    
    // To get this array, we can simply take the partial sum of the Y projection, and then take
    // score[i] = yproj[i+staff_thickness] - yproj[i-1]
    // We don't actually compute this explicitly; computing yproj is enough, but when we use the
    // score, we use it based on the above formula.

	vector<int> score = getprojection(image, numRows, numCols, 'Y');
	// Compute the partial sum (in place).
	for (int i = 1; i < numRows; i++) {
		score[i] = score[i] + score[i - 1];
	}

	// If a row has a staff line, then most of the pixels in the submatrix described above should be black.
    // We put a cutoff at 80% (experimentally determined).
    // One we find a staff line, we can jump by staff_spacing/2 to avoid duplicate detections.
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
		// Once 5 staff lines are found, then they form a staff.
        // So add these to the result and reset.
		if (staff.size() == 5) {
			staves.push_back(staff);
			staff.clear();
			confidence = 0.8;
		}
	}
}

vector<vector<int>> segment_by_staves(int numRows, vector<vector<int>>& staves, int staff_thickness, int staff_spacing)
{
	/*
    Splits tracks by the staff positions.
    Returns list of tracks, list of track y-offsets
    */
	vector<vector<int>> track_bounds;
	for (int i = 0; i < staves.size(); i++)
	{
		// Consider three imaginary staff lines above and below to account for notes
        // above and below the staff.
		int y1 = max(staves[i][0] - 3 * (staff_thickness + staff_spacing), 0);
		int y2 = min(staves[i][(int)staves[i].size() - 1] + 3 * (staff_thickness + staff_spacing), numRows);
		track_bounds.push_back({ y1,y2 });
	}
	return track_bounds;
}

vector<vector<int>> get_interesting_intervals(vector<int>& proj, int threshold)
{
	/*
    Returns a list of intervals where proj is greater than some threshold.
    */
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

void find_all_symbols(unsigned char* image, vector<vector<cv::Point>>& symbols, int starty, int endy, int staff_thickness, int numRows, int numCols) {
	/*
    Returns bounding boxes over all symbols in the image.
    Note that the coordinates are relative so be sure to add the y-offset for multitrack images.
    */
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

void computeRuns(unsigned char* image, int* dst, char axis, int numRows, int numCols) {
	/*
    Given an input binary image I, computes an output image res, where
    res[i, j] = longest run ending at this pixel.
    A run is defined as a consecutive sequence of black pixels.
    If I[i, j] = 1, i.e. represents a white pixel, then res[i, j] = 0
    */
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
				dst[i * numCols + j] = currentSequence;
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
				dst[i * numCols + j] = currentSequence;
			}
		}

	}
}

void remove_staff(unsigned char *image, vector<int> &staff, int staff_thickness, int staff_spacing,  int numRows, int numCols) {
	// For each staff line in a staff, given by its y position, remove the longest consecutive
    // sequence of black pixels, if this sequence has a length below staff_thickness + 3.
    // This is so that symbols that are on the staff line don't get chopped.
    // In order to handle notes outside the staff, consider 3 additional imaginary staff lines
    // above and below the staff.
    int* Iv = (int*) malloc(sizeof(int) * numRows * numCols);
	computeRuns(image, Iv, 'Y', numRows, numCols);
	
	// Imaginary staff
    vector<int> istaff;
    
	// Add three lines above
    for(int i = 1; i <= 3; i++){
        if(staff[0] - i*(staff_spacing + staff_thickness) >= 0) istaff.push_back(staff[0] - i * (staff_spacing + staff_thickness) );
    }
    
	// Add the staff lines
    for(int i = 0; i < staff.size(); i++) istaff.push_back(staff[i]);
    
	// Add three lines below
    for(int i = 1; i <= 3; i++) if(staff.back() + i*(staff_spacing +staff_thickness) < numRows) istaff.push_back(staff.back() + i * (staff_spacing + staff_thickness) )  ;
    
	// For each line in the imaginary staff, remove the run if it satisfies the above conditions.
	for (int i = 0; i < istaff.size(); i++) {
		int x = istaff[i] + 1;
        bool flag = (i < 3 || i > 7);
		for (int j = 0; j < numCols; j++) {
			int x2 = x;
            while(x2 < numRows && Iv[x2 * numCols + j] == 0 && (x2 - x) < staff_thickness/2) x2++;
			
            if(x2 >= numRows) continue;
            
			while (x2 < numRows && Iv[x2 * numCols + j]>0) x2++;
			if (Iv[ (x2 - 1) * numCols +  j] <= (1 + flag) * staff_thickness + 3) {
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

void scale(unsigned char* src, int k, int height, int width) {
	// Multiplies all elements in the input array by a scalar.
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int idx = i * width + j;
			src[idx] *= k;
		}
	}
}

void draw_staff(cv::Mat img, vector<int>& staff, int thickness = 1) {
	// Draws the staff on the input image.
	for (int y : staff) {
		cv::line(img, cv::Point(0, y), cv::Point(img.cols - 1, y), cv::Scalar(0, 0, 255), thickness);
	}

}


void to_grayscale_and_threshold(unsigned char* src, unsigned char* dst, int height, int width) {
	unsigned char* d_rgbImage, *d_grayImage;
	int size = width * height * sizeof(unsigned char);
	cudaMalloc((void**)&d_rgbImage, size * 3);
	cudaMalloc((void**)&d_grayImage, size);
	cudaMemcpy(d_rgbImage, src, size * 3, cudaMemcpyHostToDevice);
	dim3 dimBlock(16, 16, 1);
	dim3 DimGrid((width - 1) / 16 + 1, (height - 1) / 16 + 1, 1);
	colorToGrayscaleandThresoldKernel << <DimGrid, dimBlock >> > (d_grayImage, d_rgbImage, width, height, 3);
	cudaMemcpy(dst, d_grayImage, size, cudaMemcpyDeviceToHost);
	cudaFree(d_rgbImage);
	cudaFree(d_grayImage);
}

map<string, cv::Mat> load_dictionary() {
	// Reads all templates from templates/
    // and puts them in a dictionary of templates.
	map<string, cv::Mat> dictionary;

	string filename = "templates/*.png";
	vector<cv::String> fn;
	cv::glob(filename, fn, false);

	for (size_t i = 0; i < fn.size(); i++) {
		cv::Mat raw_image = cv::imread(fn[i], cv::IMREAD_COLOR);

		unsigned char* binary_image = (unsigned char*)malloc(sizeof(unsigned char) * raw_image.rows * raw_image.cols);
		to_grayscale_and_threshold(raw_image.data, binary_image, raw_image.rows, raw_image.cols);

		int n = fn[i].size();
		string name = fn[i].substr(10, (n - 14));
		cv::Mat template_image(raw_image.rows, raw_image.cols, CV_8UC1, binary_image);
		dictionary[name] = template_image;
	}

	return dictionary;
}


vector<vector<int>> outputToVector(int* image, int numRows, int numCols) {
	vector<vector<int>>res(numRows, vector<int>(numCols));
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			res[i][j] = image[i * numCols + j];
		}
	}
	return res;
}

vector<vector<int>> callMatchAllKernel(unsigned char* d_image, int numRows, int numCols, vector<cv::Point>& symbol, unsigned char* mask, int mask_height, int mask_width) {
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
	match_all << <DimGrid, dimBlock >> > (d_image, numRows, numCols, d_symb, d_mask, mask_height, mask_width, d_output, output_width, output_height);
	cudaMemcpy(output, d_output, sizeof(int) * output_width * output_height, cudaMemcpyDeviceToHost);
	cudaFree(d_symb);
	cudaFree(d_mask);
	cudaFree(d_output);
	return outputToVector(output, output_height, output_width);
}

void _match_all_optimized(unsigned char* d_image, vector<cv::Point>& symbol, unsigned char* mask, vector<cv::Point>& pos, int numRows, int numCols, int maskRows, int maskCols, double confidence) {

	cv::Point p1 = symbol[0];
	cv::Point p2 = symbol[1];
	auto output = callMatchAllKernel(d_image, numRows, numCols, symbol, mask, maskRows, maskCols);
	int i = p1.y;
	while (i < max(p2.y - maskRows, p1.y + 1)) {
		// If we do find a match, then we set a flag, so that we know we should jump by one 
		// whole mask_height in the next iteration of i
		bool flag = 0;
		int j = p1.x;
		while (j < max(p2.x - maskCols, p1.x + 1)) {
			double score = output[i - p1.y][j - p1.x];
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

void _match_and_slide_optimized(unsigned char* d_image, vector<cv::Point>& symbol, unsigned char* mask, int numRows, int numCols, int maskRows, int maskCols, double& score, cv::Point& pos, bool bound = false) {
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
		auto output = callMatchAllKernel(d_image, numRows, numCols, symbol, mask, maskRows, maskCols);
		// Loop over every pixel in the symbol to choose the top left corner (i, j) and try matching.
		// From experimentation, it helps to consider at the least the first pixel, even if
		// the bounds of the template exceed the bounds of the symbol. Usually, they don't exceed
		// exceed them by too much if there really is a match.
		for (int i = p1.y; i < max(p2.y - maskRows, p1.y + 1); i++) {
			for (int j = p1.x; j < max(p2.x - maskCols, p1.x + 1); j++) {
				double tmp = output[i - p1.y][j - p1.x];
				if (tmp > score) {
					score = tmp;
					pos = cv::Point(j + maskCols / 2, i + maskRows / 2); // Pos should be the exact center of where the match takes place
				}

			}
		}
	}
	score /= (maskRows * maskCols);
}

void match_symbol(unsigned char* d_image, vector<cv::Point>& symbol, map<string, cv::Mat>& dictionary, vector<pair<string, cv::Point>>& res, int staff_thickness, int staff_spacing, int numRows, int numCols) {
	double filled_confidence = 0.8;
	double empty_confidence = 0.7;
	double symbol_confidence = 0.6;

	vector<pair<double, int> > scores;
	vector<cv::Point> posv;
	vector<string> names;

	cv::Mat mask;
	double score; cv::Point pos;

	int index = 0;
	for (auto& k : dictionary) {
		string name = k.first;
		if (name == "filled_note" || name == "empty_note") continue;
		mask = k.second;

		_match_and_slide_optimized(d_image, symbol, mask.data, numRows, numCols, mask.rows, mask.cols, score, pos, true);
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
	_match_and_slide_optimized(d_image, symbol, mask.data, numRows, numCols, mask.rows, mask.cols, score, pos, false);

	if (score >= empty_confidence) {
		res.push_back(make_pair("half_note", pos));
		return;
	}

	posv.clear();
	mask = dictionary["filled_note"];
	_match_all_optimized(d_image, symbol, mask.data, posv, numRows, numCols, mask.rows, mask.cols, filled_confidence);

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
	// Advance a note by one full step according to music theory rules.
	int note = (n - 1) % 12;
	if (note == 2 || note == 7) return n + 1;
	else return n + 2;
}

int advance(int n, int n_step) {
	// Advance a note by n steps according to music theory rules.
	int i = 0;
	while (i < n_step) {
		n = full_step(n);
		i += 1;
	}
	return n;
}

int classify_note(cv::Point note, vector<int>& staff, int staff_thickness, int staff_spacing) {
	// Converts a note from y-position to key index.
	int note_increment = (staff_thickness + staff_spacing) / 2;
	int n = 44;
	int delta_n = int(round(((staff.back() + staff_thickness / 2) - note.y) / note_increment));
	return advance(n, delta_n);
}

int main()
{
	string filename = "src/*.png";
	//string filename = "src/bar_keysig.png";
	vector<cv::String> fn;
	cv::glob(filename, fn, false);

	// Load the configuration file that the templates were generated with, for specifics
    // like required staff spacing in pixels, to know by how much to resize a given image.
	ifstream in("templates/conf.txt");
	char c; in >> c;
	while (c != '=') {
		in >> c;
	}
	int target_staff_spacing;
	in >> target_staff_spacing;

	// Load the dictionary of template masks.
	map<string, cv::Mat> dictionary = load_dictionary();

	// Loop over all images in src/
	for (size_t i = 0; i < fn.size(); i++) {
		cout << fn[i] << endl;

		// Read the image, resize it, compute staff parameters.
        // Adjust the parameters to match the parameters of the image that
        // the templates used in order to be able to use template matching.
		cv::Mat raw_image = cv::imread(fn[i], cv::IMREAD_COLOR);

		int image_height = 400;
		int image_width = round((((float)image_height) / raw_image.rows) * raw_image.cols);

		cv::Mat resized_image(image_height, image_width, CV_8UC3);
		cv::resize(raw_image, resized_image, resized_image.size(), 0, 0, cv::INTER_AREA);

		image_height = resized_image.rows;
		image_width = resized_image.cols;
		int npixels = resized_image.rows * resized_image.cols;


		// Allocate the host image vectors
		unsigned char* input_image = (unsigned char*)malloc(3 * npixels * sizeof(unsigned char));
		unsigned char* binary_image = (unsigned char*)malloc(npixels * sizeof(unsigned char));
		input_image = resized_image.data;

		// Convert to grayscale
		to_grayscale_and_threshold(input_image, binary_image, image_height, image_width);

		// Compute the staff parameters
		int staff_thickness, staff_spacing;
		computeStaff(binary_image, staff_thickness, staff_spacing, image_height, image_width);

		// Adjusting image size to match templates.
		double r = (1.0 * target_staff_spacing) / staff_spacing;
		image_height *= r;
		image_width = int((((float)image_height) / resized_image.rows) * resized_image.cols);

		cv::Mat image(image_height, image_width, CV_8UC3);
		cv::resize(resized_image, image, image.size(), 0, 0, cv::INTER_AREA);
		npixels = image.rows * image.cols;

		input_image = (unsigned char*)malloc(3 * npixels * sizeof(unsigned char));
		binary_image = (unsigned char*)realloc(binary_image, npixels * sizeof(unsigned char));
		input_image = image.data;

		// Start

		// Convert to grayscale and threshold

		GpuTimer timer;
		timer.Start();

		to_grayscale_and_threshold(input_image, binary_image, image_height, image_width);

		timer.Stop();
		cout << "Time taken to convert input image to grayscale: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		// Compute staff parameters
		timer.Start();

		computeStaff(binary_image, staff_thickness, staff_spacing, image_height, image_width);

		timer.Stop();
		cout << "Time taken to compute staff parameters: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		vector<vector<int> > staves;
		// PUT THE BINARY_IMAGE IN GLOBAL MEMORY

		unsigned char * d_binary_image;
		cudaMalloc((void**)&d_binary_image, sizeof(unsigned char) * npixels);
		cudaMemcpy(d_binary_image, binary_image, sizeof(unsigned char) * npixels, cudaMemcpyHostToDevice);


		// Locate staves
		timer.Start();

		find_staves(binary_image, staves, staff_thickness, staff_spacing, image_height, image_width);

		timer.Stop();
		cout << "Time taken to locate staves in image: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		// Remove staves
		timer.Start();

		for (auto& staff : staves) {
			//             draw_staff(image, staff, staff_thickness);
			remove_staff(binary_image, staff, staff_thickness, staff_spacing, image_height, image_width);
		}

		timer.Stop();
		cout << "Time taken to remove all staves in image: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;
		
		// Find the y boundaries of each track from the staves.
		vector<vector<int>> track_bounds = segment_by_staves(image_height, staves, staff_thickness, staff_spacing);

		vector<vector<cv::Point>> symbols;
		vector<int> track_number;

		timer.Start();

		// For each track, find all the symbols on this track.
		for (int i = 0; i < staves.size(); i++) {
			vector<int>& track_bound = track_bounds[i];
			int x = symbols.size();
			find_all_symbols(binary_image, symbols, track_bound[0], track_bound[1], staff_thickness, image_height, image_width);
			int y = symbols.size();
			for (int j = x; j < y; j++) {
				track_number.push_back(i);
			}
		}

		timer.Stop();
		cout << "Time taken to bound all symbols with boxes: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		// Draw bounding rectangles for each symbol.
		for (int i = 0; i < symbols.size(); i++) {
			cv::Point p1 = symbols[i][0];
			cv::Point p2 = symbols[i][1];
			cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 2);
		}
		
		vector<pair<string, cv::Point> > res;
		vector<vector<string>> char_sequence(staves.size());

		timer.Start();
		// Recognize each symbol, and encode it correspondingly in the output file.
        // If the symbol is not a note, then output its name directly
        // Otherwise, find the note's key index n, and then output it in the form
        // "{note_type}.{n}"
		for (int i = 0; i < symbols.size(); i++) {
			int x = res.size();
			match_symbol(d_binary_image, symbols[i], dictionary, res, staff_thickness, staff_spacing, image_height, image_width);
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
		cout << "Time taken to classify all symbols: " << setprecision(4) << fixed << timer.Elapsed() << "ms." << endl;

		// Display symbol names on image.
		for (int i = 0; i < res.size(); i++) {
			cv::Point p1 = res[i].second;
			cv::putText(image, res[i].first, p1, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 0, 255));
		}

		// Write to output file and display images.
		ofstream of("GPUout.txt"); // Make sure to verify output with out.txt
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
		cudaFree(d_binary_image);
	}
	return 0;

}