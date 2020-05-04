#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#define THRESHOLD_VALUE 225

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
	if(endx == -1) endx = numCols - 1;
    if(endy == -1) endy = numRows - 1;
    
    vector<int> result;
    if(axis == 'X'){
        result.resize(endx - startx + 1, 0);
    }else{
        result.resize(endy - starty + 1, 0);
    }
	if (axis == 'X'){
		for (int j = startx; j <= endx; j++){
			for (int i = starty; i < endy; i++){
				if (image[i * numCols + j] == 0){
					result[j - startx]++;
				}
			}
		}
	}
	else{
		for (int i = starty; i <= endy; i++){
			for (int j = startx; j < endx; j++){
				if (image[i * numCols + j] == 0){
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
    for(int i = 1; i < numRows; i++){
        score[i] = score[i] + score[i-1];
    }

	double confidence = 0.8;
	int threshold = numCols * staff_thickness;
	int row = 1;
	while (row < numRows-staff_thickness){
		if (score[row+staff_thickness] - score[row-1] > threshold * confidence){
            staff.push_back(row + staff_thickness / 2);
            row += staff_spacing - 2;
        }else{
            row++;
        }
		if (staff.size() == 5){
			staves.push_back(staff);
			staff.clear();
			confidence = 0.8;
		}
	}
}

vector<vector<int>> segment_by_staves(int numRows, vector<vector<int>> staves, int staff_thickness, int staff_spacing)
{
	vector<vector<int>> track_bounds;
	for (int i = 0; i < staves.size(); i++)
	{
		int y1 = max(staves[i][0] - 2 * (staff_thickness + staff_spacing), 0);
		int y2 = min(staves[i][(int)staves[i].size() - 1] + 2 * (staff_thickness + staff_spacing), numRows);
		track_bounds.push_back({ y1,y2 });
	}
	return track_bounds;
}

vector<vector<int>> get_interesting_intervals(vector<int> proj, int threshold)
{
	vector<vector<int>> boundaries;
	int max1 = *max_element(proj.begin(), proj.end());
	for (int i = 0; i < proj.size(); i++)
	{
		if (proj[i] < threshold || proj[i] >= max1 - 1) i++;
		else
		{
			vector<int> boundary = { i };
			while (i < proj.size() && proj[i] >= threshold)
			{
				i++;
			}
			boundary[0] += i;
			boundaries.push_back(boundary);
		}
	}
	return boundaries;
}

void find_all_symbols(unsigned char* image, int staff_thickness, int staff_spacing) {
	vector<int>xproj = getprojection(image, 0, -1, 'X');
	vector<int>yproj = getprojection(image, 0, -1, 'Y');
	auto vertical_boundaries = get_interesting_intervals(xproj, staff_thickness);
	for (auto vboundary : vertical_boundaries) {
		//tricky !!!
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
				dst[i*numCols+j] = currentSequence;
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
				dst[i*numCols+j] = currentSequence;
			}
		}

	}
}

void remove_staff(unsigned char *image, vector<int> &staff, int staff_thickness, int numRows, int numCols) {
    int* Iv = (int*) malloc(sizeof(int) * numRows * numCols);
	computeRuns(image, Iv, 'Y', numRows, numCols);
	
	for (int i = 0; i < staff.size(); i++) {
		int x = staff[i] + 1;
		for (int j = 0; j < numCols; j++) {
			if (Iv[x * numCols + j] == 0) continue;
			
            int x2 = x;
			while (x2 < numRows && Iv[x2 * numCols + j]>0) x2++;
			if (Iv[ (x2 - 1) * numCols +  j] <= staff_thickness + 3) {
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

vector<vector<unsigned char>> transformBinarImageToVector(unsigned char* image, int numRows, int numCols) {
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

void to_grayscale_and_threshold(unsigned char *src, unsigned char *dst, int height, int width){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            int gray_offset = i * width + j;
            int k = 3 * gray_offset;
            dst[gray_offset] = 0.299 * src[k] + 0.587 * src[k+1] + 0.114 * src[k+2];
            if(dst[gray_offset] >= THRESHOLD_VALUE) dst[gray_offset] = 1;
            else dst[gray_offset] = 0;
        }
    }
}

void scale(unsigned char *src, int k, int height, int width){
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            int idx = i * width + j;
            src[idx] *= k;
        }
    }
}

void draw_staff(cv::Mat img, vector<int> &staff,  int thickness=1){    
    for(int y : staff){
        cv::line(img, cv::Point(0, y), cv::Point(img.cols-1, y), cv::Scalar(0,0,255), thickness);
    }

}
// ***************************************************************** CUDA KERNELS ****************************************************************** //

// __global__ void colorToGrayscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height, int numChannels)
// {
// 	int x = threadIdx.x + blockIdx.x * blockDim.x;
// 	int y = threadIdx.y + blockIdx.y * blockDim.y;
// 	if (x < width && y < height) {
// 		int grayOffset = y * width + x;
// 		int rgbOffset = grayOffset * numChannels;
// 		unsigned char r = Pin[rgbOffset];
// 		unsigned char g = Pin[rgbOffset + 1];
// 		unsigned char b = Pin[rgbOffset + 2];
// 		Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
// 	}
// }

// 
// __global__ void match_and_slide(unsigned char* image, int numRows, int numCols, int* symbol, unsigned char* mask, int mask_height, int mask_width, unsigned char* output, bool bound) {
// 
// 	output is of the size of th bounding box
// 	int x = threadIdx.x + blockIdx.x * blockDim.x;
// 	int y = threadIdx.y + blockIdx.y * blockDim.y;
// 	int x1 = symbol[0], y1 = symbol[1], x2 = symbol[2], y2 = symbol[3];
// 	int score = 0;
// 	int pos[2] = { -1,-2 };
// 	double rx = (x2 - x1 + 0.0) / mask_width, ry = (y2 - y1 + 0.0) / mask_height;
// 	double min_ratio = 0.8, max_ratio = 1.2;
// 
// 	if (!bound || (bound && rx >= min_ratio && rx <= max_ratio && ry >= min_ratio && ry <= max_ratio)) {
// 		int col = x + x1;
// 		int row = y + y1;
// 		int colbound, rowbound;
// 		if (y2 - mask_height >= y1 + 1)rowbound = y2 - mask_height;
// 		else rowbound = y1 + 1;
// 		if (x2 - mask_width >= x1 + 1)colbound = x2 - mask_width;
// 		else colbound = x1 + 1;
// 		if (col <= colbound && row <= rowbound) {
// 			int tmp = 0;
// 			for (int x3 = 0; x3 < mask_width; x3++) {
// 				for (int y3 = 0; y3 < mask_hight; y3++) {
// 					if (row + y3 < numRows && col + x3 <= numCols)
// 						tmp += image[(row + y3) * numCols + col + x3] == mask[y3 * mask_width + x3];
// 				}
// 			}
// 			output[y * (y2 - y1 + 1) + x] = tmp;
// 		}
// 	}
// }
// 
// __global__ void match_all(unsigned char* image, int numRows, int numCols, int* symbol, unsigned char* mask, int mask_height, int mask_width, unsigned char* output) {
// 
// 	int x = threadIdx.x + blockIdx.x * blockDim.x;
// 	int y = threadIdx.y + blockIdx.y * blockDim.y;
// 	int x1 = symbol[0], y1 = symbol[1], x2 = symbol[2], y2 = symbol[3];
// 	int col = x + x1;
// 	int row = y + y1;
// 	int colbound, rowbound;
// 	if (y2 - mask_height >= y1 + 1)rowbound = y2 - mask_height;
// 	else rowbound = y1 + 1;
// 	if (x2 - mask_width >= x1 + 1)colbound = x2 - mask_width;
// 	else colbound = x1 + 1;
// 	if (col < colbound && row < rowbound) {
// 		int score = 0;
// 		for (int x3 = 0; x3 < mask_width; x3++) {
// 			for (int y3 = 0; y3 < mask_hight; y3++) {
// 				if (row + y3 < numRows && col + x3 <= numCols)
// 					score += image[(row + y3) * numCols + col + x3] == mask[y3 * mask_width + x3];
// 			}
// 		}
// 		output[y * (y2 - y1 + 1) + x] = tmp;
// 	}
// }

// ********************************************************************** MAIN ****************************************************************** //


int main()
{
    string filename = "src/*.png";
    vector<cv::String> fn;
    cv::glob(filename, fn, false);
    
    
    ifstream in("templates/conf.txt");
    char c; in >> c;
    while(c != '='){
        in >> c;
    }
    int target_staff_spacing;
    in >> target_staff_spacing;
    
    for (size_t i=0; i<fn.size(); i++){
        cout << fn[i] << endl;
        cv::Mat raw_image = cv::imread(fn[i], cv::IMREAD_COLOR);
        
        int image_height = 400;
        int image_width = round( ( ((float) image_height ) /raw_image.rows)  * raw_image.cols  );
        
        cv::Mat resized_image(image_height, image_width, CV_8UC3);
        cv::resize(raw_image, resized_image, resized_image.size(), 0, 0, cv::INTER_AREA);
        
        image_height = resized_image.rows;
        image_width = resized_image.cols;
        int npixels = resized_image.rows* resized_image.cols;
        
        
        // Allocate the host image vectors
        unsigned char *input_image = (unsigned char *) malloc(3 * npixels * sizeof(unsigned char));
        unsigned char *binary_image = (unsigned char *) malloc(npixels * sizeof(unsigned char));
        input_image = resized_image.data;
        
        // Convert to grayscale
        to_grayscale_and_threshold(input_image, binary_image, image_height, image_width);
        
        // Compute the staff parameters
        int staff_thickness, staff_spacing;
        computeStaff(binary_image, staff_thickness, staff_spacing, image_height, image_width);
        
        // Adjusting image size to match templates.
        double r = (1.0 * target_staff_spacing) / staff_spacing;
        image_height  *= r;
        image_width = round( ( ((float) image_height ) /resized_image.rows)  * resized_image.cols  );
        
        cv::Mat image(image_height, image_width, CV_8UC3);
        cv::resize(resized_image, image, image.size(), 0, 0, cv::INTER_AREA);
        npixels = image.rows* image.cols;
        
        input_image = (unsigned char*) malloc(3 * npixels * sizeof(unsigned char));
        binary_image = (unsigned char*) realloc(binary_image, npixels * sizeof(unsigned char));
        input_image = image.data;
        
        // Start
        
        cout << 1 << endl;
        
        // Convert to grayscale and threshold
        to_grayscale_and_threshold(input_image, binary_image, image_height, image_width);
        
        
        cout << 2 << endl;
        // Compute staff parameters
        computeStaff(binary_image, staff_thickness, staff_spacing, image_height, image_width);        
        
        
        vector<vector<int> > staves;
        
        cout << 3 << endl;
        // Locate staves
        find_staves(binary_image, staves, staff_thickness, staff_spacing, image_height, image_width);
        
        
        cout << 4 << endl;
        // Remove staves
        for(auto &staff : staves){
            draw_staff(image, staff, staff_thickness);
            remove_staff(binary_image, staff, staff_thickness, image_height, image_width);
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
