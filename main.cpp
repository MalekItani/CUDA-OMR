#include <iostream>
#include <set>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <unordered_map>
#include <stack>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <list>
#include <map>
#include <bitset>
#include <queue>
#include <functional>  
#include <numeric>      
#include <assert.h>
#include <unordered_set>
#include <array>
#include <stdio.h>
#include<complex>
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

pair<int, int> computeStaff(vector<vector<unsigned char>>& image)
{
	int numRows = image.size(), numCols = 0;
	if (numRows > 0)numCols = image[0].size();
	vector<int> whitehist(numRows + 1, 0);
	vector<int> blackhist(numRows + 1, 0);
	for (int j = 0; j < numCols; j++)
	{
		for (int i = 0; i < numRows; i++)
		{
			int seqlen = 0;
			while (i < numRows && image[i][j] > 0)
			{
				seqlen++; i++;
			}
			if (seqlen > 0)
			{
				whitehist[seqlen] += 1;
			}
			seqlen = 0;
			while (i < numRows && image[i][j] == 0)
			{
				seqlen++; i++;
			}
			if (seqlen > 0)
			{
				blackhist[seqlen] += 1;
			}
		}
	}
	pair<int, int> ans;
	ans.first = sliding_window_argmax(blackhist, 1) - 1;
	ans.second = sliding_window_argmax(whitehist, 1) + 1;
	return ans;
}

vector<vector<int>> find_staves(vector<vector<unsigned char>>& image, int staff_thickness, int staff_spacing)
{
	int numRows = image.size(), numCols = 0;
	if (numRows > 0)numCols = image[0].size();
	vector<int> staves;
	vector<vector<int>> staves_pos;
	vector<int> score(numRows, 0);
	for (int j = 0; j < numCols; j++)
	{
		for (int i = 0; i < numRows - staff_thickness; i++)
		{
			for (int k = i; k < i + staff_spacing; k++)
			{
				score[i] += 1 - image[k][j];
			}
		}
	}
	double confidence = 0.8;
	int threshold = numCols * staff_thickness;
	int row = 0;
	while (row < numRows)
	{
		if (score[row] > threshold * confidence)
			staves.push_back(row + staff_thickness / 2);
		if (staves.size() == 5)
		{
			staves_pos.push_back(staves);
			staves.clear();
			confidence = 0.8;
		}
		row += staff_spacing - 2;
		if (confidence == 0.8)
			confidence = 0.6;
	}
	return staves_pos;
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

vector<int> getprojection(vector<vector<unsigned char>>& image, int start = 0, int end = -1, char axis = 'X')
{
	int numRows = image.size(), numCols = 0;
	if (numRows > 0)numCols = image[0].size();
	if (end == -1)
	{
		if (axis == 'X')
		{
			end = numCols;
		}
		else end = numRows;
	}
	vector<int> result(end - start, 0);
	if (axis == 'X')
	{
		for (int j = start; j <= end; j++)
		{
			for (int i; i < numRows; i++)
			{
				if (image[i][j] == 0)
				{
					result[j - start]++;
				}
			}
		}
	}
	else
	{
		for (int i = start; i <= end; i++)
		{
			for (int j; j < numCols; j++)
			{
				if (image[i][j] == 0)
				{
					result[i - start]++;
				}
			}
		}
	}
	return result;
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

void find_all_symbols(vector<vector<unsigned char>>& image, int staff_thickness, int staff_spacing) {
	vector<int>xproj = getprojection(image, 0, -1, 'X');
	vector<int>yproj = getprojection(image, 0, -1, 'Y');
	auto vertical_boundaries = get_interesting_intervals(xproj, staff_thickness);
	for (auto vboundary : vertical_boundaries) {
		//tricky !!!
	}
}

vector<vector<int>> computeRuns(vector<vector<unsigned char>>& image, char axis) {

	int numRows = image.size(), numCols = 0;
	if (numRows > 0)numCols = image[0].size();
	vector<vector<int>>res(numRows, vector<int>(numCols, 0));
	if (axis == 'X') {
		for (int i = 0; i < numRows; i++) {
			int currentSequence = 0;
			for (int j = 0; j < numCols; j++) {
				if (image[i][j] == 0) {
					currentSequence++;
				}
				else {
					currentSequence = 0;
				}
				res[i][j] = currentSequence;
			}
		}
	}
	else {
		for (int j = 0; j < numCols; j++) {
			int currentSequence = 0;
			for (int i = 0; i < numRows; i++) {
				if (image[i][j] == 0) {
					currentSequence++;
				}
				else {
					currentSequence = 0;
				}
				res[i][j] = currentSequence;
			}
		}

	}
	return res;
}

vector<vector<unsigned char>> remove_staff(vector<vector<unsigned char>>& image, vector<int>staff, int staff_thickness) {
	int numRows = image.size(), numCols = 0;
	if (numRows > 0)numCols = image[0].size();
	vector<vector<int>>Iv = computeRuns(image, 'Y');
	vector<vector<unsigned char>>res(numRows, vector<unsigned char>(numCols));
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			res[i][j] = image[i][j];
		}
	}
	for (int i = 0; i < staff.size(); i++) {
		int x = staff[i];
		for (int j = 0; j < numCols; j++) {
			if (Iv[x][j] == 0)continue;
			int x2 = x;
			while (x2 < numRows && Iv[x2][j]>0)x2++;
			if (Iv[x2 - 1][j] <= staff_thickness + 2) {
				int x1 = x2 - Iv[x2 - 1][j];
				while (x1 < x2) {
					res[x1][j] = 1;
					x1++;
				}
			}
		}
	}
	return res;
}

vector<vector<int>> find_vertical_lines(vector<vector<unsigned char >> image, int staff_thickness, int staff_spacing) {
	int numRows = image.size(), numCols = 0;
	if (numRows > 0)numCols = image[0].size();
	vector<vector<int>> Iv = computeRuns(image, 'Y');
	int expected_segment_width = 3 * staff_thickness / 2;
	if (expected_segment_width % 2 == 0)expected_segment_width--;
	vector<double>Nl(expected_segment_width + 4, 0);
	Nl[0] = Nl[1] = Nl[(int)Nl.size() - 1] = Nl[(int)Nl.size() - 2] = 0.25;
	int mask_radius = (int)Nl.size() / 2;
	vector<vector<double>> Il(numRows, vector<double>(numCols, 0));
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			if (image[i][j] == 0) {
				double pixval = 0;
				for (int k = -mask_radius; k <= mask_radius + 1; k++) {
					if (j + k >= 0 && j + k < numCols) {
						pixval += image[i][j + k] * Nl[mask_radius + k];
					}
				}
				Il[i][j] = pixval;
			}
		}
	}
	vector<vector<int>> potential_vertical_lines;
	for (int j = 0; j < numCols; j++) {
		int largest_run = 0;
		int xh = 0, xb = 0;
		for (int i = 0; i < numRows; i++) {
			if (Iv[i][j] > largest_run) {
				largest_run = Iv[i][j];
				xh = i - largest_run, xb = i;
			}
		}
		if (largest_run > 2 * staff_spacing) {
			potential_vertical_lines.push_back({ j,xh,xb });
		}
	}
	vector<vector<int>> vertical_lines;
	for (int i = 0; i < potential_vertical_lines.size(); i++) {
		vector<int>line = potential_vertical_lines[i];
		if (vertical_lines.size() == 0 || line[0] - vertical_lines[(int)vertical_lines.size() - 1][0] > 2 / 5 * staff_spacing)
		{
			vertical_lines.push_back(line);
		}
	}
	return vertical_lines;
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

//cuda kernels:
__global__ void colorToGrayscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height, int numChannels)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height) {
		int grayOffset = y * width + x;
		int rgbOffset = grayOffset * numChannels;
		unsigned char r = Pin[rgbOffset];
		unsigned char g = Pin[rgbOffset + 1];
		unsigned char b = Pin[rgbOffset + 2];
		Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
	}
}


__global__ void match_and_slide(unsigned char* image, int numRows, int numCols, int* symbol, unsigned char* mask, int mask_height, int mask_width, unsigned char* output, bool bound) {

	//output is of the size of th bounding box
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
				for (int y3 = 0; y3 < mask_hight; y3++) {
					if (row + y3 < numRows && col + x3 <= numCols)
						tmp += image[(row + y3) * numCols + col + x3] == mask[y3 * mask_width + x3];
				}
			}
			output[y * (y2 - y1 + 1) + x] = tmp;
		}
	}
}

__global__ void match_all(unsigned char* image, int numRows, int numCols, int* symbol, unsigned char* mask, int mask_height, int mask_width, unsigned char* output) {

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
			for (int y3 = 0; y3 < mask_hight; y3++) {
				if (row + y3 < numRows && col + x3 <= numCols)
					score += image[(row + y3) * numCols + col + x3] == mask[y3 * mask_width + x3];
			}
		}
		output[y * (y2 - y1 + 1) + x] = tmp;
	}
}

int main()
{
	
}