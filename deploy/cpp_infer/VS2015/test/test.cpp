// test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Windows.h"
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <io.h>
using namespace std;
using namespace cv;

HINSTANCE OcrDetectorDll;
void VisualizeBboxes(const cv::Mat &srcimg,  
	const std::vector<std::vector<std::vector<int>>> &boxes,
	std::string strFileName) {
	cv::Mat img_vis;
	srcimg.copyTo(img_vis);
	for (int n = 0; n < boxes.size(); n++) {
		cv::Point rook_points[4];
		for (int m = 0; m < boxes[n].size(); m++) {
			rook_points[m] = cv::Point(int(boxes[n][m][0]), int(boxes[n][m][1]));
		}

		const cv::Point *ppt[1] = { rook_points };
		int npt[] = { 4 };
		cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
	}

	cv::imwrite(strFileName, img_vis);
	std::cout << "The detection visualized image saved in "<< strFileName<< std::endl;
}
void getImages(std::string path, std::vector<std::string>&imagesList)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	imagesList.clear();
	hFile = _findfirst(p.assign(path).append("\\*.jpg").c_str(), &fileinfo);

	if (hFile != -1) {
		do {
			imagesList.push_back(fileinfo.name);//保存类名
		} while (_findnext(hFile, &fileinfo) == 0);
	}

	hFile = _findfirst(p.assign(path).append("\\*.bmp").c_str(), &fileinfo);

	if (hFile != -1) {
		do {
			imagesList.push_back(fileinfo.name);//保存类名
		} while (_findnext(hFile, &fileinfo) == 0);
	}

	hFile = _findfirst(p.assign(path).append("\\*.png").c_str(), &fileinfo);

	if (hFile != -1) {
		do {
			imagesList.push_back(fileinfo.name);//保存类名
		} while (_findnext(hFile, &fileinfo) == 0);
	}
}
std::string GetExePath()
{
	char szFilePath[MAX_PATH + 1] = { 0 };
	GetModuleFileNameA(NULL, szFilePath, MAX_PATH);
	/*
	strrchr:函数功能：查找一个字符c在另一个字符串str中末次出现的位置（也就是从str的右侧开始查找字符c首次出现的位置），
	并返回这个位置的地址。如果未能找到指定字符，那么函数将返回NULL。
	使用这个地址返回从最后一个字符c到str末尾的字符串。
	*/
	(strrchr(szFilePath, '\\'))[0] = 0; // 删除文件名，只获得路径字串//
	std::string path = szFilePath;
	return path;
}
int main(int argc, char **argv)
{
	std::string img_path, configPath;
	if (argc < 3) {
		/*std::cerr << "[ERROR] usage: " << argv[0]
			<< " configure_filepath image_path\n";
		exit(1);*/
		img_path = "F:\\image\\OCR\\2021-06-02\\";
		configPath = "D:\\4_code\\GitHub_Open\\PaddleOCR\\deploy\\cpp_infer\\tools\\config.txt";
	}
	else
	{
		img_path = argv[2];
		configPath = argv[1];
	}
	//创建目标文件目录
	std::string strCurPath = GetExePath();
	std::string strResultPath = strCurPath + "\\Result";
	CreateDirectory(strResultPath.c_str(), NULL);
	std::vector<std::string>imagesList;
	getImages(img_path, imagesList);
	std::vector<std::vector<std::vector<int>>>vecBoxResult(10);
	std::vector<char>vecCode(50);
	std::vector<std::vector<char>> vecResult(10, vecCode);
	const char* CS_DLLName;
	//std::string strConfigPath;
	//strConfigPath = "D:\\4_code\\GitHub_Open\\PaddleOCR\\deploy\\cpp_infer\\tools\\config.txt";
	//const char* strConfig = strConfigPath.c_str();
	CS_DLLName = "ocr_system.dll";
	OcrDetectorDll = ::LoadLibrary(CS_DLLName);
	//初始化
	typedef int(*OCRDetectorInit)(const char* strConfig);
	OCRDetectorInit OCRDetectorInit_;
	OCRDetectorInit_ = (OCRDetectorInit)GetProcAddress(OcrDetectorDll, "OCRInit");
	int i = OCRDetectorInit_(configPath.c_str());
	//执行
	//typedef int(*OCRDetectorRun)(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>&boxes, std::vector<std::string>&labels, int &nboxNum);
	typedef int(*OCRDetectorRun)(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>&boxes, std::vector<std::vector<char>>&labels, int &nboxNum);
	OCRDetectorRun OCRDetectorRun_;
	OCRDetectorRun_ = (OCRDetectorRun)GetProcAddress(OcrDetectorDll, "OCRSystem");
	for (int i = 0; i < imagesList.size(); i++)
	{
		//cv::Mat srcimg = cv::imread(strImgPath, cv::IMREAD_COLOR);
		std::string strFilePath = strResultPath + "\\" + imagesList.at(i);
		cv::Mat srcimg = cv::imread(img_path + imagesList.at(i), CV_8UC1);
		if (!srcimg.data) {
			std::cerr << "[ERROR] image read failed! image path: " << img_path << "\n";
			exit(1);
		}
		int nBoxnum;
		auto start = std::chrono::system_clock::now();
		BYTE *pByte;
		pByte = new BYTE[srcimg.cols*srcimg.rows];
		memcpy(pByte, srcimg.data, srcimg.cols*srcimg.rows * sizeof(unsigned char));
		int j = OCRDetectorRun_(pByte, srcimg.cols, srcimg.rows, vecBoxResult, vecResult, nBoxnum);
		VisualizeBboxes(srcimg, vecBoxResult, strFilePath);
		//auto end = std::chrono::system_clock::now();
		//auto duration =
		//	std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		//std::cout << "predict result:" << vecResult.size() << "\t";
		//std::cout << "Image FileName:" << imagesList.at(i) << "      Time Cost:"
		//	<< double(duration.count()) *
		//	std::chrono::microseconds::period::num /
		//	std::chrono::microseconds::period::den
		//	<< "s" << std::endl;
		//std::cout << std::endl;

	}
	//释放
	typedef int(*OCRDetectorFree)();
	OCRDetectorFree OCRDetectorFree_;
	OCRDetectorFree_ = (OCRDetectorFree)GetProcAddress(OcrDetectorDll, "OCRFree");
	OCRDetectorFree_();
	std::cout << "释放成功" << std::endl;
	system("pause");
    return 0;
}

