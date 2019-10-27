// TGMTtemplate.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <tchar.h>
#include <map>

#include "TGMTfile.h"
#include "TGMTdebugger.h"
#include "TGMTsvm.h"
#include "TGMTutil.h"
#include "TGMTConfig.h"
#include <time.h>

#define INI_SECTION "SVM_classifier"

void PrintHelp()
{
	PrintMessage("This program auto classify and move file to each folder");
	std::cout << "Using with syntax: \n";
	debug_out(3, "SvmOpenCV_Classifier.exe -file <svm_file> -in <directory> -out <directory> -w <width> -h <height>\n");
	std::cout << "with <svm_file> is SVM trained file\n";
	std::cout << "and <directory> is directory contain images to predict\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CreateEmptyFolder(std::string dir)
{
	TGMTfile::CreateDir(dir);
	for (int i = 0; i < 10; i++)
	{
		TGMTfile::CreateDir(TGMTutil::FormatString("%s%d", dir.c_str(), i));
	}

	for (char i = 'A'; i <= 'Z'; i++)
	{
		TGMTfile::CreateDir(TGMTutil::FormatString("%s%c", dir.c_str(), i));
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

int _tmain(int argc, _TCHAR* argv[])
{
	GetTGMTConfig()->LoadSettingFromFile();

	std::string svmFile = GetTGMTConfig()->ReadValueString(INI_SECTION, "input_file");
	std::string inDir = GetTGMTConfig()->ReadValueString(INI_SECTION, "input_dir");
	std::string outDir = GetTGMTConfig()->ReadValueString(INI_SECTION, "output_dir");
	outDir = TGMTfile::CorrectPath(outDir);
	CreateEmptyFolder(outDir);


	clock_t startTime = clock();

	PrintMessage("Loading data...");
	GetTGMTsvm()->LoadData(svmFile);

	PrintMessage("Loading image...");
	std::vector<std::string> files = TGMTfile::GetImageFilesInDir(inDir);
	PrintMessage("Loaded %d images", files.size());

	std::map<int, int> maps;

	for (int i = 0; i < files.size(); i++)
	{
		SET_CONSOLE_TITLE("%d / %d", i + 1, files.size());

		int result =  GetTGMTsvm()->Predict(files[i]);
		maps[result]++;

		std::string fileName = TGMTfile::GetFileName(files[i]);
		PrintMessage("%s: %c", fileName.c_str(), (char)result);
		std::string targetFile = TGMTutil::FormatString("%s%c\\%s", outDir.c_str(), result, fileName.c_str());
		
		TGMTfile::CopyFileAsync(files[i], targetFile);
	}

	//Print Result
	PrintMessageBlue("Sum up:");
	std::map<int, int>::iterator iter = maps.begin();
	while (iter != maps.end())
	{
		PrintMessage("Class %c: %d", (char)iter->first, iter->second);
		iter++;
	}

	PrintSuccess("Classify complete");
	int elapsedTime = clock() - startTime;
	getchar();
	return 0;
}