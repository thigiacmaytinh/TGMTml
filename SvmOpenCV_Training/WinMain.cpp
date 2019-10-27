// TGMTtemplate.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <tchar.h>
#include "TGMTfile.h"
#include "TGMTdebugger.h"
#include "TGMTsvm.h"
#include "TGMTutil.h"
#include "TGMTConfig.h"

#define INI_SECTION "SVM_training"

void PrintHelp()
{
	std::cout << "Using with syntax: \n";
	debug_out(3, "SvmOpenCV_Training.exe -in <directory> -out <file> -w <width> -h <height>\n");
		
	std::cout << "-in is directory contain sub directories, with each sub directory contain images same class, name of each sub directory only 1 character\n"
		<< "-out is file name you choose\n";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

void _tmain(int argc, _TCHAR* argv[])
{
	GetTGMTConfig()->LoadSettingFromFile();

	std::string dir = GetTGMTConfig()->ReadValueString(INI_SECTION, "input_dir");
	std::string fileOuput = GetTGMTConfig()->ReadValueString(INI_SECTION, "output_file");

	if (!TGMTfile::DirExist(dir))
	{
		PrintError("Directory \"%s\" does not exist", dir.c_str());
		return;
	}
	
	if (GetTGMTsvm()->TrainData(dir))
	{
		PrintSuccess("Training success");
		GetTGMTsvm()->SaveData(fileOuput);
	}	
}
