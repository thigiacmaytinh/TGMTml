#pragma once
#include "stdafx.h"
#include "TGMTml.h"

#define GetTGMTann TGMTann::GetInstance

class TGMTann : public TGMTml
{
	static TGMTann* m_instance;

	static const int numCharacters;
	cv::Ptr<ANN_MLP> ann;
	bool m_isTrained;
public:
	TGMTann();
	~TGMTann();

	static TGMTann* GetInstance()
	{
		if (!m_instance)
			m_instance = new TGMTann();
		return m_instance;
	}

	int ClassifyAnn(cv::Mat mat);
	void TrainAnn(cv::Mat trainData, cv::Mat trainClasses, int nlayers);
	bool TrainData(cv::Mat matData, cv::Mat matLabel) override;

	float Predict(cv::Mat matInput) override;
	float Predict(std::string imgPath) override;

	void SaveModel(std::string filePath);
	bool LoadModel(std::string filePath);
};

