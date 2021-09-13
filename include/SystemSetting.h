//
// Created by jiangqiu on 2021/8/26.
//

#ifndef SYSTEMSETTING_H
#define SYSTEMSETTING_H

#include<string>
#include"ORBVocabulary.h"
#include<opencv2/opencv.hpp>


namespace ORB_SLAM2 {

    class SystemSetting{

    public:
        SystemSetting(ORBVocabulary* pVoc);

        bool LoadSystemSetting(const std::string strSettingPath);

    public:
        ORBVocabulary* pVocavulary;

        //camera parameter
        float width;
        float height;
        float fx;
        float fy;
        float cx;
        float cy;
        float invfx;
        float invfy;
        float bf;
        float b;
        float fps;
        cv::Mat K;
        cv::Mat DistCoef;
        bool initialized;
        //RGB camera parameters
        int nRGB;

        //ORB features' parameters
        int nFeatures;
        float fScaleFactor;
        int nLevels;
        float fIniThFAST;
        float fMinThFAST;

        //other parameter
        float ThDepth = -1;
        float DepthMapFactor = -1;

    };

}//namespace ORB_SLAM2

#endif //SystemSetting
