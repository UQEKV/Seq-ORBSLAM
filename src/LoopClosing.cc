/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>
#include <unistd.h>

namespace ORB_SLAM2
{

    LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale, const vector<string> &imagePath):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0), vImagePath(imagePath)
{
    mnCovisibilityConsistencyTh = 3;
    imgSize = cv::Size(64, 32);
    patchSize = 8;
    ds = 8;
    threshold = 1.3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;
//    int i = 0;
    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            if(DetectLoopSeqSLAM()) //DetectLoopSeqSLAM is our sequence-based loop closure detection modul
            {
                cout<<"[";
                for(vector<Matching>::iterator match = vLoop.begin(); match!=vLoop.end(); match++){
                    cout<<"[" << match->first << ", " <<match->second<<"],";
                }
                cout<<"]"<<endl;
//                i = i+1;
//                cout<<i<<endl;
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())
               {
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

double LoopClosing::convertToSampleStddev(double popStddev, int N) {

        return sqrt(((popStddev * popStddev) * N) / (N - 1) );

    }

double LoopClosing::laplacianDist(double x, double mean, double pos) {
        double p = ( 1 / ( 2 * mean ) ) *  std::exp(-( x - pos ) / mean);
        return p;
    }

/**
* resizing to 32*64
* converting to grayscale
* normalizing the image with local path of size 8*8
*/
cv::Mat LoopClosing::Preprocessing(cv::Mat &img) {

        cv::Mat preprocessedImg = img.clone();
//        if(img.channels()>1){
//            cv::cvtColor(preprocessedImg, preprocessedImg, CV_BGR2GRAY);
//        }

        if(preprocessedImg.cols != imgSize.width && preprocessedImg.rows != imgSize.height){
            cv::resize(preprocessedImg, preprocessedImg, imgSize, cv::INTER_LANCZOS4);
        }

        return PatchNormalization(preprocessedImg);
    }


cv::Mat LoopClosing::PatchNormalization(cv::Mat &img) {

        cv::Mat normalizedImg = img.clone();
        cv::Mat patch, patchMean, patchStddev, temp;

        int patchArea = patchSize * patchSize;
        // extracting path
        for(int y=0; y<normalizedImg.rows; y+=patchSize){
            for(int x=0; x<normalizedImg.cols; x+=patchSize){
                //getting mean and standard deviation
                patch = cv::Mat(normalizedImg, cv::Rect(x, y, patchSize, patchSize));
                cv::meanStdDev(patch, patchMean, patchStddev);

                double mean = patchMean.at<double>(0, 0);
                double stddev = convertToSampleStddev(patchStddev.at<double>(0, 0), patchArea);

                patch.convertTo(temp, CV_64FC1);

                // normalizing the path
                if(stddev > 0.0){

                    for(cv::MatIterator_<double> itr = temp.begin<double>(); itr != temp.end<double>(); itr++){
                        *itr = 127 + cvRound((*itr - mean) / stddev);
                    }

                }
                else{ //std dev <= 0 set to all zero
                    temp = cv::Scalar::all(0);
                }

                // converting the buffer matrix to
                temp.convertTo(patch, CV_8UC1);

            }
        }

        return normalizedImg;

}


//dissimilarity ratio
int LoopClosing::calcNoneZero(cv::Mat &img) {

    int n = img.rows;
    int m = img.cols;
    int zeroDiff = 0;
    for(size_t i=0; i<n; i++){
        uchar* data = img.ptr<uchar>(i);

        for(size_t j=0; j<m; j++){
            if(int(data[j] == 0)){
                zeroDiff = zeroDiff + 1;
            }

        }
    }
    //    cout<<"zeros:"<<zeroDiff<<endl;
    int noneZeroDiff = imgSize.height * imgSize.width - zeroDiff;
//    float diffProportion = static_cast<float>(noneZeroDiff) / static_cast<float>(imgSize.height * imgSize.width);

    return noneZeroDiff;

}


// dissimilarity ratio matrix

void LoopClosing::calcDiffMatrix() {
    //The height of the difference matrix, the co-visible keyframes corresponding to the
    //current keyframe is removed, which means that the last n key frames of the template are removed
    int n = static_cast<int>(vTemplate.size() - mpCurrentKF->GetVectorCovisibleKeyFrames().size());
    //ds is the width of the sub-sequence
    int m = static_cast<int>(ds);

    diffMatrix = cv::Mat::zeros(n, m, CV_32FC1);
//    cout<< "template: "<<vTemplate.size()<<" covis: " <<mpCurrentKF->GetVectorCovisibleKeyFrames().size()<<n<<m<<endl;
    for(size_t i=0; i<n; i++){

        float* diffPtr = diffMatrix.ptr<float>(i);

        for(size_t j=0; j<m;j++){
            cv::Mat absDiff;
            cv::absdiff(vTemplate[vTemplate.size()-m+j].second, vTemplate[i].second, absDiff);
            diffPtr[j] = calcNoneZero(absDiff)/float(imgSize.height * imgSize.width );
                    //cv::sum( cv::abs( vTemplate[vTemplate.size()-m+j].second - vTemplate[i].second ) )[0] / static_cast<float>(imgSize.width * imgSize.height);

        }
    }
}
// sequence-based loop closure detection
bool LoopClosing::DetectLoopSeqSLAM() {

    { //this {} is to control the lock and unlock
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        cout<<mpCurrentKF->mnFrameId <<endl;
        //create template with keyframe and its image
        cv::Mat im = cv::imread(vImagePath[mpCurrentKF->mnFrameId],0);


        Template temp = make_pair(mpCurrentKF, Preprocessing(im));
        vTemplate.push_back(temp);

        // traversing all the templates in the vector and then removing keyframe labeled as bad
        for(vector<Template>::iterator it = vTemplate.begin(); it != vTemplate.end(); ++it )
        {
            KeyFrame* pKF = it->first;
            if(pKF->isBad()){
                it = vTemplate.erase(it);
                //
            }
        }

        mlpLoopKeyFrameQueue.pop_front();
        //        cout<<"after popping kf"<<mlpLoopKeyFrameQueue.front()->mnId<<endl;
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }

    if( (mpCurrentKF->mnId<mLastLoopKFid+10) || (vTemplate.size()<30) )
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    mvpEnoughConsistentCandidates.clear();

    calcDiffMatrix();

    cv::Mat total = cv::Mat::zeros(diffMatrix.size(), CV_32FC1);
    cv::Mat path = cv::Mat::zeros(diffMatrix.size(), CV_8UC1);
    diffMatrix.col(0).copyTo(total.col(0));
    float cost = 0;
    float temp = 0;
    uchar p = 0;
    double tminv;
    double tmaxv;
    cv::Point pt_min, pt_max;
    //There are segmentation problem with size_t because 1-1 = an extremely huge number such as 123456789 etc.
    for(int j=1; j<diffMatrix.cols; j++ ){
        for(int i=0; i<diffMatrix.rows; i++){

            cost = total.at<float>(i, j-1) + diffMatrix.at<float>(i, j);
            p = 0;
//
            if((i-1) >= 0){
                temp = total.at<float>(i-1, j-1) + diffMatrix.at<float>(i, j);
                if(temp < cost){
                    cost = temp;
                    p = 1;
                }
            }
//
            if((i-2) >= 0){
                temp = total.at<float>(i-2, j-1) + diffMatrix.at<float>(i, j);
                if(temp < cost){
                    cost = temp;
                    p = 2;
                }
            }
//
//            if((i-3) >= 0){
//                temp = total.at<float>(i-3, j-1) + diffMatrix.at<float>(i, j);
//                if(temp < cost){
//                    cost = temp;
//                    p = 3;
//                }
//            }
//
            total.at<float>(i, j) = cost;
            path.at<uchar>(i, j-1) = p;

//            cout<<"cost: "<< cost <<" "<< total.at<float>(i, j-1) << " "<< diffMatrix.at<float>(i, j) <<endl;
        }
    }


    cv::minMaxLoc(total.col(total.cols-1), &tminv, &tmaxv, &pt_min, &pt_max);
    double meanv = cv::mean(diffMatrix.col(diffMatrix.cols-1)).val[0];
    double minv = diffMatrix.at<float>(pt_min.y, diffMatrix.cols-1);

    double p_mean = laplacianDist(meanv, meanv, 0);
    double p_min = laplacianDist(minv, meanv, 0);

    unsigned long currentFrameID = mpCurrentKF->mnFrameId;
    unsigned long matchedFrameID = vTemplate[pt_min.y].first->mnFrameId;


    if(p_min/p_mean > threshold){
        mvpEnoughConsistentCandidates.push_back(vTemplate[pt_min.y].first);
        vLoop.emplace_back(currentFrameID, matchedFrameID);
//        return true;
    }

    cout << "minv = " << minv<< endl;
    cout<< "meanv = " <<meanv<<endl;
    cout<< "laplacian min = " << p_min<<endl;
    cout<< "laplacian mean = " << p_mean<<endl;
    cout << "proportion = " << p_min/p_mean<<endl;
    cout << "idx_min = " << pt_min.y << endl;
    cout<< "matched frame id = " << matchedFrameID<<endl;
    cout<<"current frame id = "<<currentFrameID<<endl;
    cout<< "*******************************************************" << endl;

    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        cout<<"There is no loop closure!!!!!!!!!!!!!!!!!!"<<endl;
        cout<< "*******************************************************" << endl;
        cout<<" "<<endl;
        return false;
    }
    else
    {

        cout<<"loop detected by JQ_SeqSLAM!!!!!!!!!!!!!!!!!"<<endl;
        cout<< "*******************************************************" << endl;
        cout<<" "<<endl;
        return true;
    }

    if(p_min/p_mean > threshold){
        vLoop.emplace_back(currentFrameID, matchedFrameID);
        cout<<"loop detected by JQ_SeqSLAM!!!!!!!!!!!!!!!!!"<<endl;
//        cout<<"current frame id = "<<currentFrameID<<endl;
//        cout<< "matched frame id = " << matchedFrameID<<endl;
        cout<< "*******************************************************" << endl;
        cout<<" "<<endl;
        return true;

    }else{

        cout<<"There is no loop closure!!!!!!!!!!!!!!!!!!"<<endl;
        cout<< "*******************************************************" << endl;
        cout<<" "<<endl;
        return false;
    }

}



bool LoopClosing::DetectLoop()
{
    { //this {} is to control the lock and unlock
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
//        cout<<mpCurrentKF->mnFrameId<<endl;
//        //create template with keyframe and its image
//        cv::Mat im = cv::imread(vImagePath[mpCurrentKF->mnFrameId],0);
//        Template temp = make_pair(mpCurrentKF, Preprocessing(im));
//        vTemplate.push_back(temp);

        // traversing all the templates in the vector and then removing keyframe labeled as bad
//        for(vector<Template>::iterator it = vTemplate.begin(); it != vTemplate.end(); ++it )
//        {
//            KeyFrame* pKF = it->first;
//            if(pKF->isBad()){
//                it = vTemplate.erase(it);
////
//            }
//        }

        mlpLoopKeyFrameQueue.pop_front();
//        cout<<"after popping kf"<<mlpLoopKeyFrameQueue.front()->mnId<<endl;
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }
//    if( vTemplate.size()<20 )
//    {
//        return false;
//    }

//    calcDiffMatrix();

//    if( DetectLoopSeqSLAM() ){
//        cout<<"[";
//        for(vector<Matching>::iterator match = vLoop.begin(); match!=vLoop.end(); match++){
//            cout<<"[" << match->first << ", " <<match->second<<"]";
//        }
//        cout<<"]"<<endl;
//    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if( mpCurrentKF->mnId<mLastLoopKFid+10 )
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

//    calcDiffMatrix();
//
//    if( DetectLoopSeqSLAM() ){
//
//        for(vector<Matching>::iterator match = vLoop.begin(); match!=vLoop.end(); match++){
//            cout<<"Current frame ID = " << match->first << " Matched frame ID = " <<match->second<<endl;
//        }
//
//    }
    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
//    cout<<"The number of conneted keyframes: "<<vpConnectedKeyFrames.size()<<endl;
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    //redundant
    mpCurrentKF->SetErase();
    return false;
}


//bool LoopClosing::DetectLoop()
//{
//    { //this {} is to control the lock and unlock
//        unique_lock<mutex> lock(mMutexLoopQueue);
//        mpCurrentKF = mlpLoopKeyFrameQueue.front();
//
//        //create template with keyframe and its image
//        cv::Mat im = cv::imread(vImagePath[mpCurrentKF->mnFrameId],CV_LOAD_IMAGE_UNCHANGED);
//        Template temp = make_pair(mpCurrentKF, Preprocessing(im));
//        vTemplate.push_back(temp);
//
//
//        // traversing all the templates in the vector and then removing keyframe labeled as bad
//        for(vector<Template>::iterator it = vTemplate.begin(); it != vTemplate.end(); ++it )
//        {
//            KeyFrame* pKF = it->first;
//            if(pKF->isBad()){
//                it = vTemplate.erase(it);
//                //
//            }
//        }
//        //        cout<< "template: "<<vTemplate.size()<<" covis: " <<mpCurrentKF->GetVectorCovisibleKeyFrames().size()<<endl;
//        //        cout<<"kfs with culling: ";
//        //        for(size_t i=0; i<vTemplate.size(); i++)
//        //        {
//        //            KeyFrame* pKF = vTemplate[i].first;
//        //            cout<<pKF->mnFrameId<<" ";
//        //        }
//        //        cout<<endl;
//
//        //        vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
//        //        cout<<"map: "<<vpKFs.size()<<endl;//this container is a red black tree, so the sequence is not continuous.
//        //        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
//        //        cout<<"kf: ";
//        //        for(size_t i=0; i<vpKFs.size(); i++)
//        //        {
//        //            KeyFrame* pKF = vpKFs[i];
//        //            cout<<pKF->isBad()<<" ";
//        //        }
//        //        cout<<endl;
//        mlpLoopKeyFrameQueue.pop_front();
//        //        cout<<"after popping kf"<<mlpLoopKeyFrameQueue.front()->mnId<<endl;
//        // Avoid that a keyframe can be erased while it is being process by this thread
//        mpCurrentKF->SetNotErase();
//    }
//
//    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
//    if( (mpCurrentKF->mnId<mLastLoopKFid+10) || (vTemplate.size()<20) )
//    {
//        mpKeyFrameDB->add(mpCurrentKF);
//        mpCurrentKF->SetErase();
//        return false;
//    }
//
//    calcDiffMatrix();
//
//    // Compute reference BoW similarity score
//    // This is the lowest score to a connected keyframe in the covisibility graph
//    // We will impose loop candidates to have a higher similarity than this
//    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
//    //    cout<<"The number of conneted keyframes: "<<vpConnectedKeyFrames.size()<<endl;
//    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
//    float minScore = 1;
//    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
//    {
//        KeyFrame* pKF = vpConnectedKeyFrames[i];
//        if(pKF->isBad())
//            continue;
//        const DBoW2::BowVector &BowVec = pKF->mBowVec;
//
//        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);
//
//        if(score<minScore)
//            minScore = score;
//    }
//
//    // Query the database imposing the minimum score
//    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);
//
//    // If there are no loop candidates, just add new keyframe and return false
//    if(vpCandidateKFs.empty())
//    {
//        mpKeyFrameDB->add(mpCurrentKF);
//        mvConsistentGroups.clear();
//        mpCurrentKF->SetErase();
//        return false;
//    }
//
//    // For each loop candidate check consistency with previous loop candidates
//    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
//    // A group is consistent with a previous group if they share at least a keyframe
//    // We must detect a consistent loop in several consecutive keyframes to accept it
//    mvpEnoughConsistentCandidates.clear();
//
//    vector<ConsistentGroup> vCurrentConsistentGroups;
//    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
//    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
//    {
//        KeyFrame* pCandidateKF = vpCandidateKFs[i];
//
//        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
//        spCandidateGroup.insert(pCandidateKF);
//
//        bool bEnoughConsistent = false;
//        bool bConsistentForSomeGroup = false;
//        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
//        {
//            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;
//
//            bool bConsistent = false;
//            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
//            {
//                if(sPreviousGroup.count(*sit))
//                {
//                    bConsistent=true;
//                    bConsistentForSomeGroup=true;
//                    break;
//                }
//            }
//
//            if(bConsistent)
//            {
//                int nPreviousConsistency = mvConsistentGroups[iG].second;
//                int nCurrentConsistency = nPreviousConsistency + 1;
//                if(!vbConsistentGroup[iG])
//                {
//                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
//                    vCurrentConsistentGroups.push_back(cg);
//                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
//                }
//                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
//                {
//                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
//                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
//                }
//            }
//        }
//
//        // If the group is not consistent with any previous group insert with consistency counter set to zero
//        if(!bConsistentForSomeGroup)
//        {
//            ConsistentGroup cg = make_pair(spCandidateGroup,0);
//            vCurrentConsistentGroups.push_back(cg);
//        }
//    }
//
//    // Update Covisibility Consistent Groups
//    mvConsistentGroups = vCurrentConsistentGroups;
//
//
//    // Add Current Keyframe to database
//    mpKeyFrameDB->add(mpCurrentKF);
//
//    if(mvpEnoughConsistentCandidates.empty())
//    {
//        mpCurrentKF->SetErase();
//        return false;
//    }
//    else
//    {
//        return true;
//    }
//
//    mpCurrentKF->SetErase();
//    return false;
//}




bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())
            {
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
                    bMatch = true;
                    mpMatchedKF = pKF;
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();

            if(pKFi!=mpCurrentKF)
            {
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId;   
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
