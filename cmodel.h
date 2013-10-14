/*
 * Copyright (C) 2013 by
 * 
 * Cheng Zhang
 * chengz@kth.se
 * and
 * Xavi Gratal
 * javiergm@kth.se
 * Computer Vision and Active Perception Lab
 * KTH Royal Institue of Techonology
 *
 * TopicModel_C++ is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * TopicModel_C++ is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TopicModel_C++; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 */

#ifndef _APPS_ONLINEHDP_CMODEL_H_
#define _APPS_ONLINEHDP_CMODEL_H_

#include "ccorpus.h"
#include <buola/mat.h>
#include <unordered_map>

namespace buola { namespace hdp {

//sufficient statistic 
class CHDPStats
{
public:
    CHDPStats(int pK,int pWt,int pDt)
        :   mBatchSize(pDt)
        ,   mVarSticksSS(mat::zeros(pK))
        ,   mVarBetaSS(mat::zeros(pK,pWt))
    {}

    void clear()
    {
        mVarSticksSS=mat::zeros(mVarSticksSS.Rows());
        mVarBetaSS=mat::zeros(mVarBetaSS.Rows(),mVarBetaSS.Cols());
    }

    int mBatchSize;
    mat::CVec_d mVarSticksSS; //K sized vector
    mat::CMat_d mVarBetaSS; // K *N
};

class CLabelStats
{
public:
    std::vector<mat::CVec_d> mBarsM;
    std::vector<mat::CMat_d> mBarsVar;
};

// model and onlinehdp
class CModel
{
public:
    //results
    CModel(int pK,int pT,int pD,int pW,double pEta,double pAlpha,double mGamma,double pKappa,double pTau,int pNumClasses);
    
    double InferenceHDP(mat::CMat_d &pPhi,mat::CMat_d &pVarPhi,const mat::CMat_d &pELogBetaDoc,const mat::CVec_d &pMatCounts);
    double InferenceSHDP(mat::CMat_d &pPhi,mat::CMat_d &pVarPhi,const CDocument &pDoc,const mat::CMat_d &pELogBetaDoc,
                          const mat::CVec_d &pMatCounts);
    double DocEStepHDP(const CDocument &pDoc,CHDPStats &pSS,CLabelStats &pLSS,const mat::CVec_d &pELogSticks1st, const std::unordered_map<int,int> &pUniqueWords);
    double DocEStepSHDP(const CDocument &pDoc,CHDPStats &pSS,CLabelStats &pLSS,const mat::CVec_d &pELogSticks1st, const std::unordered_map<int,int> &pUniqueWords,bool pSHDP);
    double InferenceLDA(const CDocument &pDoc,mat::CMat_d &pPhi);
    double InferenceSLDA(const CDocument &pDoc,mat::CMat_d &pPhi);
    void UpdateLogProbW();
    void UpdateMu(const CLabelStats &pSS,const CCorpus &pCorpus,const std::vector<int> &pDocIndices);
    double DocEStepLDA(const CDocument &pDoc,bool pUpdateMu,CLabelStats &pSS);
    double DocEStepSLDA(const CDocument &pDoc,bool pUpdateMu,CLabelStats &pSS);
    void ProcessDocumentsOnlineHDP(const CCorpus &pCorpus, const std::vector<int> &pIndices);
    void ProcessDocumentsOnlineSHDP(const CCorpus &pCorpus, const std::vector<int> &pIndices);
    void ProcessDocumentsHDP(const CCorpus &pCorpus);
    void ProcessDocumentsSHDP(const CCorpus &pCorpus);
    void ProcessDocumentsLDA(const CCorpus &pCorpus);
    void ProcessDocumentsSLDA(const CCorpus &pCorpus);
    void OptimalOrdering();
    
    void UpdateLambda(const CHDPStats &pSS,std::vector<int> &pWordList,bool pOptimalOrdering);
    void UpdateLambdaAll(const CHDPStats &pSS,std::vector<int> &pWordList,bool pOptimalOrdering);

    void UpdateExpectations(bool pAll);
    int Classify(const CDocument &pDoc);
    int ClassifyHDP(const CDocument &pDoc);
    void Print();
    
private:  
    int K; // top level truncation
    int T; // second level truncation
    int D; //number of documents
    int W; //as W in python; size of vocabulary

    double mEta; // the topic/words Dirichlet parameter
    double mAlpha; // the document level beta parameter or the alpha in LDA 
    double mGamma; // the corpus level beta parameter
    double mKappa; //learning rate which is a parameter to compute rho
    double mTau; // slow down which is a parameter to compute rho
    int mNumClasses;

    //the class para mu
    //C*K sized
    mat::CMat_d mMu;
    // the sticks
    mat::CMat_d mVarSticks; // 2*K-1
    mat::CVec_d mVarPhiSS; // K sized vector
    mat::CMat_d mLogProbW; //K*W matrix
    mat::CMat_d mLambda; // K * W matrix
    mat::CVec_d mLambdaSum; // sum over W, result a K sized vecor
    mat::CMat_d mELogBeta; //K*W
    mat::CVec_d mELogSticks1st; // K sized vector

    //timestamps and normalizers for lazy updates
    mat::CVec_d mTimestamps;
    std::vector<double> mR; //TODO check size

    double mVarConverge;
    double mScale;
    double mRhot;
    int mUpdateCount;
};

/*namespace hdp*/ } /*namespace buola*/ }

#endif
