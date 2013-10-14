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
 * along with TopicModel_c++; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 */
#include "cmodel.h"
#include "opt.h"
#include <buola/mat/gamma.h>
#include <buola/mat/initializers.h>
#include <buola/mat/optimize.h>
#include <buola/mat/optimize_nlopt.h>
#include <buola/iterator/counter.h>

///\todo remove this... it's only to match cheng's version
#include <boost/random/gamma_distribution.hpp>
#include <boost/random.hpp>
#include <fstream>

namespace buola { namespace hdp {
    
const int MSTEP_MAX_ITER = 50;
const double sRhotBound=0.0;

mat::CMat_d dirichlet_expectation(const mat::CMat_d &pAlpha)
{
    // For a vector theta ~ Dir(alpha), compute E[log(theta)] given alpha
    mat::CMat_d lDigAlpha=mat::digamma(pAlpha);
    mat::CVec_d lDigSum=mat::digamma(sum(rows(pAlpha)));
    return lDigAlpha-extend(lDigSum);
}

mat::CVec_d expect_log_sticks(const mat::CMat_d& pSticks)
{
     //For stick-breaking hdp, this returns the E[log(sticks)]
     // it is initaled as 2 * K-1 Matrix. The first row is 1, the seocnd row is K-1:-1:1
     mat::CRow_d lSSum=sum(cols(pSticks));
     
     mat::CRow_d lDigSum=mat::digamma(lSSum);
     mat::CRow_d lDigPara1=mat::digamma(pSticks(0,nAll));
     mat::CRow_d lDigPara2=mat::digamma(pSticks(1,nAll));

     //ElogW is E[log beta']
     mat::CRow_d lELogW=lDigPara1-lDigSum;
     //Elog!_W is E[log(1-beta')] 
     mat::CRow_d lELog1W=lDigPara2-lDigSum;

     int n = pSticks.Cols()+1;
     mat::CVec_d lELogSticks=mat::zeros(n);
     for(int i=0;i<n-1;i++)
         lELogSticks[i]=lELogW[i];
     double lSum=0.0;
     for(int i=1;i<n;i++)
     {
         lSum+=lELog1W[i-1];
         lELogSticks[i]+=lSum;
     }

     return lELogSticks;
}

double log_normalize(const mat::CRow_d &pV)
{
    const double lLogMax=100.0;
    double lLogShift=lLogMax-log(pV.Cols()+1.0)-max(pV);
    return log(sum(exp(pV+lLogShift)))-lLogShift;
}

mat::CVec_d log_normalize(const mat::CMat_d &pV)
{
    const double lLogMax=100.0;
    double lVLog=log(pV.Cols()+1);
    
    mat::CVec_d lLogShift=lLogMax-lVLog-max(rows(pV));
    mat::CVec_d lLogTot=log(sum(rows(exp(pV+extend(lLogShift)))));
    return lLogTot-lLogShift;
}

CModel::CModel(int pK,int pT,int pD,int pW,double pEta,double pAlpha,double pGamma,double pKappa,double pTau,int pNumClasses)
    :   K(pK)
    ,   T(pT)
    ,   D(pD)
    ,   W(pW)
    ,   mEta(pEta)
    ,   mAlpha(pAlpha)
    ,   mGamma(pGamma)
    ,   mKappa(pKappa)
    ,   mTau(pTau+1)
    ,   mNumClasses(pNumClasses)
    ,   mVarConverge(0.0001)
    ,   mScale(1.0)
    ,   mUpdateCount(0)
{
    // initialize the sticks
    mVarSticks=mat::ones(2,K-1);
    //set the second row from K-1 to 1 in a descend orer
    for(int i=0;i<K-1;i++)
        mVarSticks(1,i)=K-1-i;
    
    mVarPhiSS=mat::zeros(K);
    mLambda=mat::zeros(K,W);
    std::ifstream lF("/home/xavi/random_lambda");
    lF.read((char*)mLambda.Data(),K*W*sizeof(double));

    mELogBeta=dirichlet_expectation(mLambda+mEta);
    mLambdaSum=sum(rows(mLambda));

    //Eq 6 in online LDA paper E[logbeta_kw] = psi(lambda_kw) -psi(sum^W lambda_kw)
    
    //initial the class para mu
    //initial to zero
    //mMu=mat::random(mNumClasses-1,K,std::uniform_real_distribution<double>(-1.0,1.0));
    mMu=mat::zeros(mNumClasses,K);
    
    //time stamps and normalizers for lazy updates
    mTimestamps=mat::zeros(W);
    mR.push_back(0);
}

double CModel::DocEStepHDP(mat::CMat_d &pPhi,mat::CMat_d &pVarPhi,const mat::CMat_d &pELogBetaDoc,const mat::CVec_d &pMatCounts)
{
    double lConverge=1.0;
    double lLikelihood=-1e100;
    
    mat::CMat_d lV=mat::ones(2,T-1);
    lV(1,nAll)=mAlpha;

    mat::CVec_d lELogSticks2nd=expect_log_sticks(lV); //T sized vector

    for(int lIter=0;lIter<100&&(lConverge<0.0||lConverge>mVarConverge);lIter++)
    {
        //##############var_phi / rho in the ICCV paper######################
        // sum over words K*#wordis is scaled by word count
        mat::CMat_d lELogBetaDocCounts=pELogBetaDoc**extend(pMatCounts.T());
        
        //var phi is a T*K matrix. T-#document level topics K-# corpus level topics
        //this is the first part in eq(17) in chong s paper
        //Phi: N*T Elogbeta:K*N
        pVarPhi=pPhi.T()*lELogBetaDocCounts.T();
        
        //Elogsticks_1st K sized vector (K*1)
        //plus the second part
        if(lIter>1)
            pVarPhi+=extend(mELogSticks1st.T());

        //var_phi is T*K log norm is T sized vector
        mat::CVec_d lLogNorm=log_normalize(pVarPhi);
        
        mat::CMat_d lLogVarPhi=pVarPhi-extend(lLogNorm);
        pVarPhi=exp(lLogVarPhi);
        //##############phi / zeta ######################
        //phi is N*T N is the doc.words.size()
        pPhi=pELogBetaDoc.T()*pVarPhi.T();

            //phi N*T Elogsticks_2nd is T size vector (T*1)
        if(lIter>1)
            pPhi+=extend(lELogSticks2nd.T());

        lLogNorm=log_normalize(pPhi);
        //log_normi N*1  phi N*T
        mat::CMat_d lLogPhi=pPhi-extend(lLogNorm);
        pPhi=exp(lLogPhi);

        //##############v######################
        //phi N*Ti
        //phi_all N*T mat_counts N
        mat::CMat_d lPhiAll=pPhi**extend(pMatCounts);
        lV(0,nAll)=1.0+sum(cols(lPhiAll(nAll,0,T-1)));
        mat::CMat_d lPhiAllColSum=sum(cols(lPhiAll(nAll,1,T-1)));
        mat::CMat_d lPhiAllInvCumSum(1,T-1);
        lPhiAllInvCumSum(0,T-2)=lPhiAllColSum(0,T-2);
        for (int k = 1; k< T-1; k++)
            lPhiAllInvCumSum(0,T-2-k) = lPhiAllInvCumSum(0,T-1-k) + lPhiAllColSum(0,T-2-k);

        lV(1,nAll)=mAlpha+lPhiAllInvCumSum;
        lELogSticks2nd=expect_log_sticks(lV);


        //##################compute likelihood############
        double lNewLikelihood=0.0;
        //var_phi / c part
        //Elogsticks_1st K*1
        //log_var_phi T*K
        lNewLikelihood+=sum(pVarPhi**(extend(mELogSticks1st.T())-lLogVarPhi));
        //v part //in the python code: m_T, m_alpha
        lNewLikelihood+=(T-1)*log(mAlpha);
        //v is 2*(T-1)
        mat::CRow_d lDigSum=digamma(sum(cols(lV))); // 1*(T-1) sized
        mat::CMat_d lTT1=mat::ones(2,lV.Cols());
        lTT1(1,nAll)*=mAlpha;
        lTT1-=lV;
        mat::CMat_d lTT2=digamma(lV);
        lTT2-=extend(lDigSum);
        
        lNewLikelihood+=sum(lTT1**lTT2);

        lTT1=lgamma(sum(cols(lV)));
        lTT2=lgamma(lV);
        lNewLikelihood-=sum(lTT1)-sum(lTT2);

        //z part
        //Elogsticks_2nd T*1
        
        lNewLikelihood+=sum((-lLogPhi+extend(lELogSticks2nd.T()))**pPhi);

        // x part, the data part
        //phi: N*T
        //Elog_beta_doc: K*N
        //var_phi: T*K
        //mat_counts:  N
        mat::CMat_d lTT=pELogBetaDoc**extend(pMatCounts.T());
        lNewLikelihood+=sum(pPhi.T()**(pVarPhi*lTT));

        lConverge=(lNewLikelihood-lLikelihood)/abs(lLikelihood);
        msg_info() << "converge:" << lConverge << "\n";
        lLikelihood=lNewLikelihood;
        
        if(lConverge<-0.000001)
            msg_warn() << "likelihood is decreasing\n";
    }

    msg_info() << "DocEStepHDP " << lLikelihood << "\n";
    
    return lLikelihood;
}

double CModel::DocEStepSHDP(mat::CMat_d &pPhi,mat::CMat_d &pVarPhi,const CDocument &pDoc,const mat::CMat_d &pELogBetaDoc,
                          const mat::CVec_d &pMatCounts)
{
   

    int FP_MAX_ITER = 10;

    //compute  sf_aux which is the role of eq(6) in chong 09f
    mat::CVec_d lSFAux=mat::ones(mNumClasses);
    for(int l=0;l<mNumClasses;l++)
    {
        mat::CRow_d lTV=exp((1.0/pDoc.mTotal)*mMu(l,nAll));
        lSFAux(l)=prod(pPhi*pVarPhi*lTV.T());
    }

    mat::CMat_d lELogBetaDocCounts=pELogBetaDoc**extend(pMatCounts.T());

    mat::CMat_d lV=mat::ones(2,T-1);
    lV(1,nAll)=mAlpha;

    mat::CVec_d lELogSticks2nd=expect_log_sticks(lV); //T sized vector

    CVarPhiFunc lOptFunc(mNumClasses,pDoc.mLabel,pDoc.mTotal,mMu,pPhi,pMatCounts,lELogBetaDocCounts,mELogSticks1st,0.01);
    pVarPhi=mat::minimize_gsl_fdf(log(pVarPhi/(1-pVarPhi)),lOptFunc,0.02,1e-4,1e-4);
    pVarPhi=exp(pVarPhi)/(exp(pVarPhi)+1);
//    pVarPhi[pVarPhi<1e-10]=1e-10;
//    pVarPhi=mat::minimize_nlopt(pVarPhi,lOptFunc,1e-10,1.2,0.02,1e-4);
    
    pVarPhi[pVarPhi<1e-100]=1e-100;
        for ( int rr=0; rr< pVarPhi.Rows(); rr++){
        for ( int cc=0; cc< pVarPhi.Cols(); cc++){
            if(isnan(pVarPhi(rr,cc)) || isinf(pVarPhi(rr,cc))){
  
                pVarPhi(rr,cc)=1;
            }
        }
        }

    
    mat::CVec_d lNormVarPhi=sum(rows(pVarPhi));
 
    pVarPhi/=extend(lNormVarPhi);

    mat::CMat_d lLogVarPhi=log(pVarPhi);
    


    //#################phi/zeta#######################
    // the fixed point update is applied for zeta
    mat::CMat_d lLogPhi=mat::zeros(pPhi.Rows(),pPhi.Cols());

    for(int n=0;n<pDoc.mWords.size();n++)
    {
        //compute h
        mat::CVec_d lSFParams=mat::zeros(T);
        
        for(int l=0;l<mNumClasses;l++)
        {
            //var_phi*zeta result in  K*1 vector
            mat::CRow_d lTV=pPhi(n,nAll)*pDoc.mCounts[n]*pVarPhi;
            mat::CVec_d lTV2=exp(pDoc.mCounts[n]*mMu(l,nAll)/pDoc.mTotal).T();
            lSFAux(l)/=lTV*lTV2; // take out the n_now word

            for(int t=0;t<T;t++)
                lSFParams(t)+=lSFAux(l)*pVarPhi(t,nAll)*lTV2; 
        }
        
        //VectorXd oldphi = phi.row(n).transpose(); //T siazed vector
        for (int lFPIter=0;lFPIter<FP_MAX_ITER;lFPIter++)
        {
            double lSFVal=pPhi(n,nAll)*lSFParams;

            //phi is N*T N is the doc.words.size()
            mat::CVec_d lTmpPhi=pVarPhi*pELogBetaDoc(nAll,n);
            pPhi(n,nAll)=lTmpPhi.T();

            pPhi(n,nAll)+=lELogSticks2nd.T();

            //add the softmax part
            pPhi(n,nAll)+=(pDoc.mCounts[n]*mMu(pDoc.mLabel,nAll)*pVarPhi.T())/pDoc.mTotal;
            pPhi(n,nAll)+=lSFParams.T()/(lSFVal*pDoc.mCounts[n]);
            
            mat::CVec_d lLogNorm{log_normalize(mat::CRow_d(pPhi(n,nAll)))};
            //log_normi N*1  phi N*T
            lLogPhi(n,nAll)=pPhi(n,nAll)-extend(lLogNorm);
            pPhi(n,nAll)=exp(lLogPhi(n,nAll));
        }

        // back to sf_aux value
        for(int l=0;l<mNumClasses;l++)
        {
            //var_phi*zeta result in  K*1 vector
            mat::CVec_d lTV=pVarPhi.T()*pPhi(n,nAll).T()*pDoc.mCounts[n]; 
            mat::CRow_d lTV2=exp(pDoc.mCounts[n]*mMu(l,nAll)/pDoc.mTotal);

            lSFAux[l]*=lTV2*lTV; // take out the n_now word
        }
    }
    
    //##############v / beta parameter ######################
    //phi N*Ti
    //phi_all N*T mat_counts N*1
    mat::CMat_d lPhiAll=pPhi**extend(pMatCounts);
    // eq(15) in chong 11
    lV(0,nAll)=sum(cols(lPhiAll(nAll,0,T-1)));
    mat::CRow_d lPhiAllColSum=sum(cols(lPhiAll(nAll,1,T-1)));
    mat::CRow_d lPhiAllInvCumSum(T-1);
    lPhiAllInvCumSum(T-2)=lPhiAllColSum(T-2);
    for (int k=1;k<T-1;k++)
        lPhiAllInvCumSum(T-2-k)=lPhiAllInvCumSum(T-1-k)+lPhiAllColSum(T-2-k);

    lV(1,nAll)=mAlpha+lPhiAllInvCumSum;
    lELogSticks2nd=expect_log_sticks(lV);

    //##################compute likelihood############
    //var_phi / c part
    //Elogsticks_1st K*1
    //log_var_phi T*K
    double lLikelihood=sum(pVarPhi**(extend(mELogSticks1st.T()) - lLogVarPhi));
    //v part //in the python code: m_T, m_alpha
    lLikelihood += (T-1)*mAlpha;
    //v is 2*(T-1)
    mat::CRow_d lDigSum=digamma(sum(cols(lV)));
    mat::CMat_d lTT1=mat::ones(2,lV.Cols());
    lTT1(1,nAll)*=mAlpha;
    lTT1-=lV;
    mat::CMat_d lTT2=digamma(lV)-extend(lDigSum);

    lLikelihood += sum(lTT1**lTT2);

    lLikelihood-=sum(lgamma(sum(cols(lV))))-sum(lgamma(lV));

    //z part
    //Elogsticks_2nd T*1
    lLikelihood+=sum((-lLogPhi+extend(lELogSticks2nd.T()))**pPhi);

    // x part, the data part
    //phi: N*T
    //Elog_beta_doc: K*N
    //var_phi: T*K
    //mat_counts:  N
    mat::CMat_d lTT=pELogBetaDoc**extend(pMatCounts.T());
    //the word part
    lLikelihood+=sum(pPhi.T()**(pVarPhi*lTT));

    // label part
        //the mu*(1/N sum sum var_phi *zeta) part
    mat::CMat_d lS1=pPhi*pVarPhi;
    lLikelihood+=mMu(pDoc.mLabel,nAll)*sum(cols(lS1)).T();
    lLikelihood-=log(sum(lSFAux));
    
    msg_info() << "DocEStepSHDP " << lLikelihood << "\n";

    
    return lLikelihood;
}

void CModel::DocEStep(const CDocument &pDoc,CSuffStats &pSS,CLabelStats &pLSS,const mat::CVec_d &pELogSticks1st,
                        const std::unordered_map<int,int> &pUniqueWords)
{
    //e step for a single  document
    std::vector<int> lIDs;
    // batchids are the value in unique_words correspond to every word of the document
    // the value in the unique_words record the order of the words has been seen
    //eg. training corpus [2 8 6 2] unique_words will be 2:0 8:1 6:2
    for(int i : pDoc.mWords)
        lIDs.push_back(pUniqueWords.at(i));
    
    mat::CMat_d lELogBetaDoc=mELogBeta(nAll,pDoc.mWords);

    mat::CMat_d lV=mat::ones(2,T-1);
    lV(1,nAll)=mAlpha;

    mat::CVec_d lELogSticks2nd=expect_log_sticks(lV); //T sized vector

    mat::CMat_d lPhi=mat::constant(pDoc.mWords.size(),T,1.0/T);
    mat::CMat_d lVarPhi=mat::constant(T,K,1.0/K);

    mat::CVec_d lMatCounts=mat::make_vec<double>(pDoc.mCounts);
    
    DocEStepHDP(lPhi,lVarPhi,lELogBetaDoc,lMatCounts);
    DocEStepSHDP(lPhi,lVarPhi,pDoc,lELogBetaDoc,lMatCounts);

    //update suff_stats
    //m_var_sticks_ss K*1
    pSS.mVarSticksSS=pSS.mVarSticksSS+sum(cols(lVarPhi)).T();

    mat::CMat_d lTS1=lPhi.T()**extend(lMatCounts.T());
    mat::CMat_d lTS2=lVarPhi.T()*lTS1;
    
    for(int i=0;i<lIDs.size();i++)
        pSS.mVarBetaSS(nAll,lIDs[i])+=lTS2(nAll,i);

    //update label_stats
    mat::CMat_d lThetaM=lPhi*lVarPhi;
    pLSS.mTheta.push_back(lThetaM);
}

void CModel::OptimalOrdering()
{
    //m_lambda_sum is a K sized vector
    std::vector<size_t> lIdx(counter_iterator(0),counter_iterator((int)mLambdaSum.Rows()));
    
    std::sort(lIdx.begin(),lIdx.end(),[this](size_t a,size_t b){return mLambdaSum[a]>mLambdaSum[b];});

    mat::CVec_d lVarPhiSSTmp=mVarPhiSS;
    mat::CMat_d lLambdaTmp=mLambda;
    mat::CVec_d lLambdaSumTmp=mLambdaSum;
    mat::CMat_d lELogBetaTmp=mELogBeta;
    
    for(int i=0;i<lIdx.size();i++)
    {
        mVarPhiSS(i)=lVarPhiSSTmp(lIdx[i]);
        mLambda(i,nAll)=lLambdaTmp(lIdx[i],nAll);
        mLambdaSum(i)=lLambdaSumTmp(lIdx[i]);
        mELogBeta(i,nAll)=lELogBetaTmp(lIdx[i],nAll);
    }
    
}

void CModel::UpdateLambda(CSuffStats &pSS,std::vector<int> &pWordList,bool pOptimalOrdering)
{
    // rhot will be between 0 and 1, and says how much to weight
    //the information we got from this mini-batch.
    
    mRhot=max(sRhotBound,mScale*pow(mTau+mUpdateCount,-mKappa));

    //Update appropriate columns of lambda based on documents.
        //cheng: lambada = (1-rhot)*lambda + rhot * ~lambda
    for(int w=0;w<pWordList.size();w++)
        mLambda(nAll,pWordList[w])=mLambda(nAll,pWordList[w])*(1-mRhot)+mRhot*pSS.mVarBetaSS(nAll,w)*D/pSS.mBatchSize;
    mLambdaSum=sum(rows(mLambda));
    
    UpdateExpectations();

    static int lIndex=1;
    
    std::ofstream lLambdaFile("lambda"+semantic_cast<std::string>(lIndex));
    for(int i=0;i<mLambda.Rows();i++)
    {
        for(int j=0;j<mLambda.Cols();j++)
        {
            lLambdaFile << mLambda(i,j) << " ";
        }
        lLambdaFile << "\n";
    }
    
    std::ofstream lELogBetaFile("elogbeta"+semantic_cast<std::string>(lIndex++));
    for(int i=0;i<mELogBeta.Rows();i++)
    {
        for(int j=0;j<mELogBeta.Cols();j++)
        {
            lELogBetaFile << mELogBeta(i,j) << " ";
        }
        lELogBetaFile << "\n";
    }
    
    mUpdateCount++;
    for(int w=0;w<pWordList.size();w++)
        mTimestamps(pWordList[w])=mUpdateCount;
    
    mR.push_back(mR.back()+log(1-mRhot));

    //m_varphi_ss is a K sized vector. 
    mVarPhiSS=(1.0-mRhot)*mVarPhiSS+mRhot*pSS.mVarSticksSS*D/pSS.mBatchSize;
    
    if(pOptimalOrdering)
        OptimalOrdering();

    //uodate top level sticks
    //Eq(26) Chong
    mVarSticks(0,nAll)=1+mVarPhiSS(0,K-1,nAll).T();
    //Eq(27) Chong
    ///xavi: the elements used from mVarPhiSS were different in Cheng's!!
    mat::CRow_d lInvSumVarPhi(K-1);
    lInvSumVarPhi(K-2)=mVarPhiSS(K-1);
    for(int t=1;t<K-1;t++)
        lInvSumVarPhi(K-2-t)=lInvSumVarPhi(K-1-t)+mVarPhiSS(K-1-t);

    mVarSticks(1,nAll)=lInvSumVarPhi+mGamma;
}

void CModel::ProcessDocuments(const CCorpus &pCorpus,const std::vector<int> &pIndices)//std::vector<int> & doc_unseen, double & score, int & count, double & unseen_score, int & unseen_count){
{
    msg_info() << "processing documents\n";

    std::unordered_map<int, int> lUniqueWords;
    std::vector<int> lWordList;

    for(int i : pIndices)
    {
        const CDocument &lDoc=pCorpus.mDocs[i];

        for(int w : lDoc.mWords)
        {
            if(lUniqueWords.find(w)==lUniqueWords.end())
            {
                lUniqueWords[w]=lWordList.size();
                lWordList.push_back(w);
            }
        }
    }

    
  //  msg_info() << "wordlist.size():" << lWordList.size() << "\n";
    
    for(int w : lWordList)
    {
        //the lazy updates on the necessart columns of lambda
        double lRW=mR[mTimestamps[w]];
        mLambda(nAll,w)*=exp(mR.back()-lRW);
        // update_Elogbeta
        //beta here is the beta in the online LDA eq(6) which is the topic-words distribution 
        //E[log beta_{kw} ] = psi(lambda) -psi(sum lambda)
        mELogBeta(nAll,w)=digamma(mEta+mLambda(nAll,w))-digamma(W*mEta+mLambdaSum);
    }

    CSuffStats lSS(K,lWordList.size(),pIndices.size());
    CLabelStats lLSS;
    lLSS.mBatchSize=pIndices.size();
    
    mELogSticks1st=expect_log_sticks(mVarSticks);

    //run variational inference on some new docs
    for(int i : pIndices)
    {
        msg_info() << "processing document " << i << "\n";
        const CDocument &lDoc=pCorpus.mDocs[i];

        DocEStep(lDoc,lSS,lLSS,mELogSticks1st,lUniqueWords);
    }

    msg_info() << "before update lambda\n";
    UpdateLambda(lSS,lWordList,true);

    //update label part
    msg_info() << "before update mu\n";
    UpdateLabelParameters(lLSS,pCorpus,pIndices);
}

void CModel::UpdateLabelParameters(CLabelStats &pLSS,const CCorpus &pCorpus,const std::vector<int> &pDocIndices)
{
    CSoftMaxFunc lOptFunc(pLSS,pCorpus,pDocIndices,mNumClasses,D,0.01);
//    mat::CMat_d lX=mat::minimize_gsl_fdf(mMu(0,mNumClasses-1,nAll),lOptFunc,0.02,1e-6,1e-6);
    mat::CMat_d lX=mat::minimize_gsl_fdf(mMu(0,mNumClasses-1,nAll),lOptFunc,0.1,1e-2,1e-2);
    mat::CMat_d lDF=-lOptFunc.DF(lX);
    
    double lF=-lOptFunc.F(lX);
    
    for(int i=0;i<mNumClasses-1;i++)
    {
        for(int j=0;j<mMu.Cols();j++)
        {
            mat::CMat_d lNewX=lX;
            lNewX(i,j)+=0.00001;
        }
    }
    
    
    

    mMu(0,mNumClasses-1,nAll)=(1-mRhot)*mMu(0,mNumClasses-1,nAll)+mRhot*lX;
   // mMu(0,mNumClasses-1,nAll)+=mRhot*lDF;
    
    msg_info() << "mu:\n" << mMu << "\n";
//    abort();
}

void CModel::UpdateExpectations()
{
    /*Since we're doing lazy updates on lambda, at any given moment 
    the current state of lambda may not be accurate. This function
    updates all of the elements of lambda and Elogbeta so that if (for
    example) we want to print out the topics we've learned we'll get the
    correct behavior.*/

    for (int w=0;w<W;w++)
    {
        //TODO check how to use m_r to update lambda
        //m_r<<m_r(m_r.tail(1)) + log(1-rhot);
        //m_timestamp(word_list[ww]) = m_updatect;
        mLambda(nAll,w)*=exp(mR.back()-mR[mTimestamps(w)]);
    }
    
    mat::CMat_d lTT1=digamma(mEta+mLambda);
    mat::CVec_d lTT2=digamma(W*mEta+mLambdaSum);
    mELogBeta=lTT1-extend(lTT2);
    mTimestamps=mat::constant(W,1,(double)mUpdateCount);
}

void CModel::HDP2LDA()
{
    UpdateExpectations();

    //m_var_sticks are 2*K-1
    mat::CRow_d lSticks = mVarSticks(0,nAll)/(mVarSticks(0,nAll)+mVarSticks(1,nAll));

    mLDAAlpha=mat::zeros(K);
    double lLeft=1.0;
    for(int i=0;i<K-1;++i)
    {
        mLDAAlpha(i)=lSticks(i)*lLeft;
        lLeft-=mLDAAlpha(i);
    }
    
    mLDAAlpha(K-1)=lLeft;
    mLDAAlpha*=mAlpha;

    // m_lambda_sum: K sized Vectot
    mat::CVec_d lTT2=W*mEta+mLambdaSum;
    mLDABeta=(mLambda+mEta)/extend(lTT2);
}

void CModel::Print()
{
    UpdateExpectations();
    msg_info() << "lambda:" << mLambda << "\n\n";
    msg_info() << "elogbeta:" << mELogBeta << "\n\n";
}

double CModel::LDAEStep(const CDocument &pDoc)
{
    mLDAGamma=mat::ones(mLDAAlpha.Rows());
    mat::CVec_d lELogTheta=dirichlet_expectation(mLDAGamma);
    mat::CVec_d lExpELogTheta=exp(lELogTheta);

    mat::CVec_d lMatCounts=mat::make_vec<double>(pDoc.mCounts);
    mat::CMat_d lBetaD=mLDABeta(nAll,pDoc.mWords);

    mat::CRow_d lPhiNorm=lExpELogTheta.T()*lBetaD+1e-100;

    //TODO how to set meanchange threshresh
    double lMeanChangeThresh=0.01;

    // set the max iter to 100 as chong
    for(int lIter=0;lIter<100;lIter++)
    {
        mat::CVec_d lLastGamma=mLDAGamma;
        mat::CVec_d lTT=lMatCounts/lPhiNorm.T();
        //betad K*N
        mLDAGamma=mLDAAlpha+lExpELogTheta**(lTT.T()*lBetaD.T()).T();
        lELogTheta=dirichlet_expectation(mLDAGamma);
        lExpELogTheta=exp(lELogTheta);

        lPhiNorm=lExpELogTheta.T()*lBetaD+1e-100;

        double lMeanChange=mean(abs(mLDAGamma-lLastGamma));

        if (lMeanChange<lMeanChangeThresh )
            break;
    }
    
    double lLikelihood=sum(lMatCounts**log(lPhiNorm.T()));

    // E[log p(theta | alpha ) - log q(theta | gamma )
    lLikelihood+=sum((mLDAAlpha-mLDAGamma)**lELogTheta);
    lLikelihood+=sum(lgamma(mLDAGamma)-lgamma(mLDAAlpha));
    lLikelihood+=lgamma(sum(mLDAAlpha))-lgamma(sum(mLDAGamma));
    
    return lLikelihood;
}

#if 0

void model::lda_e_step_split(document & doc, double & score){

    //split the document
    int num_train=ceil(doc.length / 2.0); // even numbers
    int num_test = floor( doc.length /2.0);

    VectorXd words_train(num_train);
    VectorXd counts_train(num_train);
    VectorXd words_test(num_test);
    VectorXd counts_test(num_test);
    int ii=0;
    int jj=0;
    for( int i=0; i< doc.length; ++i){
        if(i%2 == 0){
            words_train(ii)=doc.words[i];
            counts_train(ii)= doc.counts[i];
            ii++;
        }else{
            words_test(jj) = doc.words[i];
            counts_test(jj) = doc.counts[i];
            jj++;
        }
    }
    
    //do lda e step on the train part
    //the same as online lda alpgorithm
    lda_gamma = VectorXd::Ones(lda_alpha.rows()); // K sized vector
    MatrixXd Elogtheta = dirichlet_expectation(lda_gamma); // K sized 
    MatrixXd expElogtheta = exp(Elogtheta.array()); //Ksized
    MatrixXd betad(K, words_train.rows() ); // K*N sized

    VectorXd mat_counts(counts_train.rows()); //N*1
    for (int n=0; n< words_train.rows(); ++n){
        betad.col(n) = lda_beta.col(words_train(n));
        mat_counts(n) = doc.counts[n];
    }

    // the optimal phi_{dwk} is proportional to 
    //expElogtheta_k * expElogbetad_w. Phinorm is the normalizer
    MatrixXd phinorm = (expElogtheta.transpose()* betad).array() + 1e-100;// (1*K) *(K*N) = 1*N

    int iter = 0;
    VectorXd last_gamma;
    double meanchangethresh = 0.01;;
    // set the max iter to 100 as chong
    while(iter < 100){
        last_gamma = lda_gamma;
        iter++;
        double likelihood = 0.0;
        MatrixXd tt =  mat_counts.array()/phinorm.transpose().array(); // N*1
        // this way to present phi implicitly is to save memory and time
        //substituting the value of the optimal ohi back into
        // the uopdate for gamma gives this update. Cf. Lee&Seung 2001
        lda_gamma = lda_alpha.array() + expElogtheta.array()*
            ( tt.transpose() * betad.transpose()).transpose().array();
        Elogtheta = dirichlet_expectation(lda_gamma);
        expElogtheta = exp(Elogtheta.array());
        phinorm = (expElogtheta.transpose()* betad).array() + 1e-100;// (1*K) *(K*N) = 1*N

        VectorXd change=abs((lda_gamma.array()-last_gamma.array()).array());
        double meanchange = change.sum() / change.rows();

        if ( meanchange < meanchangethresh ){
            break;
        }
    }
    lda_gamma = lda_gamma.array()/ lda_gamma.sum();
    mat_counts = counts_test;
    count_split = mat_counts.sum();

    MatrixXd betad_tst(K, words_test.rows()); // K*N sized
    for (int n=0; n< words_test.rows(); ++n){
        betad_tst.col(n) = lda_beta.col(words_test(n));
    }
    score = (mat_counts.transpose().array() * log(((lda_gamma.transpose()* betad_tst).array() + 1e-100).array())).sum();
}
#endif

int CModel::Classification()
{
    mat::CVec_d lScore=mMu*mLDAGamma;
   /// msg_info() << mLDAGamma << "\n\nscore:\n" << lScore << "\n\n";

    int lIndex=indmax(lScore).first;
   // std::cout<<lIndex<<std::endl;

    return lIndex;
}

/*namespace hdp*/ } /*namespace buola*/ }
