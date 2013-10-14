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
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <string>
namespace buola { namespace hdp {
    
const int MSTEP_MAX_ITER = 50;
const double sRhotBound=0.0;

static const int sHDPEMMaxIter=50;

static const int sLDAEMMaxIter=50;
static const int sLDAVarMaxIter=20;
static const double sLDAEMConvergence=1e-4;
static const double sLDAVarConvergence=1e-3;
static const double sLDAL2Penalty=0.01;
static const double sHDPEMConvergence=1e-4;

static const int sFPMaxIter=10;

static double log_sum(double log_a, double log_b)
{
    if(log_a<log_b)
        return log_b + log( 1+ exp(log_a - log_b));
    else
        return log_a + log( 1+ exp(log_b - log_a));
}

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
     //////do this properly and check if it is wrong in the original!!!
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
    mLambda=mat::random(K,W,std::gamma_distribution<double>(1.0));
    mLambda*=100.0*D/(K*W);
    mLambda-=mEta;

    mLambda=mat::random(K,W,std::uniform_real_distribution<double>(0,0.1));
    mLambda+=1.0/W;
    

    
    mELogBeta=dirichlet_expectation(mLambda+mEta);
    mLambdaSum=sum(rows(mLambda));
    mLogProbW=mat::zeros(K,W);
    
    UpdateLogProbW();

    mMu=mat::zeros(mNumClasses,K);
    
    //time stamps and normalizers for lazy updates
    mTimestamps=mat::zeros(W);
    mR.push_back(0);
}

double CModel::InferenceHDP(mat::CMat_d &pPhi,mat::CMat_d &pVarPhi,const mat::CMat_d &pELogBetaDoc,const mat::CVec_d &pMatCounts)
{
    double lConverge=1.0;
    double lLikelihood=-1e100;
    
    mat::CMat_d lV=mat::ones(2,T-1);
    lV(1,nAll)=mAlpha;

    mat::CVec_d lELogSticks2nd=expect_log_sticks(lV); //T sized vector

    for(int lIter=0;lIter<50&&(lConverge<0.0||lConverge>mVarConverge);lIter++)
    {
        //##############var_phi  / rho in the final version of the paper######################
        // sum over words K*#wordis is scaled by word count
        mat::CMat_d lELogBetaDocCounts=pELogBetaDoc**extend(pMatCounts.T());
        
        //var phi is a T*K matrix. T-#document level topics K-# corpus level topics
        //this is the first part in eq(17) in chong s paper
        //Phi: N*T Elogbeta:K*N
        pVarPhi=pPhi.T()*lELogBetaDocCounts.T();
        
        //Elogsticks_1st K sized vector (K*1)
        //plus the second part
        if(lIter>0)
            pVarPhi+=extend(mELogSticks1st.T());

        //var_phi is T*K log norm is T sized vector
        mat::CVec_d lLogNorm=log_normalize(pVarPhi);
        
        mat::CMat_d lLogVarPhi=pVarPhi-extend(lLogNorm);
        pVarPhi=exp(lLogVarPhi);
        //##############phi / zeta ######################
        //phi is N*T N is the doc.words.size()
        pPhi=pELogBetaDoc.T()*pVarPhi.T();

            //phi N*T Elogsticks_2nd is T size vector (T*1)
        if(lIter>0)
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
        //TODO check the computation of likelihood
        //var_phi / c part
        //Elogsticks_1st K*1
        //log_var_phi T*K
        double lNewLikelihood=sum(pVarPhi**(-lLogVarPhi+extend(mELogSticks1st.T())));
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

        lNewLikelihood-=sum(lgamma(sum(cols(lV))))-sum(lgamma(lV));

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
        lLikelihood=lNewLikelihood;
        
//        if(lConverge<-0.000001)
//            msg_warn() << "likelihood is decreasing\n";
    }

    return lLikelihood;
}

double CModel::InferenceSHDP(mat::CMat_d &pPhi,mat::CMat_d &pVarPhi,const CDocument &pDoc,const mat::CMat_d &pELogBetaDoc,
                          const mat::CVec_d &pMatCounts)
{
    //compute  sf_aux which is the role of eq(6) in chong 09f
    mat::CVec_d lSFAux=mat::ones(mNumClasses);
    //compute sf_aux. eq(6) in chong 09 in log space
    ///!!! this was wrong in the original
    for(int l=0;l<mNumClasses;l++)
    {
        mat::CRow_d lTV=exp((1.0/pDoc.mTotal)*mMu(l,nAll));
        lSFAux(l)=prod(pPhi*pVarPhi*lTV.T());
    }

    mat::CMat_d lELogBetaDocCounts=pELogBetaDoc**extend(pMatCounts.T());

    mat::CMat_d lV=mat::ones(2,T-1);
    lV(1,nAll)=mAlpha;

    mat::CVec_d lELogSticks2nd=expect_log_sticks(lV); //T sized vector
   //pVarPhi is the rho in the ICCV ws paper
    CVarPhiFunc lOptFunc(mNumClasses,pDoc.mLabel,pDoc.mTotal,mMu,pPhi,pMatCounts,lELogBetaDocCounts,mELogSticks1st,0.01);
    pVarPhi=mat::minimize_gsl_fdf(log(pVarPhi/(1-pVarPhi)),lOptFunc,0.02,1e-4,1e-4,10);
    pVarPhi=exp(pVarPhi)/(exp(pVarPhi)+1);
//    pVarPhi[pVarPhi<1e-10]=1e-10;
//    pVarPhi=mat::minimize_nlopt(pVarPhi,lOptFunc,1e-10,1.2,0.02,1e-4);
    
    pVarPhi[pVarPhi<1e-100]=1e-100;
    pVarPhi[is_nan(pVarPhi)]=1e-100;
    pVarPhi[is_inf(pVarPhi)]=1e-100;
//    if(!all(is_finite(pVarPhi)))
//        throw XProcess("not finite value in pVarPhi!!");
    
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
        for (int lFPIter=0;lFPIter<sFPMaxIter;lFPIter++)
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
            //std::cout<<"norm phi"<<phi.rowwise().sum()<<std::endl;
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
    lV(0,nAll)=1.0+sum(cols(lPhiAll(nAll,0,T-1)));
    mat::CRow_d lPhiAllColSum=sum(cols(lPhiAll(nAll,1,T-2)));
    mat::CRow_d lPhiAllInvCumSum(T-1);
    lPhiAllInvCumSum(T-2)=lPhiAllColSum(T-1);
    for (int k=1;k<T-1;k++)
        lPhiAllInvCumSum(T-2-k)=lPhiAllInvCumSum(T-1-k)+lPhiAllColSum(T-2-k);

    lV(1,nAll)=mAlpha+lPhiAllInvCumSum;
    lELogSticks2nd=expect_log_sticks(lV);
    
    //##################compute likelihood############
    //TODO check the computation of likelihood
    //var_phi / c part
    //Elogsticks_1st K*1
    //log_var_phi T*K
    double lLikelihood=sum(pVarPhi**(extend(mELogSticks1st.T()) - lLogVarPhi));
    //v part //in the python code: m_T, m_alpha
    lLikelihood += (T-1)*log(mAlpha);
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
    
    msg_info() << "InferenceSHDP " << lLikelihood << "\n";
    
    return lLikelihood;
}

double CModel::DocEStepSHDP(const CDocument &pDoc,CHDPStats &pSS,CLabelStats &pLSS,const mat::CVec_d &pELogSticks1st,
                        const std::unordered_map<int,int> &pUniqueWords,bool pSHDP)
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
    
    double lLikelihood;
    lLikelihood=InferenceHDP(lPhi,lVarPhi,lELogBetaDoc,lMatCounts);
    if(pSHDP)
        lLikelihood=InferenceSHDP(lPhi,lVarPhi,pDoc,lELogBetaDoc,lMatCounts);

    //update suff_stats
    //m_var_sticks_ss K*1
    pSS.mVarSticksSS=pSS.mVarSticksSS+sum(cols(lVarPhi)).T();

    mat::CMat_d lTS=lVarPhi.T()*(lPhi.T()**extend(lMatCounts.T()));
    
    for(int i=0;i<lIDs.size();i++)
        pSS.mVarBetaSS(nAll,lIDs[i])+=lTS(nAll,i);

    //update label_stats
    mat::CVec_d lBarM=sum(rows(lTS));
    mat::CMat_d lBarVar=diagm(sum(rows(lTS**extend(lMatCounts.T()))))-lTS*lTS.T();
    
    lBarM/=pDoc.mTotal;
    lBarVar/=sq(double(pDoc.mTotal));
    
    pLSS.mBarsM.push_back(std::move(lBarM));
    pLSS.mBarsVar.push_back(std::move(lBarVar));
    
    return lLikelihood;
}

double CModel::DocEStepHDP(const CDocument &pDoc,CHDPStats &pSS,CLabelStats &pLSS,const mat::CVec_d &pELogSticks1st, const std::unordered_map<int,int> &pUniqueWords)
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
    
    double lLikelihood;
    lLikelihood=InferenceHDP(lPhi,lVarPhi,lELogBetaDoc,lMatCounts);

    //update suff_stats
    //m_var_sticks_ss K*1
    pSS.mVarSticksSS=pSS.mVarSticksSS+sum(cols(lVarPhi)).T();

    mat::CMat_d lTS=lVarPhi.T()*(lPhi.T()**extend(lMatCounts.T()));
    
    for(int i=0;i<lIDs.size();i++)
        pSS.mVarBetaSS(nAll,lIDs[i])+=lTS(nAll,i);

    return lLikelihood;
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

double CModel::InferenceSLDA(const CDocument &pDoc,mat::CMat_d &pPhi)
{
    mat::CVec_d lGamma=mat::constant(K,1,mAlpha+pDoc.mTotal/double(K));
    mat::CVec_d lDigammaGamma=digamma(lGamma);
    pPhi=mat::constant(pDoc.mWords.size(),K,1.0/K);

    mat::CVec_d lSFAux(mNumClasses-1);
    
    for(int l=0;l<mNumClasses-1;l++)
    {
        lSFAux(l)=1.0;
        
        for(int n=0;n<pDoc.mWords.size();n++)
        {
            double t=0.0;
            for(int k=0;k<K;k++)
                t+=pPhi(n,k)*exp(mMu(l,k)*pDoc.mCounts[n]/pDoc.mTotal);
            lSFAux(l)*=t;
        }
    }
    
    double lLikelihood=0.0;
    
    for(int lIter=0;lIter<sLDAVarMaxIter;lIter++)
    {
        for(int n=0;n<pDoc.mWords.size();n++)
        {
            mat::CVec_d lSFParams=mat::zeros(K);
            for(int l=0;l<mNumClasses-1;l++)
            {
                double t=0.0;
                for(int k=0;k<K;k++)
                    t+=pPhi(n,k)*exp(mMu(l,k)*pDoc.mCounts[n]/pDoc.mTotal);
                lSFAux(l)/=t;
                for(int k=0;k<K;k++)
                    lSFParams(k)+=lSFAux(l)*exp(mMu(l,k)*pDoc.mCounts[n]/pDoc.mTotal);
            }
            mat::CRow_d lOldPhi=pPhi(n,nAll);
            for(int lFPIter=0;lFPIter<sFPMaxIter;lFPIter++)
            {
                double lSFVal=1.0+pPhi(n,nAll)*lSFParams;
                
                double lPhiSum=0.0;
                for(int k=0;k<K;k++)
                {
                    pPhi(n,k)=lDigammaGamma(k)+mLogProbW(k,pDoc.mWords[n]);
                    
                    if(pDoc.mLabel<mNumClasses-1)
                        pPhi(n,k)+=mMu(pDoc.mLabel,k)/pDoc.mTotal;
                    pPhi(n,k)-=lSFParams(k)/(lSFVal*pDoc.mCounts[n]);
                    
                    if(k>0)
                        lPhiSum=log_sum(lPhiSum,pPhi(n,k));
                    else
                        lPhiSum=pPhi(n,k);
                }
                for(int k=0;k<K;k++)
                    pPhi(n,k)=exp(pPhi(n,k)-lPhiSum);
            }
            
            for(int l=0;l<mNumClasses-1;l++)
            {
                double t=0.0;
                for(int k=0;k<K;k++)
                    t+=pPhi(n,k)*exp(mMu(l,k)*pDoc.mCounts[n]/pDoc.mTotal);
                lSFAux(l)*=t;
            }

            lGamma+=pDoc.mCounts[n]*(pPhi(n,nAll)-lOldPhi).T();
            lDigammaGamma=digamma(lGamma);
        }
        
        //compute likelihood (was separate function)
        double lGammaSum=sum(lGamma);
        double lDigammaGammaSum=mat::digamma(lGammaSum);
        double lAlphaSum=K*mAlpha;
        double lNewLikelihood=lgamma(lAlphaSum)-lgamma(lGammaSum);
        double t=0.0;
        for(int k=0;k<K;k++)
        {
            //this can be simplified, but do it later
            lNewLikelihood+=-lgamma(mAlpha)+(mAlpha-1)*(lDigammaGamma[k]-lDigammaGammaSum)+
                          lgamma(lGamma(k))-(lGamma(k)-1)*(lDigammaGamma(k)-lDigammaGammaSum);
                          
            for(int n=0;n<pDoc.mWords.size();n++)
            {
                if(pPhi(n,k)>0)
                {
                    lNewLikelihood+=pDoc.mCounts[n]*(pPhi(n,k)*((lDigammaGamma(k)-lDigammaGammaSum)-log(pPhi(n,k))+mLogProbW(k,pDoc.mWords[n])));
                    if(pDoc.mLabel<mNumClasses-1)
                        t+=mMu(pDoc.mLabel,k)*pDoc.mCounts[n]*pPhi(n,k);
                }
            }
        }
        lNewLikelihood+=t/pDoc.mTotal;
        
        t=1.0;
        for(int l=0;l<mNumClasses-1;l++)
        {
            double lT1=1.0;
            for(int n=0;n<pDoc.mWords.size();n++)
            {
                double lT2=0.0;
                for(int k=0;k<K;k++)
                    lT2+=pPhi(n,k)*exp(mMu(l,k)*pDoc.mCounts[n]/pDoc.mTotal);
                lT1*=lT2;
            }
            t+=lT1;
        }
        lNewLikelihood-=log(t);
        
        double lConverged=abs((lLikelihood-lNewLikelihood)/lLikelihood);
        lLikelihood=lNewLikelihood;
        if(lConverged<sLDAVarConvergence) break;
    }
    
    return lLikelihood;
}

double CModel::InferenceLDA(const CDocument &pDoc,mat::CMat_d &pPhi)
{
    mat::CVec_d lGamma=mat::constant(K,1,mAlpha+pDoc.mTotal/double(K));
    mat::CVec_d lDigammaGamma=digamma(lGamma);
    pPhi=mat::constant(pDoc.mWords.size(),K,1.0/K);
    
    double lLikelihood=0.0;
    
    for(int lIter=0;lIter<sLDAVarMaxIter;lIter++)
    {
        for(int n=0;n<pDoc.mWords.size();n++)
        {
            mat::CRow_d lOldPhi=pPhi(n,nAll);
            double lPhiSum=0;
            for(int k=0;k<K;k++)
            {
                pPhi(n,k)=lDigammaGamma(k)+mLogProbW(k,pDoc.mWords[n]);
                
                if(k>0)
                    lPhiSum=log_sum(lPhiSum,pPhi(n,k));
                else
                    lPhiSum=pPhi(n,k);
            }
            
            for(int k=0;k<K;k++)
            {
                pPhi(n,k)=exp(pPhi(n,k)-lPhiSum);
                lGamma(k)+=pDoc.mCounts[n]*(pPhi(n,k)-lOldPhi(k));
            }
            lDigammaGamma=digamma(lGamma);
        }


        //compute likelihood (was separate function)
        double lGammaSum=sum(lGamma);
        double lDigammaGammaSum=mat::digamma(lGammaSum);
        double lAlphaSum=K*mAlpha;
        double lNewLikelihood=lgamma(lAlphaSum)-lgamma(lGammaSum);
        
        for(int k=0;k<K;k++)
        {
            lNewLikelihood+=-lgamma(mAlpha)+(mAlpha-1)*(lDigammaGamma[k]-lDigammaGammaSum)+
                          lgamma(lGamma(k))-(lGamma(k)-1)*(lDigammaGamma(k)-lDigammaGammaSum);
            
            for(int n=0;n<pDoc.mWords.size();n++)
            {
                if(pPhi(n,k)>0)
                {
                    lNewLikelihood+=pDoc.mCounts[n]*(pPhi(n,k)*((lDigammaGamma(k)-lDigammaGammaSum)-log(pPhi(n,k))+mLogProbW(k,pDoc.mWords[n])));
                }
            }
        }
        
        double lConverged=abs((lLikelihood-lNewLikelihood)/lLikelihood);
        lLikelihood=lNewLikelihood;
        if(lConverged<sLDAVarConvergence) break;
    }


    return lLikelihood;
}

void CModel::UpdateLogProbW()
{
    mat::CVec_d lLambdaSum=sum(rows(mLambda));
    for(int k=0;k<K;k++)
    {
        for(int w=0;w<W;w++)
        {
            if(mLambda(k,w)>0)
                mLogProbW(k,w)=log(mLambda(k,w))-log(lLambdaSum(k));
            else
                mLogProbW(k,w)=-100.0;
        }
    }
}

void CModel::UpdateMu(const CLabelStats &pSS,const CCorpus &pCorpus,const std::vector<int> &pDocIndices)
{
    CSoftMaxFunc lFunc(pSS,pCorpus,pDocIndices,sLDAL2Penalty);
    mat::CMat_d lX=mat::minimize_gsl_fdf(mMu(0,mNumClasses-1,nAll),lFunc,0.02,1e-4,1e-3,50);

    if (pDocIndices.size()==D)
        mMu(0,mNumClasses-1,nAll)=lX;
    else
        mMu(0,mNumClasses-1,nAll)=(1-mRhot)*mMu(0,mNumClasses-1,nAll)+mRhot*lX;
}

double CModel::DocEStepSLDA(const CDocument &pDoc,bool pUpdateMu,CLabelStats &pSS)
{
    mat::CMat_d lPhi;
    double lLikelihood;
    if(pUpdateMu)
        lLikelihood=InferenceSLDA(pDoc,lPhi);
    else
        lLikelihood=InferenceLDA(pDoc,lPhi);
    
    mat::CVec_d lBarM=mat::zeros(K);
    mat::CMat_d lBarVar=mat::zeros(K,K);
        
    for(int n=0;n<pDoc.mWords.size();n++)
    {
        for(int k=0;k<K;k++)
        {
            mLambda(k,pDoc.mWords[n])+=pDoc.mCounts[n]*lPhi(n,k);
            
            lBarM(k)+=lPhi(n,k)*pDoc.mCounts[n];
            for(int i=k;i<K;i++)
            {
                if(i==k)
                    lBarVar(k,i)+=lPhi(n,k)*pDoc.mCounts[n]*pDoc.mCounts[n];
                lBarVar(k,i)-=lPhi(n,k)*lPhi(n,i)*pDoc.mCounts[n]*pDoc.mCounts[n];
            }
        }
    }
    
    lBarM/=pDoc.mTotal;
    lBarVar/=sq(double(pDoc.mTotal));
    
    pSS.mBarsM.push_back(std::move(lBarM));
    pSS.mBarsVar.push_back(std::move(lBarVar));
        
    return lLikelihood;
}

double CModel::DocEStepLDA(const CDocument &pDoc,bool pUpdateMu,CLabelStats &pSS)
{
    mat::CMat_d lPhi;
    double lLikelihood;
    lLikelihood=InferenceLDA(pDoc,lPhi);
    
        
    for(int n=0;n<pDoc.mWords.size();n++)
    {
        for(int k=0;k<K;k++)
        {
            mLambda(k,pDoc.mWords[n])+=pDoc.mCounts[n]*lPhi(n,k);
            
        }
    }
    
        
    return lLikelihood;
}
void CModel::UpdateLambda(const CHDPStats &pSS,std::vector<int> &pWordList,bool pOptimalOrdering)
{
    // rhot will be between 0 and 1, and says how much to weight
    //the information we got from this mini-batch.
    
    mRhot=max(sRhotBound,mScale*pow(mTau+mUpdateCount,-mKappa));

    //Update appropriate columns of lambda based on documents.
        //cheng: lambada = (1-rhot)*lambda + rhot * ~lambda
        // eq 25 in Chong s paper
    for(int w=0;w<pWordList.size();w++)
        mLambda(nAll,pWordList[w])=mLambda(nAll,pWordList[w])*(1-mRhot)+mRhot*pSS.mVarBetaSS(nAll,w)*D/pSS.mBatchSize;

    ////////xavi: very important!!! lambdasum is no longer the sum of lambdas, because of lazy updates, so it must
    /// be updated according to the python code
    mLambdaSum=(1-mRhot)*mLambdaSum+mRhot*D*sum(rows(pSS.mVarBetaSS))/pSS.mBatchSize;

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

void CModel::UpdateLambdaAll(const CHDPStats &pSS,std::vector<int> &pWordList,bool pOptimalOrdering)
{
    // rhot will be between 0 and 1, and says how much to weight
    //the information we got from this mini-batch.
    
    //Update appropriate columns of lambda based on documents.
        //cheng: lambada = (1-rhot)*lambda + rhot * ~lambda
        // eq 25 in Chong s paper
    for(int w=0;w<pWordList.size();w++)
        mLambda(nAll,pWordList[w])=pSS.mVarBetaSS(nAll,w);
    
    mLambdaSum=sum(rows(mLambda));
    
    UpdateExpectations(true);

    //m_varphi_ss is a K sized vector. 
    mVarPhiSS=pSS.mVarSticksSS;
    
//    if(pOptimalOrdering)
//        OptimalOrdering();

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

void CModel::ProcessDocumentsOnlineHDP(const CCorpus &pCorpus,const std::vector<int> &pIndices)
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
    
    CHDPStats lSS(K,lWordList.size(),pIndices.size());
    CLabelStats lLSS;
    
    mELogSticks1st=expect_log_sticks(mVarSticks);

    //run variational inference on some new docs
    //cheng
    for(int i : pIndices)
    {
        msg_info() << "processing document " << i << "\n";
        const CDocument &lDoc=pCorpus.mDocs[i];

        DocEStepHDP(lDoc,lSS,lLSS,mELogSticks1st,lUniqueWords);
    }

    UpdateLambda(lSS,lWordList,true);

}

void CModel::ProcessDocumentsOnlineSHDP(const CCorpus &pCorpus,const std::vector<int> &pIndices)
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
    
    CHDPStats lSS(K,lWordList.size(),pIndices.size());
    CLabelStats lLSS;
    
    mELogSticks1st=expect_log_sticks(mVarSticks);

    //run variational inference on some new docs
    //cheng
    for(int i : pIndices)
    {
        msg_info() << "processing document " << i << "\n";
        const CDocument &lDoc=pCorpus.mDocs[i];

        DocEStepSHDP(lDoc,lSS,lLSS,mELogSticks1st,lUniqueWords,false);
    }

    UpdateLambda(lSS,lWordList,true);

    //update label part
    UpdateMu(lLSS,pCorpus,pIndices);
}
void CModel::ProcessDocumentsSHDP(const CCorpus &pCorpus)
{
    std::cout<<"here"<<std::endl;
    std::vector<int> lDocIndices(counter_iterator(0),counter_iterator((int)pCorpus.mDocs.size()));
    
    std::unordered_map<int, int> lUniqueWords;
    std::vector<int> lWordList;

    for(const CDocument &lDoc : pCorpus.mDocs)
    {
        for(int w : lDoc.mWords)
        {
            if(lUniqueWords.find(w)==lUniqueWords.end())
            {
                lUniqueWords[w]=lWordList.size();
                lWordList.push_back(w);
            }
        }
    }

    mELogSticks1st=expect_log_sticks(mVarSticks);

    double lLikelihood=0.0;
    
    for(int lIter=0;lIter<sHDPEMMaxIter;lIter++)
    {
        CHDPStats lSS(K,lWordList.size(),pCorpus.mDocs.size());
        CLabelStats lLSS;
    
        double lNewLikelihood=0.0;
        //Estep
        msg_info() << "e step\n";
        for(const CDocument &lDoc : pCorpus.mDocs){
           lNewLikelihood+=DocEStepSHDP(lDoc,lSS,lLSS,mELogSticks1st,lUniqueWords, lIter>3);
            
        }
        
        
        msg_info() << "likelihood:" << lNewLikelihood << "\n";
        
        //Mstep
        msg_info() << "m step\n";
        UpdateLambdaAll(lSS,lWordList,true);
        if(lIter>=3)
            UpdateMu(lLSS,pCorpus,lDocIndices);

        double lConverged=(lLikelihood-lNewLikelihood)/lLikelihood;
        lLikelihood=lNewLikelihood;
        if(lConverged<sHDPEMConvergence)
            break;
    }
 
    UpdateLogProbW();
}

void CModel::ProcessDocumentsHDP(const CCorpus &pCorpus)
{
    std::vector<int> lDocIndices(counter_iterator(0),counter_iterator((int)pCorpus.mDocs.size()));
    
    std::unordered_map<int, int> lUniqueWords;
    std::vector<int> lWordList;

    for(const CDocument &lDoc : pCorpus.mDocs)
    {
        for(int w : lDoc.mWords)
        {
            if(lUniqueWords.find(w)==lUniqueWords.end())
            {
                lUniqueWords[w]=lWordList.size();
                lWordList.push_back(w);
            }
        }
    }

    mELogSticks1st=expect_log_sticks(mVarSticks);

    double lLikelihood=0.0;
    
    for(int lIter=0;lIter<sHDPEMMaxIter;lIter++)
    {
        CHDPStats lSS(K,lWordList.size(),pCorpus.mDocs.size());
        CLabelStats lLSS;
    
        double lNewLikelihood=0.0;
        //Estep
        msg_info() << "e step\n";
        int dd=0;
        for(const CDocument &lDoc : pCorpus.mDocs){
            if(dd%1000 == 0)
                std::cout<<"d:"<<dd<<" "<<std::flush;
           lNewLikelihood+=DocEStepHDP(lDoc,lSS,lLSS,mELogSticks1st,lUniqueWords);
           dd++;
            
        }
        
        
        msg_info() << "likelihood:" << lNewLikelihood << "\n";
        
        //Mstep
        msg_info() << "m step\n";
        UpdateLambdaAll(lSS,lWordList,true);
        double lConverged=(lLikelihood-lNewLikelihood)/lLikelihood;
        lLikelihood=lNewLikelihood;
        if(lConverged<sHDPEMConvergence)
            break;
    }
 
    UpdateLogProbW();
    
    if(0){
        //write lambda to file
        std::string ELogProbW_filename = "HDP_K_"+ boost::lexical_cast<std::string>(K) + "alpha_"+ boost::lexical_cast<std::string>((int) (mAlpha*100))+ "_eta_"+ boost::lexical_cast<std::string>((int) (mEta*100))+ "_ELogProbW.txt";
        std::ofstream ELogProbW_file;
        ELogProbW_file.open(ELogProbW_filename, std::ios::app);
        if(ELogProbW_file.is_open()){
            for( int kk=0; kk<K; kk++){
                for ( int ww=0; ww< W; ww++){
                    ELogProbW_file << mLogProbW(kk, ww);
                    ELogProbW_file <<" ";
                }
                ELogProbW_file <<"\n";
            }

        }
        ELogProbW_file.close();
    }
}
void CModel::ProcessDocumentsLDA(const CCorpus &pCorpus)
{
    std::vector<int> lDocIndices(counter_iterator(0),counter_iterator((int)pCorpus.mDocs.size()));

    double lLikelihood=0.0;
    
    for(int lIter=0;lIter<sLDAEMMaxIter;lIter++)
    {
        CLabelStats lSS;
        mLambda=mat::zeros(K,W);
        double lNewLikelihood=0.0;

        msg_info() << "e step\n";
        
        for(int d=0;d<pCorpus.mDocs.size();d++)
        {
            if(d%1000==0)
                std::cout<<" d:"<<d<<std::flush;

            lNewLikelihood+=DocEStepLDA(pCorpus.mDocs[d],false,lSS);
        }

        msg_info() << "likelihood:" << std::setprecision(20) << lNewLikelihood << "\n";
        
        msg_info() << "m step\n";
        UpdateLogProbW();
        double lConverged=abs((lLikelihood-lNewLikelihood)/lLikelihood);
        lLikelihood=lNewLikelihood;
        if(lConverged<sLDAEMConvergence)
            break;
            if(0){
                std::string ELogProbW_filename = "LDA_K_"+ boost::lexical_cast<std::string>(K) + "alpha_"+ boost::lexical_cast<std::string>((int) (mAlpha*100))+ "_eta_"+ boost::lexical_cast<std::string>((int) (mEta*100))+ "_ELogProbW.txt";
                std::ofstream ELogProbW_file;
                ELogProbW_file.open(ELogProbW_filename, std::ios::app);
                if(ELogProbW_file.is_open()){
                    for( int kk=0; kk<K; kk++){
                        for ( int ww=0; ww< W; ww++){
                            ELogProbW_file << mLogProbW(kk, ww);
                            ELogProbW_file <<" ";
                        }
                        ELogProbW_file <<"\n";
                    }

                }
                ELogProbW_file.close();
            }
    }
}

void CModel::ProcessDocumentsSLDA(const CCorpus &pCorpus)
{
    std::vector<int> lDocIndices(counter_iterator(0),counter_iterator((int)pCorpus.mDocs.size()));

    double lLikelihood=0.0;
    
    for(int lIter=0;lIter<sLDAEMMaxIter;lIter++)
    {
        CLabelStats lSS;
        mLambda=mat::zeros(K,W);
        double lNewLikelihood=0.0;

        msg_info() << "e step\n";
        
        for(int d=0;d<pCorpus.mDocs.size();d++)
        {
            lNewLikelihood+=DocEStepSLDA(pCorpus.mDocs[d],lIter>=2,lSS);
        }

        msg_info() << "likelihood:" << std::setprecision(20) << lNewLikelihood << "\n";
        
        msg_info() << "m step\n";
        UpdateLogProbW();
        if(lIter>=2)
            UpdateMu(lSS,pCorpus,lDocIndices);
        
        double lConverged=abs((lLikelihood-lNewLikelihood)/lLikelihood);
        lLikelihood=lNewLikelihood;
        if(lConverged<sLDAEMConvergence)
            break;
    }
}

void CModel::UpdateExpectations(bool pAll)
{
    /*Since we're doing lazy updates on lambda, at any given moment 
    the current state of lambda may not be accurate. This function
    updates all of the elements of lambda and Elogbeta so that if (for
    example) we want to print out the topics we've learned we'll get the
    correct behavior.*/

    if(!pAll)
    {
        for (int w=0;w<W;w++)
        {
            //m_r<<m_r(m_r.tail(1)) + log(1-rhot);
            //m_timestamp(word_list[ww]) = m_updatect;
            mLambda(nAll,w)*=exp(mR.back()-mR[mTimestamps(w)]);
        }
    }
    
    mat::CMat_d lTT1=digamma(mEta+mLambda);
    mat::CVec_d lTT2=digamma(W*mEta+mLambdaSum);
    mELogBeta=lTT1-extend(lTT2);
    mTimestamps=mat::constant(W,1,(double)mUpdateCount);
}

void CModel::Print()
{
    UpdateExpectations(false);
    msg_info() << "lambda:" << mLambda << "\n\n";
    msg_info() << "mu:" << mMu << "\n";
    //    msg_info() << "elogbeta:" << mELogBeta << "\n\n";
    
    msg_info() << "lambdasum:" << mLambdaSum.T();
}

int CModel::Classify(const CDocument &pDoc)
{
    mat::CMat_d lPhi;
    InferenceLDA(pDoc,lPhi);
    mat::CRow_d lTopics=mat::zeros(1,K);
    for(int n=0;n<pDoc.mWords.size();n++)
        lTopics+=pDoc.mCounts[n]*lPhi(n,nAll);
    lTopics/=pDoc.mTotal;
    mat::CVec_d lScore=mMu*lTopics.T();
    return indmax(lScore).first;
}

int CModel::ClassifyHDP(const CDocument &pDoc)
{
    //e step for a single  document
    std::vector<int> lIDs;
    // batchids are the value in unique_words correspond to every word of the document
    // the value in the unique_words record the order of the words has been seen
    //eg. training corpus [2 8 6 2] unique_words will be 2:0 8:1 6:2
    for(int i : pDoc.mWords)
        lIDs.push_back(i);
    
    mat::CMat_d lELogBetaDoc=mELogBeta(nAll,pDoc.mWords);

    mat::CMat_d lPhi=mat::constant(pDoc.mWords.size(),T,1.0/T);
    mat::CMat_d lVarPhi=mat::constant(T,K,1.0/K);
    mat::CVec_d lMatCounts=mat::make_vec<double>(pDoc.mCounts);
    
    InferenceHDP(lPhi,lVarPhi,lELogBetaDoc,lMatCounts);

    mat::CRow_d lTopics=sum(cols((lPhi**lMatCounts)*lVarPhi))/pDoc.mTotal;
    mat::CVec_d lScore=mMu*lTopics.T();
    return indmax(lScore).first;
}

/*namespace hdp*/ } /*namespace buola*/ }
