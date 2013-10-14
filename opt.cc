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
#include "opt.h"

#include <iomanip>
/*
 * function to compute the value of the obj function, then 
 * return it
 */
namespace buola { namespace hdp {
    
int map_idx( int row, int col, int dim)
{
    if(row>col) std::swap(row,col);
    return (2*dim - row +1) * row/2 + col - row;
}

static double log_sum(double log_a, double log_b)
{
    if(log_a<log_b)
        return log_b + log( 1+ exp(log_a - log_b));
    else
        return log_a + log( 1+ exp(log_b - log_a));
}

CVarPhiFunc::CVarPhiFunc(int pNumClasses,int pLabel,int pTotal,const mat::CMat_d &pMu,const mat::CMat_d &pPhi,
                         const mat::CVec_d &pMatCounts,
                         const mat::CMat_d &pELogBetaDocCounts,const mat::CVec_d &pELogSticks1st,double pPenalty)
    :   mNumClasses(pNumClasses)
    ,   mLabel(pLabel)
    ,   mTotal(pTotal)
    ,   mMu(pMu)
    ,   mPhi(pPhi)
    ,   mMatCounts(pMatCounts)
    ,   mELogBetaDocCounts(pELogBetaDocCounts)
    ,   mELogSticks1st(pELogSticks1st)
//    ,   mPenalty(pPenalty)
{
}
    
double CVarPhiFunc::F(const mat::CMat_d& pX)
{
    //maybe we don't really need the varphi in the model, just use t and k and create a new one here!
    double lF=0.0;

    mat::CMat_d lVarPhi=exp(pX)/(exp(pX)+1);
    lVarPhi[lVarPhi<1e-100]=1e-100;
    lVarPhi[is_nan(lVarPhi)]=1e-100;
    
    mat::CVec_d lNormVarPhi=sum(rows(lVarPhi));
    lVarPhi/=extend(lNormVarPhi);

    //double lFRegularization=-sum(sq(lVarPhi))*mPenalty/2.0;

    lF+=sum((mPhi.T()*mELogBetaDocCounts.T())**lVarPhi);
    lF+=sum(lVarPhi**extend(mELogSticks1st.T()));
    lF-=sum(log(lVarPhi)**lVarPhi);

    mat::CRow_d lPVT=sum(cols((mPhi**extend(mMatCounts))*lVarPhi));
    lF+=lPVT*mMu(mLabel,nAll).T()/mTotal;

    double lT=0.0;

//     for (int l=0;l<mNumClasses;l++)
//     {
//         mat::CVec_d lTV=exp(mMu(l,nAll).T()/mTotal);
//         lT+=exp(sum((mPhi*lVarPhi*lTV)**extend(mMatCounts)));
// 
//         assert(!std::isnan(lT));
//     }

    mat::CVec_d lFTmp(mNumClasses);

    for(int l=0;l<mNumClasses;l++)
    {
        lFTmp(l)=1;
        for ( int n =0; n< mMatCounts.size(); n++){
        mat::CVec_d lTT=exp(mMu(l,nAll)/mTotal).T();
        lFTmp(l) *=pow(mPhi(n,nAll)*lVarPhi*lTT,mMatCounts[n]);
        }
    }

    lT=sum(lFTmp);
    
    if(lT<=0)
        throw XProcess("negative T in F");
    
    lF-=log(lT);
    
    if(isnan(lF))
    {
        msg_info() << "nan in lF\n";
        msg_info() << pX << "\n";
        msg_info() << lVarPhi << "\n";
        abort();
    }

    return -(lF);//+lFRegularization);
}

/*
 * function to compute the derivatives of function 
 *
 */

mat::CMat_d CVarPhiFunc::DF(const mat::CMat_d& pX)
{
    mat::CMat_d lDF(pX.Rows(),pX.Cols());
    double lF=F(pX);
    
    for(int i=0;i<pX.Rows();i++)
    {
        for(int j=0;j<pX.Cols();j++)
        {
            mat::CMat_d lNewX=pX;
            double lDelta=1e-10*pX(i,j);
            lNewX(i,j)+=lDelta;
            
            lDF(i,j)=(F(lNewX)-lF)/lDelta;
        }
    }
    
    return lDF;

#if 0
    mat::CMat_d lVarPhi=exp(pX)/(exp(pX)+1);
    
 
    lVarPhi[lVarPhi<1e-100]=1e-100;
    for ( int rr=0; rr< pX.Rows(); rr++)
        for ( int cc=0; cc< pX.Cols(); cc++)
            if(isnan(lVarPhi(rr,cc)))
                lVarPhi(rr,cc)=1;
    
    mat::CVec_d lNormVarPhi=sum(rows(lVarPhi));
    lVarPhi/=extend(lNormVarPhi);
  
    
    mat::CMat_d lDF=mat::zeros(lVarPhi.Rows(),lVarPhi.Cols()); //-mPenalty*lVarPhi;

    mat::CVec_d lFTmp(mNumClasses);

    for(int l=0;l<mNumClasses;l++)
    {
        lFTmp(l)=1;
        for ( int n =0; n< mMatCounts.size(); n++){
        mat::CVec_d lTT=exp(mMu(l,nAll)/mTotal).T();
        lFTmp(l) *=pow(mPhi(n,nAll)*lVarPhi*lTT,mMatCounts[n]);
        }
    }
    
    double lSumFTmp=sum(lFTmp);
    
    if(lFTmp(0)>10e30)
    {
        msg_info() << "mPhi:" << max(abs(mPhi)) << "\n";
        msg_info() << "lVarPhi:" << max(abs(lVarPhi)) << "\n";
        msg_info() << "mMu:" << max(abs(mMu)) << "\n";
    }
    
    //the first part as hdp
    mat::CMat_d lDFTmp=mPhi.T()*mELogBetaDocCounts.T()+extend(mELogSticks1st)+1-log(lVarPhi);
    
    for(int t=0;t<lDFTmp.Rows();t++)
    {
        for(int k=0;k<lDFTmp.Cols();k++)
        {
            //cheng
            double lMuPhi=0.0;
            lMuPhi=mMu(mLabel,k)*sum(mPhi(nAll,t)**mMatCounts)/mTotal;
            double lWL=0.0;
            for(int l=0;l<mNumClasses;l++)
            {   mat::CVec_d lNumerator(mMatCounts.size());
                mat::CVec_d lDenominator(mMatCounts.size());
                
                for ( int nn=0; nn< mMatCounts.size(); nn++){
                    lNumerator(nn)=(mPhi(nn,t)*mMatCounts[nn])*exp(mMu(l,k)/mTotal);
                    lDenominator(nn)=mPhi(nn,nAll)*lVarPhi*exp(mMu(l,nAll).T()/mTotal);
                }
                lWL+=lFTmp(l)*sum(lNumerator/lDenominator);
                
                if( isnan(sum(lNumerator/lDenominator))){
                    std::cout<<"lNumerator:"<<lNumerator<<std::endl;
                    std::cout<<"lDenominator"<<lDenominator<<std::endl;
                    std::cout<<"lVarphi"<<lVarPhi<<std::endl;
                    abort();
                }
            }

            lDF(t,k)+=(lDFTmp(t,k)+lMuPhi+lWL)/lSumFTmp;
        }
    }
   

    return -lDF;
#endif
}

double CVarPhiFunc::RowConstraint(const mat::CMat_d& pX,int pRow)
{
    return 1.0-sum(pX(pRow,nAll));
}

CSoftMaxFunc::CSoftMaxFunc(const CLabelStats &pSS,const CCorpus &pCorpus,const std::vector<int> &pIndices,double pPenalty)
    :   mSS(pSS)
    ,   mCorpus(pCorpus)
    ,   mIndices(pIndices)
    ,   mPenalty(pPenalty)
{
}

double CSoftMaxFunc::F(const mat::CMat_d& pX)
{
    double lF=0.0;
    //f_regularization
    double lFRegularization=-mPenalty/2.0*sum(sq(pX));
    
    for(int i=0;i<mIndices.size();i++)
    {
        const CDocument &lDoc=mCorpus.mDocs[mIndices[i]];
        const mat::CVec_d &lBarM=mSS.mBarsM[i];
        const mat::CMat_d &lBarVar=mSS.mBarsVar[i];
        
        for(int k=0;k<pX.Cols();k++)
        {
            if(lDoc.mLabel<pX.Rows())
                lF+=pX(lDoc.mLabel,k)*lBarM(k);
        }

        double t=0.0;
        for(int l=0;l<pX.Rows();l++)
        {
            double lA1=0.0;
            double lA2=0.0;
            for(int k=0;k<pX.Cols();k++)
            {
                lA1+=pX(l,k)*lBarM(k);
                for(int j=0;j<pX.Cols();j++)
                {
                    int r,c;
                    std::tie(r,c)=std::minmax(k,j);
                    lA2+=pX(l,k)*lBarVar(r,c)*pX(l,j);
                }
             }
             lA2=1.0+0.5*lA2;
             t=log_sum(t,lA1+log(lA2));
         }
         lF-=t;      
    }

    return -(lF+lFRegularization);
}

mat::CMat_d CSoftMaxFunc::DF(const mat::CMat_d& pX)
{
    mat::CMat_d lDF=-mPenalty*pX;
    
    for(int i=0;i<mIndices.size();i++)
    {
        const CDocument &lDoc=mCorpus.mDocs[mIndices[i]];
        const mat::CVec_d &lBarM=mSS.mBarsM[i];
        const mat::CMat_d &lBarVar=mSS.mBarsVar[i];

        for(int k=0;k<pX.Cols();k++)
        {
            if(lDoc.mLabel<pX.Rows())
                lDF(lDoc.mLabel,k)+=lBarM(k);
        }
        
        double t=0.0;
        mat::CMat_d lDFTmp=mat::zeros(pX.Rows(),pX.Cols());
        
        for(int l=0;l<pX.Rows();l++)
        {
            mat::CVec_d lEtaAux=mat::zeros(pX.Cols());
            double lA1=0.0;
            double lA2=0.0;
            for(int k=0;k<pX.Cols();k++)
            {
                lA1+=pX(l,k)*lBarM(k);
                for(int j=0;j<pX.Cols();j++)
                {
                    int r,c;
                    std::tie(r,c)=std::minmax(k,j);
                    lA2+=pX(l,k)*lBarVar(r,c)*pX(l,j);
                    lEtaAux(k)+=lBarVar(r,c)*pX(l,j);
                }
            }
            lA2=1.0+0.5*lA2;
            t=log_sum(t,lA1+log(lA2));
            
            for(int k=0;k<pX.Cols();k++)
            {
                lDFTmp(l,k)-=exp(lA1)*(lBarM(k)*lA2+lEtaAux(k));
            }
        }
        
        lDF+=lDFTmp*exp(-t);
    }
 
    return -lDF;
}

/*namespace hdp*/ } /*namespace buola*/ }
