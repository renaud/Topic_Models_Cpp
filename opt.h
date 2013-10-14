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
#ifndef OPT_H_INCLUDED
#define OPT_H_INCLUDED

#include "cmodel.h"
/*
 * structure for the gsl optimization routine
 *
 */
namespace buola { namespace hdp {
    
class CVarPhiFunc
{
public:
    CVarPhiFunc(int pNumClasses,int pLabel,int pTotal,const mat::CMat_d &pMu,const mat::CMat_d &pPhi,const mat::CVec_d &pMatCounts,
                const mat::CMat_d &pELogBetaDocCounts,const mat::CVec_d &pELogSticks1st,double pPenalty);
    
    double F(const mat::CMat_d &pX);
    mat::CMat_d DF(const mat::CMat_d &pX);
    double RowConstraint(const mat::CMat_d &pX,int pRow);
    
    std::pair<double,mat::CMat_d> FDF(const mat::CMat_d &pX)
    {
        return {F(pX),DF(pX)};
    }

private:
    int mNumClasses;
    int mLabel;
    int mTotal;
    const mat::CMat_d &mMu;
    const mat::CMat_d &mPhi;
    const mat::CVec_d &mMatCounts;
    const mat::CMat_d &mELogBetaDocCounts;
    const mat::CVec_d &mELogSticks1st;
//    double mPenalty;
};

class CSoftMaxFunc
{
public:
    CSoftMaxFunc(const CLabelStats &pSS,const CCorpus &pCorpus,const std::vector<int> &pIndices,double pPenalty);
    
    double F(const mat::CMat_d &pX);
    mat::CMat_d DF(const mat::CMat_d &pX);
    
    std::pair<double,mat::CMat_d> FDF(const mat::CMat_d &pX)
    {
        return {F(pX),DF(pX)};
    }

private:
    const CLabelStats &mSS;
    const CCorpus &mCorpus;
    const std::vector<int> &mIndices;
    double mPenalty;
};

/*namespace hdp*/ } /*namespace buola*/ }

#endif // OPT_H_INCLUDED
