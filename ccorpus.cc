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
 * TopticModel_C++ is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * TopticModel_C++ is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TopicModel_C++; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 */
 
#include "ccorpus.h"

#include <buola/zz.h>
#include <buola/zz/zinteger.h>
#include <buola/zz/zchar.h>
#include <buola/zz/if.h>
#include <buola/zz/operators.h>
#include <buola/functors/predicates/char.h>
#include <buola/io/lines.h>
#include <fstream>

namespace buola { namespace hdp {

CDocument::CDocument()
    :   mLabel(-1)
    ,   mTotal(0)
    ,   mMaxWord(-1)
{
}

CDocument::CDocument(const std::string &pLine)
    :    mLabel(-1)
{
    auto lWordsCounts=zz::tie_vector(mWords,mCounts);

    phrase_parse(pLine.begin(),pLine.end(),
                 zz::ZInteger<int>() >> *(zz::ZInteger<int>() >> ~zz::chr(':') >> zz::ZInteger<int>()),
                 zz::if_true(fn::is_space()),std::tie(std::ignore,lWordsCounts));

    mTotal=std::accumulate(mCounts.begin(),mCounts.end(),0);
    mMaxWord=*std::max_element(mWords.begin(),mWords.end());
}

CDocument::CDocument(const std::string &pLine,const std::string &pLabelLine)
    :   CDocument(pLine)
{
    mLabel=semantic_cast<int>(pLabelLine);
}

CCorpus::CCorpus()
    :   V(0)
    ,   mTotalNumDocs(0)
    ,   mTotalWordCount(0)
{
}

void CCorpus::Read(const std::string &pURI)
{
    Clear();

    std::ifstream lIn(pURI.c_str());

    for(const auto &lLine : io::lines_range(lIn))
    {
        mDocs.emplace_back(lLine);
        V=std::max(V,mDocs.back().mMaxWord);
        mTotalWordCount+=mDocs.back().mTotal;
    }

    mTotalNumDocs+=mDocs.size();
}
 
void CCorpus::ReadWithLabel(const std::string &pData,const std::string &pLabel)
{
    Clear();

    std::ifstream lData(pData.c_str());
    std::ifstream lLabel(pLabel.c_str());

    for(const auto &lLine : io::lines_range(lData))
    {
        std::string lLabelLine;
        getline(lLabel,lLabelLine);
        mDocs.emplace_back(lLine,lLabelLine);
        V=std::max(V,mDocs.back().mMaxWord);
        mTotalWordCount+=mDocs.back().mTotal;
    }

    mTotalNumDocs+=mDocs.size();
}
 
void CCorpus::Read(std::istream &pIn,int pCount)
{
    mDocs.clear();

    while(pCount--)
    {
        std::string lLine;
        getline(pIn,lLine);
        if(pIn.eof())
            break;
        
        mDocs.emplace_back(lLine);
        V=std::max(V,mDocs.back().mMaxWord);
        mTotalWordCount+=mDocs.back().mTotal;
    }

    mTotalNumDocs+=mDocs.size();
}


void CCorpus::ReadWithLabel(std::istream &pIn,std::istream &pLabelIn,int pCount)
{
    mDocs.clear();

    while(pCount--)
    {
        std::string lLine;
        getline(pIn,lLine);
        std::string lLabelLine;
        getline(pLabelIn,lLabelLine);
        if(pIn.eof()||pLabelIn.eof())
            break;
            
        mDocs.emplace_back(lLine,lLabelLine);
        V=std::max(V,mDocs.back().mMaxWord);
        mTotalWordCount+=mDocs.back().mTotal;
    }

    mTotalNumDocs+=mDocs.size();
}

void CCorpus::Clear()
{
    mDocs.clear();
    mTotalNumDocs=0;
}

/*namespace hdp*/ } /*namespace buola*/ }
