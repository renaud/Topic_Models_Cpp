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

#ifndef _CORPUS_H_
#define _CORPUS_H_

#include <buola/buola.h>

namespace buola { namespace hdp {

// the class for a single document
class CDocument 
{
public:
    CDocument();
    explicit CDocument(const std::string &pLine);
    explicit CDocument(const std::string &pLine,const std::string &pLabelLine);
    CDocument(const CDocument&)=default;
    CDocument(CDocument&&)=default;
    
    CDocument &operator=(const CDocument&)=default;
    CDocument &operator=(CDocument&&)=default;
    
    void Init();
    
    std::vector<size_t> mWords;
    std::vector<size_t> mCounts;
    int mLabel;
    int mTotal; // sum of word counts
    int mMaxWord;
};

// the class for the whole corpus
class CCorpus
{
public:
    CCorpus();
    
    //read in data--the second read data 
    void Read(const std::string &pURI);
    void ReadWithLabel(const std::string &pData,const std::string &pLabel);
    //python code is out side of the class
    void Read(std::istream &pIn,int pCount);
    void ReadWithLabel(std::istream &pIn,std::istream &pLabelIn,int pCount);

    //python code is out side of the class
    int CountTokens(const std::string &pURI);
    
private:
    void ParseLine(const std::string &pLine,CDocument &pDoc);

    void Clear();
    
public:
    int V; //the vocab size. size_vocab in the ptython code
    std::vector<CDocument> mDocs; //all the documents in the corpus
    int mTotalNumDocs; // number of documents in the corpus, was D!!
    int mTotalWordCount; // total number of word counts in the corpus
};

/*namespace hdp*/ } /*namespace buola*/ }

#endif
