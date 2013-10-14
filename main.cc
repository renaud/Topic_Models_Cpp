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

#include "ccorpus.h"
#include "cmodel.h"
#include <buola/iterator/counter.h>
#include <buola/app/ccmdline.h>
#include <fstream>

using namespace buola;

buola::CCmdLineOption<int> sTOption('t',L"second level truncation",20);
buola::CCmdLineOption<int> sKOption('k',L"top level truncation",100);
buola::CCmdLineOption<int> sDOption('d',L"number of documents");
buola::CCmdLineOption<int> sWOption('w',L"size of vocabulary");
buola::CCmdLineOption<double> sEtaOption("eta",L"the topic/words Dirichlet parameter",0.5);
buola::CCmdLineOption<double> sAlphaOption("alpha",L"the corpus level beta parameter", 1);
buola::CCmdLineOption<double> sGammaOption("gamma",L"the document level beta parameter", 1);
buola::CCmdLineOption<double> sKappaOption("kappa",L"learning rate which is a parameter to compute rho",0.8);
buola::CCmdLineOption<double> sTauOption("tau",L"slow down which is a parameter to compute rho",20);
buola::CCmdLineOption<int> sSeedOption("seed",L"seed for random",2);
buola::CCmdLineOption<int> sBatchSizeOption("batchsize",L"size of each document batch",200);
buola::CCmdLineOption<int> sNumClassesOption("num_classes",L"number of classes",1);
buola::CCmdLineOption<int> sMaxTimeOption("max_time",L"maximum training time in seconds",86400);
buola::CCmdLineOption<int> sMaxIterOption("max_iter",L"maximum number of iterations for training");
buola::CCmdLineOption<std::string> sCorpusNameOption("corpus_name",L"name of the corpus");
buola::CCmdLineOption<std::string> sDataPathOption("data",L"path to training data",nRequired);
buola::CCmdLineOption<std::string> sLabelPathOption("label",L"path to labels for training data");
buola::CCmdLineOption<std::string> sTestPathOption("test",L"path to test data");
buola::CCmdLineOption<std::string> sTruthPathOption("truth",L"path to labels for test data");
buola::CCmdLineFlag sShuffleOption("shuffle",L"shuffle training documents");
buola::CCmdLineFlag sLDAOption("lda",L"use LDA instead of SHDP");
buola::CCmdLineFlag sHDPOption("hdp",L"use HDP instead of SHDP");
buola::CCmdLineFlag sSLDAOption("slda",L"use SLDA instead of SHDP");
buola::CCmdLineFlag sOnlineHDPOption("onlinehdp",L"use online HDP instead of SHDP");
buola::CCmdLineFlag sOnlineSHDPOption("onlineshdp",L"use online SHDP instead of SHDP");

int main(int pNArg,char **pArgs)
{
    buola_init(pNArg,pArgs);
    random::engine().seed(*sSeedOption);
    try 
    {
        hdp::CCorpus lTrain;
        if(sLDAOption || sHDPOption || sOnlineHDPOption){
            lTrain.Read(*sDataPathOption);
        }else{
            lTrain.ReadWithLabel(*sDataPathOption,*sLabelPathOption);
        }
        if(sShuffleOption)
            std::random_shuffle(lTrain.mDocs.begin(),lTrain.mDocs.end());
        
        //initialize the model 
        double lD=sDOption?*sDOption:lTrain.mTotalNumDocs;
        double lW=sWOption?*sWOption:(lTrain.V+1);
        
        if(sTestPathOption)
        {
            msg_info() << "making final prediction!!\n";
            hdp::CCorpus lTest;
            lTest.Read(*sTestPathOption);
            if(lTest.V>=lW)
                lW=lTest.V+1;
        }
        
        msg_info() << "initializing the model..." << "\n";
        hdp::CModel lModel(*sKOption,*sTOption,lD,lW,*sEtaOption,*sAlphaOption,*sGammaOption,
                           *sKappaOption,*sTauOption,*sNumClassesOption);

        int lFirstIndex=0;

        if(sSLDAOption)
        {
            lModel.ProcessDocumentsSLDA(lTrain);
        }
        else if(sLDAOption)
        {
            lModel.ProcessDocumentsLDA(lTrain);
        }
        else if(sHDPOption)
        {
            lModel.ProcessDocumentsHDP(lTrain);
        }
        else if(!sOnlineHDPOption && !sOnlineSHDPOption)
        {
            lModel.ProcessDocumentsSHDP(lTrain);
        }
        else
        {
            for(int lIter=0;;lIter++)
            {
                msg_info() << "iteration " << lIter << "\n";

                start_timer();

                std::vector<int> lIndices(counter_iterator(lFirstIndex),
                                        counter_iterator(std::min(lFirstIndex+*sBatchSizeOption,(int)lTrain.mDocs.size())));
                lFirstIndex+=*sBatchSizeOption;
                if(lIndices.empty())
                {
                    msg_info() << "no more documents in batch, break\n";
                    break;
                }

                //Do online inference and evaluate on the fly dataset
                msg_info() << "\t process documents..." << "\n";
                if ( sOnlineHDPOption ){
                    lModel.ProcessDocumentsOnlineHDP(lTrain,lIndices);
                }else{
                    lModel.ProcessDocumentsOnlineHDP(lTrain,lIndices);
                }
            }
        }

//        lModel.Print();
        
        if(sTestPathOption)
        {
            msg_info() << "making final prediction!!\n";
            hdp::CCorpus lTest;
            if(sTruthPathOption)
                lTest.ReadWithLabel(*sTestPathOption,*sTruthPathOption);
            else
                lTest.Read(*sTestPathOption);

            std::cout<<"\t working on fixed test data"<<std::endl;
//            double lTestScore=0;
//            double lTestScoreSplit=0;

            int lOk=0;
            
            for(const hdp::CDocument &lDoc : lTest.mDocs)
            {
//                lTestScore+=lModel.LDAEStep(lDoc);
                int L=lModel.Classify(lDoc);
//                lTestScoreSplit+=lModel.LDAEStepSplit(lDoc);
//                int L=lModel.Classification();
                msg_info() << L<<" ";
//                msg_info() << "split " << L << "\n";
                if(L==lDoc.mLabel) lOk++;
            }
            
            msg_info() << "\n";
            
            if(sTruthPathOption)
                msg_info() << "accuracy:" << double(lOk)/lTest.mDocs.size() << "\n";
        }
    }
    catch(std::exception &pE)
    {
        msg_info() << "caught exception in main:" << pE.what() << "\n";
    }
    
    return buola_finish();
}
