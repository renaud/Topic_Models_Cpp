Topic_Models_Cpp
================

[Cheng Zhang](http://www.csc.kth.se/~chengz/TopicModelCode.html)'s Topic Models C++ library


# Install 

Tested on Ubuntu 12.04

## GCC 4.8

    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install gcc-4.8 g++-4.8
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50

Then add this to your .bashrc

    export CC=gcc-4.8
    export CXX=g++-4.8


## NLopt

A dependency from Buola

    cd /home/richarde/dev/bluebrain/lda/TopicModels_chengz
    wget http://ab-initio.mit.edu/nlopt/nlopt-2.3.tar.gz
    tar -xzf nlopt-2.3.tar.gz && rm nlopt-2.3.tar.gz
    cd nlopt-2.3
    ./configure
    make
    sudo make install

## Buola

A matrix manipulation libary (among others)

    cd /home/richarde/dev/bluebrain/lda/TopicModels_chengz
    sudo apt-get install libeigen3-dev libxml2-dev libdbus-1-dev libncurses5-dev
    wget http://www.csc.kth.se/~chengz/minibuola.tar.gz
    tar -xzf minibuola.tar.gz && rm minibuola.tar.gz
    cd minibuola
    vim include/buola/mat/detail/eigen_wrapper.h
        Changing from:
        #include <Eigen/Dense>
        to:
        #include "Eigen/Dense"
    mkdir build 
    cd build
    cmake ..
    make -j5
    sudo make install


## TopicModel 

    cd /home/richarde/dev/bluebrain/lda/TopicModels_chengz
    wget http://www.csc.kth.se/~chengz/TopicModel.tar.gz
    tar -xzf TopicModel.tar.gz && rm TopicModel.tar.gz
    cd TopicModel
    mkdir build
    cd build
    cmake ..
    make



# Play with Topic Models

## Dataset

[3class KTH action data for fun](http://www.csc.kth.se/~chengz/KTH.tar.gz)
This data is preprocessed with bag-of-STIP

## To check the options:

    ./TopicModel --help

## Example 1: SLDA

    ./TopicModel --slda --alpha 0.1 --corpus_name KTH --data YOURPATH/KTH/Train.dat --label YOURPATH/KTH/ImgLabel.txt --test YOURPATH/KTH/Test.dat --shuffle --num_classes 3   -k 30  --truth YOURPATH/KTH/GroundTruth.txt --seed 2

The result will be:

    0 0 2 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 1 1 1 2 1 1 1 1 1 1 2 1 2 2 2 2 2 2 2 2 1 1 2 2 2 2 2 0 2 2 2
    accuracy:0.84745762711864403016

## Example 2: SHDP

    ./TopicModel  --corpus_name KTH --data YOURPATH/KTH/Train.dat --label YOURPATH/KTH/ImgLabel.txt --test YOURPATH/KTH/Test.dat --shuffle --num_classes 3  -k 80 -t 20 --truth YOURPATH/KTH/GroundTruth.txt --seed 2

The result will be:

    0 0 0 2 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 1 1 2 2 2 2 2 0 0 2 2
    accuracy:0.864407

To use LDA  use `--lda`
To use HDP use `--hdp`. In this case the label document is not needed anymore
For the `onlineSHDP` and `onlineHDP`, it need large data to converge. So it does not work for the KTH data that we used here as example.


# References

- LDA:  D. M. Blei, A. Y. Ng, and M. I. Jordan. Latent Dirichlet Allocation. Journal of Machine Learning Research, 3:993–1022, 2003.
- SLDA: C. Wang, D. M. Blei, and L. Fei-Fei. Simultaneous image classification and annotation. In CVPR, 2009.
- HDP: Y. W. Teh, M. I. Jordan, M. J. Beal, and D. M. Blei. Hierarchical Dirichlet processes. Journal of the American Statistical Association, 101(476):1566–1581, 2006.
- SHDP&onlineSHDP:  C. Zhang, C.H. Ek, X. Gratal, F. Pokorny and H. Kjellström, Supervised Hierarchical Dirichlet Process with Variational Inference, In ICCV,2013
PS: The suplement of this paper gives the computation of the bound and update equation in detail. Recomand for beginners.
- OnlineHDP: C. Wang, J. Paisley, and D. Blei. Online variational inference for the Hierarchical Dirichlet Process. In AISTATS, 2011. 