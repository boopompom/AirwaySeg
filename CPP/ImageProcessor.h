#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#define RAND_SEED 4324342

#include <vector>
#include <map>
#include <memory>
#include <fstream>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <mutex>

#include "cnpy.h"

using namespace std;
using namespace itk;


struct DICOMJob {
    vector<unsigned int> enabledLabels;
    unsigned int voiPerLabel;
    unsigned int diameter;
    string inputPath;
    string outputPath;
};

typedef float IntensityPixelType;
typedef unsigned int LabelPixelType;

typedef Image<IntensityPixelType, 3>  IntensityImageType;
typedef ImageFileReader<IntensityImageType> IntensityReaderType;
typedef IntensityImageType::IndexType IntensityIndexType;

typedef Image<LabelPixelType, 3>  LabelImageType;
typedef ImageFileReader<LabelImageType> LabelReaderType;
typedef LabelImageType::IndexType LabelIndexType;

typedef map<unsigned int, vector<IntensityIndexType> > LabelMapType;
typedef shared_ptr<LabelMapType> LabelMapPtrType;

typedef RegionOfInterestImageFilter< IntensityImageType, IntensityImageType > VOIFilterType;
typedef ImageRegionConstIterator< LabelImageType > LblRegionIteratorType;
typedef ImageRegionConstIterator< IntensityImageType > ImgRegionIteratorType;



class ImageProcessor {

public:
    ImageProcessor(DICOMJob job) {

        mInputPath = processPath(job.inputPath);
        mOutputPath = processPath(job.outputPath);
        mItemName = mInputPath.substr(mInputPath.rfind("/") + 1);

        mDiameter = job.diameter;
        mLabelCount = (unsigned int)job.enabledLabels.size();
        mVOIPerLabel = job.voiPerLabel;

        if(mDiameter % 2 == 0) {
            throw ExceptionObject("Even sized VOIs are not supported");
        }

        mEnabledLabelsList = job.enabledLabels;
        for(unsigned int i=0; i<mLabelCount; ++i) {
            mEnabledLabelsLookup[job.enabledLabels[i]] = true;
            mEnabledLabelOneHot[job.enabledLabels[i]].resize(mLabelCount);
            for(unsigned int j=0; j<mLabelCount; ++j) {
                unsigned int val = 0;
                if(i == j) {
                    val = 1;
                }
                mEnabledLabelOneHot[job.enabledLabels[i]][j] = val;
            }
        }

        mVOICount = mLabelCount * mVOIPerLabel;


        loadLabelMap();
    }

    void process();

private:

    string processPath(string path) {
        std::size_t found = path.rfind("/");
        if(found == string::npos) {
            throw ExceptionObject("Invalid Path");
        }

        string lastSeg = path.substr(found);
        if(lastSeg.length() == 1 && lastSeg[0] == '/') {
            path = path.substr(0, path.length() - 1);
        }
        return path;
    }

    void loadLabelMap();

    static bool mIsJSONMapWritten;
    static mutex mJSONMutex;

    string mInputPath;
    string mOutputPath;

    unsigned int mDiameter;
    unsigned int mLabelCount;
    unsigned int mVOICount;
    unsigned int mVOIPerLabel;

    LabelMapType mLabelMap;

    IntensityImageType::Pointer mIntensityImage;
    LabelImageType::Pointer mLabelImage;

    map<unsigned int, bool> mEnabledLabelsLookup;
    vector<unsigned int> mEnabledLabelsList;
    map<unsigned int, vector<unsigned int>> mEnabledLabelOneHot;

    string mItemName;
};

#endif
