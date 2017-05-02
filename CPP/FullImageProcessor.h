#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#define RAND_SEED 4324342

#include <mutex>
#include <vector>
#include <map>
#include <memory>
#include <fstream>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageSeriesReader.h>

#include "tinyxml/tinyxml.h"

#include <itkRegionOfInterestImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRandomConstIteratorWithIndex.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkChangeLabelImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkDerivativeImageFilter.h>


#include "cnpy.h"

using namespace std;
using namespace itk;


struct DICOMJob {
    vector<unsigned int> enabledLabels;
    unsigned int voiPerLabel;
    unsigned int diameter;
    string inputPath;
    string outputPath;
	bool isBinary;
};

typedef float IntensityPixelType;
typedef int LabelPixelType;

typedef Image<IntensityPixelType, 3>  IntensityImageType;
typedef ImageSeriesReader<IntensityImageType> IntensityReaderType;
typedef IntensityImageType::IndexType IntensityIndexType;

typedef Image<LabelPixelType, 3>  LabelImageType;
typedef ImageFileReader<LabelImageType> LabelReaderType;
typedef ImageFileWriter<LabelImageType> LabelWriterType;
typedef LabelImageType::IndexType LabelIndexType;

typedef map<unsigned int, vector<IntensityIndexType> > LabelMapType;
typedef shared_ptr<LabelMapType> LabelMapPtrType;

typedef RegionOfInterestImageFilter< IntensityImageType, IntensityImageType > VOIFilterType;
typedef ChangeLabelImageFilter< LabelImageType, LabelImageType > ChangeLabelFilterType;
typedef SubtractImageFilter< LabelImageType, LabelImageType > SubtractLabelFilterType;
typedef AddImageFilter< LabelImageType, LabelImageType > AddLabelFilterType;
typedef DerivativeImageFilter<IntensityImageType, IntensityImageType> DerivativeFilterType;
typedef DiscreteGaussianImageFilter<IntensityImageType, IntensityImageType> GaussianFilterType;


typedef ImageRegionConstIterator< LabelImageType > LblRegionIteratorType;
//typedef ImageRandomConstIteratorWithIndex< LabelImageType > LblRegionIteratorType;
typedef ImageRegionIterator< LabelImageType > LblEditIteratorType;
typedef ImageRegionConstIterator< IntensityImageType > ImgRegionIteratorType;


#define BooleanMacro(name, def) \
protected: \
	boolean m##name = ##def; \
public: \
	virtual void Set##name##On () { this->m##name = true;  } \
	virtual void Set##name##Off () { this->m##name = false;  } \
	virtual void Set##name (boolean val) { this->m##name = val;  } 


#define SetGetMacro(name, type, def) \
protected: \
	type m##name = ##def; \
public: \
	virtual void Set##name(type val) { this->m##name = val;  } \
	virtual type Get##name() { return this->m##name; }



class FullImageProcessor {

public:
    FullImageProcessor(DICOMJob job) {

        mInputPath = processPath(job.inputPath);
        mOutputPath = processPath(job.outputPath);
        mItemName = mInputPath.substr(mInputPath.rfind("/") + 1);

        mDiameter = job.diameter;
		mDiagonal = int(round(sqrt((mDiameter*mDiameter) + (mDiameter*mDiameter) + (mDiameter*mDiameter))));
		if (mDiagonal % 2 == 0) {
			mDiagonal--;
		}

        mLabelCount = (unsigned int)job.enabledLabels.size();
        mVOIPerLabel = job.voiPerLabel;

        if(mDiameter % 2 == 0) {
            throw ExceptionObject("Even sized VOIs are not supported");
        }
		
		mIsBinary = job.isBinary;
		if (mIsBinary) {
			job.enabledLabels.clear();
			job.enabledLabels.push_back(0);
			job.enabledLabels.push_back(255);
			mLabelCount = 2;
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
		
		mOffsetFromCenter = ((mDiagonal - 1) / 2);
		if (mIsBinary) {
			mReferenceLabelMap["Airyway"] = 255;
			mReferenceLabelMap["Background"] = 0;
		} else {
			mReferenceLabelMap["Trachea"] = 1;
			mReferenceLabelMap["BronInt"] = 7;
			mReferenceLabelMap["RMB"] = 3;
			mReferenceLabelMap["RB1"] = 25;
			mReferenceLabelMap["RB2"] = 45;
			mReferenceLabelMap["RB3"] = 44;
			mReferenceLabelMap["RB4"] = 48;
			mReferenceLabelMap["RB5"] = 49;
			//mReferenceLabelMap["RB4+5"] = 26;
			mReferenceLabelMap["RB6"] = 51;
			mReferenceLabelMap["RB7"] = 92;
			mReferenceLabelMap["RB8"] = 137;
			mReferenceLabelMap["RB9"] = 188;
			mReferenceLabelMap["RB10"] = 189;
			mReferenceLabelMap["RLL7"] = 27;
			mReferenceLabelMap["RUL"] = 6;
			mReferenceLabelMap["RLL"] = 91;
			/*
			mReferenceLabelMap["LMB"] = 2;
			mReferenceLabelMap["LB1"] = 43;
			mReferenceLabelMap["LB2"] = 42;
			mReferenceLabelMap["LB1+2"] = 23;
			mReferenceLabelMap["LB3"] = 22;
			mReferenceLabelMap["LB4"] = 20;
			mReferenceLabelMap["LB5"] = 21;
			mReferenceLabelMap["LB4+5"] = 10;
			mReferenceLabelMap["LB6"] = 9;
			mReferenceLabelMap["LB8"] = 17;
			mReferenceLabelMap["LB9"] = 28;
			mReferenceLabelMap["LLB6"] = 4;
			mReferenceLabelMap["LB10"] = 29;
			mReferenceLabelMap["LUL"] = 5;
			mReferenceLabelMap["LLB"] = 8;
			*/
		}


		for (auto it = mReferenceLabelMap.begin(); it != mReferenceLabelMap.end(); ++it) {
			mRevReferenceLabelMap[it->second] = it->first;
		}

        init();
    }

    void saveNpz();
	void saveIndices();

private:

	void writeJSON();
	string processPath(string path) {
		for (auto it = path.begin(); it != path.end(); ++it) {
			char x = *it;
			if (x == '\\') {
				*it = '/';
			}
		}
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
	void genReplacementLabelMap();
	void init();

    static bool mIsJSONMapWritten;
    static mutex mJSONMutex;

    string mInputPath;
    string mOutputPath;

	unsigned int mDiagonal;
    unsigned int mDiameter;
    unsigned int mLabelCount;
    unsigned int mVOICount;
    unsigned int mVOIPerLabel;
	int mOffsetFromCenter;

    LabelMapType mLabelMap;

	IntensityImageType::Pointer mIntensityImage;
	IntensityImageType::Pointer mChannels[3];
	LabelImageType::Pointer mLabelImage;

	map<string, unsigned int> mReferenceLabelMap;
	map<unsigned int, string> mRevReferenceLabelMap;

	map<string, unsigned int> mRepLabelMap;
	map<unsigned int, string> mRevRepLabelMap;


    map<unsigned int, bool> mEnabledLabelsLookup;
    vector<unsigned int> mEnabledLabelsList;
    map<unsigned int, vector<unsigned int>> mEnabledLabelOneHot;

    string mItemName;
	bool mIsBinary;
};

#endif
