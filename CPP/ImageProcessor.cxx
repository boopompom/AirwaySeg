#include "ImageProcessor.h"


using namespace std;
using namespace itk;

bool ImageProcessor::mIsJSONMapWritten = false;
mutex ImageProcessor::mJSONMutex;

void ImageProcessor::process() {

    IntensityImageType::SizeType imageSize = mIntensityImage->GetLargestPossibleRegion().GetSize();
    IntensityImageType::IndexType start;
    IntensityImageType::SizeType size;
    unsigned int offset = (mDiameter-1)/2;

    unsigned int shape_X[] = {mVOICount, mDiameter, mDiameter, mDiameter};
    unsigned int shape_Y[] = {mVOICount, mLabelCount};
    float* data_X = new float[mVOICount * mDiameter * mDiameter * mDiameter];
    int* data_Y = new int[mVOICount * mLabelCount];

    unsigned long offset_X = 0;
    unsigned long offset_Y = 0;
    unsigned int labelCounter = 0;
    unsigned int labelIdx = 0;
    for(auto labelIt = mEnabledLabelsList.begin(); labelIt != mEnabledLabelsList.end(); ++labelIt) {

        unsigned int label = *labelIt;

        vector<IntensityIndexType> idxList = mLabelMap[label];
        if(idxList.size() < mVOIPerLabel) {
            throw ExceptionObject("Insufficient VOIs for label");
        }

        for(int i=0;i<mVOIPerLabel; i++) {

            VOIFilterType::Pointer filter = VOIFilterType::New();

            auto idxIt = idxList.begin();
            int off = int(idxList.size() * (double)std::rand() / RAND_MAX);
            std::advance(idxIt, off);

            auto idx = *idxIt;
            start = idx;
            start[0] -= offset;
            start[1] -= offset;
            start[2] -= offset;

            size[0] = mDiameter;
            size[1] = mDiameter;
            size[2] = mDiameter;

            cout
                << "VOI Center: "
                << idx[0] << ", "
                << idx[1] << ", "
                << idx[2] << " | "
                << imageSize[0] - idx[0] << ", "
                << imageSize[1] - idx[1] << ", "
                << idx[2] << " | "
                << mIntensityImage->GetPixel(idx) << " | "
                << mLabelImage->GetPixel(idx) << endl;

            filter->SetInput(mIntensityImage);
            filter->SetRegionOfInterest(IntensityImageType::RegionType(start, size));
            try {
                filter->Update();
            } catch(ExceptionObject e) {
                //FIXME: Possibility of infinite loop
                i--;
                continue;
            }

            IntensityImageType::Pointer VOI = filter->GetOutput();
            ImgRegionIteratorType imIterator(VOI, VOI->GetLargestPossibleRegion());


            unsigned int counter = 0;
            for(imIterator.GoToBegin(); !imIterator.IsAtEnd(); ++imIterator) {
                data_X[offset_X + counter] = imIterator.Get();
                ++counter;
            }

            for(int j=0;j<mLabelCount;j++) {
                data_Y[offset_Y + j] = mEnabledLabelOneHot[label][j];
            }

            offset_X += mDiameter * mDiameter * mDiameter;
            offset_Y += mLabelCount;
        }
        labelCounter++;
        labelIdx++;
    }


    cnpy::npz_save(mOutputPath + "/" + mItemName + "_X.npz", mItemName, data_X, shape_X , 4, "w");
    cnpy::npz_save(mOutputPath + "/" + mItemName + "_Y.npz", mItemName, data_Y, shape_Y , 2, "w");

    delete data_X;
    delete data_Y;

    lock_guard<mutex> lock(mJSONMutex);
    if(!mIsJSONMapWritten) {
        stringstream ss;
        ss << "{" << endl;
        for(auto it=mEnabledLabelOneHot.begin(); it != mEnabledLabelOneHot.end(); ++it) {
            ss << "\t" << it->first <<  ": " << "[";
            for(int i=0; i<it->second.size(); ++i) {
                if(i != 0) {
                    ss << ", ";
                }
                ss << it->second[i];
            }
            ss << "]," << endl;
        }
        ss << "}" << endl;


        ofstream jsonFile;
        jsonFile.open (mOutputPath + "/class_map.json");
        jsonFile << ss.str();
        jsonFile.close();
        mIsJSONMapWritten = true;

        cout << "JSON class map file written" << endl;
    }


}

void ImageProcessor::loadLabelMap() {

    string intensityFilename = mInputPath + "/image.nrrd";
    IntensityReaderType::Pointer readerIntensity = IntensityReaderType::New();
    readerIntensity->SetFileName(intensityFilename);
    readerIntensity->Update();

    string labelFilename = mInputPath + "/label.nrrd";
    LabelReaderType::Pointer readerLabels = LabelReaderType::New();
    readerLabels->SetFileName(labelFilename);
    readerLabels->Update();

    mIntensityImage= readerIntensity->GetOutput();
    mLabelImage= readerLabels->GetOutput();

    LabelImageType::SizeType dim = mLabelImage->GetLargestPossibleRegion().GetSize();

    IntensityIndexType intensityIdx = IntensityIndexType();

    LblRegionIteratorType it(mLabelImage, mLabelImage->GetLargestPossibleRegion());
    for(it.GoToBegin(); !it.IsAtEnd(); ++it) {
        LabelPixelType label = it.Get();
        if(label == 0 || mEnabledLabelsLookup.find(label) == mEnabledLabelsLookup.end()) {
            continue;
        }
        mLabelMap[label].push_back(it.GetIndex());
    }
}
