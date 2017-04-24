#include "ImageProcessor.h"


using namespace std;
using namespace itk;

bool ImageProcessor::mIsJSONMapWritten = false;
mutex ImageProcessor::mJSONMutex;

void ImageProcessor::process() {

	IntensityImageType::Pointer images[4] = {
		mIntensityImage,
		mChannels[0],
		mChannels[1],
		mChannels[2],
	};
    IntensityImageType::SizeType imageSize = mIntensityImage->GetLargestPossibleRegion().GetSize();
    IntensityImageType::IndexType start;
    IntensityImageType::SizeType size;
    unsigned int offset = (mDiameter-1)/2;

	unsigned int channels = 1;
	unsigned int shape_X_size = 5;
	unsigned int shape_X_wChannels[]  = { mVOICount, channels, mDiameter, mDiameter, mDiameter };
	unsigned int shape_X_woChannels[] = { mVOICount, mDiameter, mDiameter, mDiameter };
	if (channels == 1) {
		shape_X_size = 4;
	}

    unsigned int shape_Y[] = {mVOICount, mLabelCount};
    float* data_X = new float[mVOICount * channels * mDiameter * mDiameter * mDiameter];
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

            std::cout
                << "VOI Center: "
                << idx[0] << ", "
                << idx[1] << ", "
                << idx[2] << " | "
                << imageSize[0] - idx[0] << ", "
                << imageSize[1] - idx[1] << ", "
                << idx[2] << " | "
                << mIntensityImage->GetPixel(idx) << " | "
                << mLabelImage->GetPixel(idx) << endl;

			for (int i = 0; i < channels; i++) {
				VOIFilterType::Pointer filter = VOIFilterType::New();
				filter = VOIFilterType::New();
				filter->SetInput(images[i]);
				filter->SetRegionOfInterest(IntensityImageType::RegionType(start, size));
				filter->Update();

				IntensityImageType::Pointer VOI = filter->GetOutput();
				ImgRegionIteratorType imIterator(VOI, VOI->GetLargestPossibleRegion());

				unsigned int counter = 0;
				for (imIterator.GoToBegin(); !imIterator.IsAtEnd(); ++imIterator) {
					data_X[(mDiameter * mDiameter * mDiameter * i) + offset_X + counter] = imIterator.Get();
					++counter;
				}

			}


            for(int j=0;j<mLabelCount;j++) {
                data_Y[offset_Y + j] = mEnabledLabelOneHot[label][j];
            }

            offset_X += mDiameter * mDiameter * mDiameter * channels;
            offset_Y += mLabelCount;
        }
        labelCounter++;
        labelIdx++;
    }


    cnpy::npz_save(mOutputPath + "/" + mItemName + "_X.npz", mItemName, data_X, channels==1?shape_X_woChannels: shape_X_wChannels, shape_X_size, "w");
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

	SubtractLabelFilterType::Pointer subtractFilter = SubtractLabelFilterType::New();
	AddLabelFilterType::Pointer addFilter = AddLabelFilterType::New();

	//Read Labels
	string airwayLabelFilename = mInputPath + "/ZUNU_vida-aircolor.img.gz";
	string lungLabelFilename = mInputPath + "/ZUNU_vida-lung.img.gz";

	LabelReaderType::Pointer airwayReader = LabelReaderType::New();
	LabelReaderType::Pointer lungReader = LabelReaderType::New();

	airwayReader->SetFileName(airwayLabelFilename);
	airwayReader->Update();
	LabelImageType::Pointer airwayImage = airwayReader->GetOutput();

	lungReader->SetFileName(lungLabelFilename);
	lungReader->Update();
	LabelImageType::Pointer lungImage = lungReader->GetOutput();

	LblEditIteratorType it;

	it = LblEditIteratorType(lungImage, lungImage->GetLargestPossibleRegion());
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
		LabelPixelType label = it.Get();
		if (label == 30 || label == 20) {
			it.Set(1);
		}
	}
	subtractFilter->SetInput1(lungImage);
	subtractFilter->SetInput2(airwayImage);
	subtractFilter->Update();
	lungImage = subtractFilter->GetOutput();

	it = LblEditIteratorType(lungImage, lungImage->GetLargestPossibleRegion());
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
		LabelPixelType label = it.Get();
		if (label > 0) {
			it.Set(500);
		} else {
			it.Set(0);
		}
	}

	addFilter->SetInput1(lungImage);
	addFilter->SetInput2(airwayImage);
	addFilter->Update();
	mLabelImage = addFilter->GetOutput();

	//LabelWriterType::Pointer writerLabels = LabelWriterType::New();
	//writerLabels->SetFileName("C:/Projects/AirwaySegmentation/Output/test.nrrd");
	//writerLabels->SetInput(mLabelImage);
	//writerLabels->Update();

}
void ImageProcessor::init() {

	typedef GDCMImageIO          ImageIOType;
	typedef GDCMSeriesFileNames  NamesGeneratorType;

	//Read DICOM
	string intensityDirname = mInputPath + "/dicom/";
	ImageIOType::Pointer gdcmIO = ImageIOType::New();
	NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
	IntensityReaderType::Pointer readerIntensity = IntensityReaderType::New();
	namesGenerator->SetInputDirectory(intensityDirname);
	readerIntensity->SetImageIO(gdcmIO);
	readerIntensity->SetFileNames(namesGenerator->GetInputFileNames());
    readerIntensity->Update();

	mIntensityImage = readerIntensity->GetOutput();

	/*
	GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
	gaussianFilter->SetInput(mIntensityImage);
	gaussianFilter->SetVariance(5.0);

	DerivativeFilterType::Pointer xDerivativeFilter = DerivativeFilterType::New();
	DerivativeFilterType::Pointer yDerivativeFilter = DerivativeFilterType::New();
	DerivativeFilterType::Pointer zDerivativeFilter = DerivativeFilterType::New();

	xDerivativeFilter->SetDirection(0);
	yDerivativeFilter->SetDirection(1);
	zDerivativeFilter->SetDirection(2);

	xDerivativeFilter->SetInput(gaussianFilter->GetOutput());
	yDerivativeFilter->SetInput(gaussianFilter->GetOutput());
	zDerivativeFilter->SetInput(gaussianFilter->GetOutput());

	xDerivativeFilter->Update();
	yDerivativeFilter->Update();
	zDerivativeFilter->Update();

	mChannels[0] = xDerivativeFilter->GetOutput();
	mChannels[1] = yDerivativeFilter->GetOutput();
	mChannels[2] = zDerivativeFilter->GetOutput();
	*/

	loadLabelMap();

    
    LabelImageType::SizeType dim = mLabelImage->GetLargestPossibleRegion().GetSize();

    IntensityIndexType intensityIdx = IntensityIndexType();

	long loadedVOIs = 0;
	LblRegionIteratorType it(mLabelImage, mLabelImage->GetLargestPossibleRegion());
    for(it.GoToBegin(); !it.IsAtEnd(); ++it) {

        LabelPixelType label = it.Get();

		if (loadedVOIs >= mVOICount * 5) {
			break;
		}

		//Discard label if it is not in the enabled list or if we already sampled 5 times the VOI per label number
		if(label == 0 || mEnabledLabelsLookup.find(label) == mEnabledLabelsLookup.end() || mLabelMap[label].size() > mVOIPerLabel * 5) {
			continue;
        }
        mLabelMap[label].push_back(it.GetIndex());
		++loadedVOIs;

    }

	cout << "Done" << endl;
}
