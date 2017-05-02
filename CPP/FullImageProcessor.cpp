#include "FullImageProcessor.h"


using namespace std;
using namespace itk;

bool FullImageProcessor::mIsJSONMapWritten = false;
mutex FullImageProcessor::mJSONMutex;

void FullImageProcessor::genReplacementLabelMap() {

	string xmlPath = mInputPath + string("/ZUNU_vida-xmlTree.xml");
	TiXmlDocument doc( xmlPath.c_str() );
	doc.LoadFile();

	TiXmlElement* root = doc.FirstChildElement("TreeFile");
	TiXmlElement* segNames = root->FirstChildElement("SegmentNames");

	for (TiXmlElement* e = segNames->FirstChildElement("SegmentName"); e != NULL; e = e->NextSiblingElement("SegmentName")) {
		string segName = e->Attribute("anatomicalName");
		unsigned int segLabel = (unsigned int)stoi(e->Attribute("linkIds"));
		mRepLabelMap[segName] = segLabel;
		mRevRepLabelMap[segLabel] = segName;
	}
	std::cout << "Done Rep Map" << endl;
}


void FullImageProcessor::saveIndices() {
	//TODO: Store indices and leave VOI extraction and rotation to the VOI Queue
	writeJSON();

	IntensityImageType::SizeType imageSize = mLabelImage->GetLargestPossibleRegion().GetSize();

	unsigned long offset_X = 0;
	unsigned long offset_Y = 0;
	unsigned int labelCounter = 0;
	unsigned int labelIdx = 0;
	stringstream vois;
	
	vois << "{" << endl;
	vois << "\t\"id\": \"" << mItemName << "\"," << endl;
	vois << "\t\"dataset_path\": \"" << mInputPath << "\"," << endl;
	vois << "\t\"dicom_path\": \"" << mInputPath + "/dicom/" << "\"," << endl;
	vois << "\t\"vois\": [" << endl;


	for (auto labelIt = mEnabledLabelsList.begin(); labelIt != mEnabledLabelsList.end(); ++labelIt) {

		unsigned int label = *labelIt;

		vector<IntensityIndexType> idxList = mLabelMap[label];

		unsigned int labelsToLookFor = min<unsigned int>(idxList.size(), mVOIPerLabel);
		if (idxList.size() < mVOIPerLabel) {
			std::cerr << "Insufficient VOIs for label " << label << " in path " << mInputPath << endl;
		}

		for (int i = 0; i < labelsToLookFor; i++) {
			bool isFirstVOI = i == 0 && labelIt == mEnabledLabelsList.begin();
			if (!isFirstVOI) {
				vois << ", " << endl;
			}
			auto idxIt = idxList.begin();
			int off = int(idxList.size() * (double)std::rand() / RAND_MAX);
			std::advance(idxIt, off);

			string labelText = mRevReferenceLabelMap[label];
			if (label == 0) {
				labelText = "Background";
			}
			if (label == 500) {
				labelText = "Lung";
			}
			IntensityImageType::IndexType center = *idxIt;
			vois << "\t\t{" << endl;
			vois << "\t\t\t\"lbl_idx\": [" << center[0] << ", " << center[1] << ", " << center[2] << "]," << endl;
			vois << "\t\t\t\"int_idx\": [" << center[0] << ", " << center[1] << ", " << imageSize[2] - center[2] << "]," << endl;
			vois << "\t\t\t\"cls\": " << label << "," << endl;
			if (!mIsBinary && label != 0 && label != 500) {
				vois << "\t\t\t\"org_cls\": " << mRepLabelMap[labelText] << "," << endl;
			}
			vois << "\t\t\t\"cls_name\": \"" << labelText << "\"," << endl;
			vois << "\t\t\t\"cls_arr\": [";

			int counter = 0;
			for (auto it = mEnabledLabelOneHot[label].begin(); it != mEnabledLabelOneHot[label].end(); ++it) {
				unsigned int v = *it;
				if (counter != 0) {
					vois << ", ";
				}
				vois << v;
				counter++;
			}
			vois << "]" << endl;
			vois << "\t\t}";
		}
	}

	vois << endl << "\t]" << endl << "}" << endl;

	ofstream jsonFile;
	jsonFile.open(mOutputPath + "/" + mItemName + ".json");
	jsonFile << vois.str();
	jsonFile.close();

	//cout << ss.str();
}

void FullImageProcessor::saveNpz() {

	writeJSON();

	IntensityImageType::Pointer images[4] = {
		mIntensityImage,
		mChannels[0],
		mChannels[1],
		mChannels[2],
	};
    IntensityImageType::SizeType imageSize = mIntensityImage->GetLargestPossibleRegion().GetSize();
    IntensityImageType::IndexType start;
    IntensityImageType::SizeType size;
    
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
        
		unsigned int labelsToLookFor = min<unsigned int>(idxList.size(), mVOIPerLabel);
		if(idxList.size() < mVOIPerLabel) {
			std::cerr << "Insufficient VOIs for label " << label << " in path " << mInputPath << endl;
        }

		for (int i = 0; i < labelsToLookFor; i++) {


			auto idxIt = idxList.begin();
			int off = int(idxList.size() * (double)std::rand() / RAND_MAX);
			std::advance(idxIt, off);

			auto idx = *idxIt;
			start = idx;
			start[0] -= mOffsetFromCenter;
			start[1] -= mOffsetFromCenter;
			start[2] -= mOffsetFromCenter;

            size[0] = mDiameter;
            size[1] = mDiameter;
            size[2] = mDiameter;

			/*
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
			*/
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




}



void FullImageProcessor::writeJSON() {
	lock_guard<mutex> lock(mJSONMutex);
	if (!mIsJSONMapWritten) {
		stringstream ss;
		
		ss << "{" << endl;
		for (auto labelIt = mEnabledLabelsList.begin(); labelIt != mEnabledLabelsList.end(); ++labelIt) {
			if (labelIt != mEnabledLabelsList.begin()) {
				ss << ", " << endl;
			}

			unsigned int label = *labelIt;

			ss << "\t\"" << label << "\": {" << endl;

			string labelText = mRevReferenceLabelMap[label];
			if (label == 0) {
				labelText = "Background";
			}
			if (label == 500) {
				labelText = "Lung";
			}
			ss << "\t\t\"cls\": " << label << "," << endl;
			if (!mIsBinary && label != 0 && label != 500) {
				ss << "\t\t\"org_cls\": " << mRepLabelMap[labelText] << "," << endl;
			}
			ss << "\t\t\"cls_name\": \"" << labelText << "\"," << endl;
			ss << "\t\t\"cls_arr\": [";

			int cls_idx = -1;
			int counter = 0;
			for (auto it = mEnabledLabelOneHot[label].begin(); it != mEnabledLabelOneHot[label].end(); ++it) {
				unsigned int v = *it;
				if (counter != 0) {
					ss << ", ";
				}
				if (v == 1) {
					cls_idx = counter;
				}
				ss << v;
				counter++;
			}
			ss << "], " << endl;
			ss << "\t\t\"cls_idx\": " << cls_idx << endl;
			ss << "\t}" << endl;
			
		}
		ss << "}" << endl;


		//ss << endl << "\t]" << endl << "}" << endl;

		ofstream jsonFile;
		jsonFile.open(mOutputPath + "/class_map.json");
		jsonFile << ss.str();
		jsonFile.close();
		mIsJSONMapWritten = true;

		cout << "JSON class map file written" << endl;
	}
}
void FullImageProcessor::loadLabelMap() {

	if (!mIsBinary) {
		genReplacementLabelMap();
	}

	writeJSON();

	SubtractLabelFilterType::Pointer subtractFilter = SubtractLabelFilterType::New();
	AddLabelFilterType::Pointer addFilter = AddLabelFilterType::New();

	//Read Labels
	string airwayLabelFilename = mInputPath + "/ZUNU_vida-aircolor.img.gz";
	string airwayBinaryFilename = mInputPath + "/ZUNU_vida-airtree.img.gz";
	string lungLabelFilename = mInputPath + "/ZUNU_vida-lung.img.gz";

	LabelReaderType::Pointer airwayReader = LabelReaderType::New();
	LabelReaderType::Pointer lungReader = LabelReaderType::New();

	if (mIsBinary) {
		airwayReader->SetFileName(airwayBinaryFilename);
		airwayReader->Update();
		mLabelImage = airwayReader->GetOutput();
		return;
	} else {
		airwayReader->SetFileName(airwayLabelFilename);
		airwayReader->Update();
	}
	LabelImageType::Pointer airwayImage = airwayReader->GetOutput();

	lungReader->SetFileName(lungLabelFilename);
	lungReader->Update();
	LabelImageType::Pointer lungImage = lungReader->GetOutput();

	LblEditIteratorType it;

	it = LblEditIteratorType(airwayImage, airwayImage->GetLargestPossibleRegion());
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
		LabelPixelType label = it.Get();
		if (label == 0) {
			continue;
		}
		string segName = mRevRepLabelMap[label];
		LabelPixelType newLabel = mReferenceLabelMap[segName];
		it.Set(newLabel);
	}

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
void FullImageProcessor::init() {

	typedef GDCMImageIO          ImageIOType;
	typedef GDCMSeriesFileNames  NamesGeneratorType;

	loadLabelMap();

	//Read DICOM
	string intensityDirname = mInputPath + "/dicom/";
	ImageIOType::Pointer gdcmIO = ImageIOType::New();
	NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
	IntensityReaderType::Pointer readerIntensity = IntensityReaderType::New();
	namesGenerator->SetInputDirectory(intensityDirname);
	readerIntensity->SetImageIO(gdcmIO);
	readerIntensity->SetFileNames(namesGenerator->GetInputFileNames());
    //readerIntensity->Update();

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



    
    LabelImageType::SizeType dim = mLabelImage->GetLargestPossibleRegion().GetSize();

    IntensityIndexType intensityIdx = IntensityIndexType();

	long loadedVOIs = 0;
	LblRegionIteratorType it(mLabelImage, mLabelImage->GetLargestPossibleRegion());
	//it.SetNumberOfSamples(mVOICount * 5);
	vector<LabelIndexType>* backgroundList = &mLabelMap[0];
	vector<LabelIndexType>* lungList = &mLabelMap[500];

    for(it.GoToBegin(); !it.IsAtEnd(); ++it) {


        LabelPixelType label = it.Get();
		LabelIndexType idx = it.GetIndex();
		
		if (idx[0] < mDiagonal || idx[1] < mDiagonal || idx[2] < mDiagonal) {
			continue;
		}

		if (idx[0] + mDiagonal > dim[0] || idx[1] + mDiagonal > dim[1] || idx[2] + mDiagonal > dim[2]) {
			continue;
		}

		//Improve pefromance by avoiding doing hash lookups for frequent labels like 0 and 500 
		if (label == 0) {
			if (backgroundList->size() < mVOICount * 3) {
				backgroundList->push_back(idx);
				++loadedVOIs;
			}
			continue;
		}

		if (label == 500) {
			if (lungList->size() < mVOICount * 3) {
				lungList->push_back(idx);
				++loadedVOIs;
			}
			continue;
		}

		if(mEnabledLabelsLookup.find(label) == mEnabledLabelsLookup.end()) {
			continue;
        }

		mLabelMap[label].push_back(idx);
		++loadedVOIs;


    }

	cout << "Done" << endl;
}
