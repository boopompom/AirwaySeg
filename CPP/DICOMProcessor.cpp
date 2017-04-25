
#include <vector>
#include <map>
#include <memory>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegionConstIterator.h>
#include <itkNeighborhoodIterator.h>

#include "optionparser.h"
#include "ThreadedQueue.h"
#include "ImageProcessor.h"
#include "CommandLineProcessor.h"

using namespace std;
using namespace itk;



void worker(shared_ptr<ThreadedQueue<DICOMJob>> q) {
    while(q->size() != 0) {
        ImageProcessor(q->dequeue()).process();
        if(q->size() == 0) {
            q->terminate();
        }
    }
}

int main(int argc, char* argv[]) {

    argc-=(argc>0); argv+=(argc>0);
    option::Stats  stats(usage, argc, argv);
    option::Option options[1024], buffer[1024];
    option::Parser parse(usage, argc, argv, options, buffer);

    if (parse.error()) {
        return 1;
    }

    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, usage);
        return 0;
    }

    vector<unsigned int> enabledLabels;
    if (!options[LABEL_LIST] ) {
        cerr << "You need at least one label" << endl;
        return 1;
    }
	//enabledLabels.push_back(0);
	enabledLabels.push_back(500);
	for(option::Option* opt = options[LABEL_LIST]; opt; opt = opt->next()) {
        enabledLabels.push_back((unsigned int)stoi(opt->arg));
    }

    vector<string> inputPaths;
    if (!options[INPUT_LIST] || options[INPUT_LIST].count() == 0) {
        cerr << "You need at least one input directory" << endl;
        return 1;
    }
    for(option::Option* opt = options[INPUT_LIST]; opt; opt = opt->next()) {
        const char * x = opt[1].arg;

        inputPaths.push_back(string(opt->arg));
    }

    if (!options[OUTPUT_PATH] || options[LABEL_LIST].count() == 0) {
        cerr << "You need to specify an output path" << endl;
        return 1;
    }
    string outputPath = options[OUTPUT_PATH].first()->arg;

    unsigned int diameter = 53;
    if(options[DIAMETER]) {
        diameter = (unsigned int)std::stoi(options[DIAMETER].first()->arg);
    }

    int randomSeed = 1;
    if(options[RANDOM_SEED]) {
        randomSeed = std::stoi(options[RANDOM_SEED].first()->arg);
    }
    std::srand((unsigned int)randomSeed);

    unsigned int threadCount = 1;
    if(options[THREAD_COUNT]) {
        threadCount = (unsigned int)std::stoi(options[THREAD_COUNT].first()->arg);
    }

    unsigned int voiPerLabel = 50;
    if(options[VOI_PER_LABEL]) {
        voiPerLabel = (unsigned int)std::stoi(options[VOI_PER_LABEL].first()->arg);
    }

    shared_ptr<ThreadedQueue<DICOMJob>> queue = shared_ptr<ThreadedQueue<DICOMJob>>(new ThreadedQueue<DICOMJob>());
    for(int i=0;i<inputPaths.size(); i++) {
        DICOMJob job;
        job.diameter = diameter;
        job.voiPerLabel = voiPerLabel;
        job.enabledLabels = enabledLabels;
        job.inputPath = inputPaths[i];
        job.outputPath = outputPath;
        queue->enqueue(job);
    }

    vector<thread> activeThreads;
    for(int i=0;i<threadCount;i++) {
        activeThreads.push_back(thread(worker, queue));
    }

    for(int i=0;i<activeThreads.size();i++) {
        activeThreads[i].join();
    }

    return 0;
}