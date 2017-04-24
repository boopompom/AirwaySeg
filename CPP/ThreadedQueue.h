#ifndef DICOMPROCESSOR_THREADEDQUEUE_H
#define DICOMPROCESSOR_THREADEDQUEUE_H


#include <thread>
#include <list>
#include <condition_variable>
#include <mutex>
#include <queue>

using namespace std;

template <typename T> class ThreadedQueue {

    queue<T>   mQueue;

    std::mutex mQueueMutex;
    std::condition_variable mCondition;

    bool mIsTerminated = false;

public:

    ThreadedQueue(const ThreadedQueue&) = delete;
    ThreadedQueue(const ThreadedQueue&&) = delete;


    ThreadedQueue() { }
    ~ThreadedQueue() { terminate(); }

    void terminate() {
        mIsTerminated = true;
    }

    void enqueue(T item) {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        mQueue.push(item);
        mCondition.notify_one();
    }

    T dequeue() {
        std::unique_lock<std::mutex> lock(mQueueMutex);
        if (mQueue.empty()) {
            mCondition.wait(lock, [&](){ return !mQueue.empty() || mIsTerminated; });
        }
        T item = mQueue.front();
        mQueue.pop();
        return item;
    }

    unsigned long size() {
        std::lock_guard<std::mutex> lock(mQueueMutex);
        return mQueue.size();
    }
};



#endif //DICOMPROCESSOR_THREADEDQUEUE_H
