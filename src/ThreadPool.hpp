#pragma once

#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <functional>
#include <queue>
#include <condition_variable>

struct Job {
	std::function<void(int)> job;
	unsigned int repeat;
	unsigned int repeatsLeft;
};

class ThreadPool {
public:
	ThreadPool(int threadCount) {
		if (threadCount > 0) {
			this->occupied = new bool[threadCount];
			memset(this->occupied, false, threadCount * sizeof(bool));
			this->threads.reserve(threadCount);
			for (int i = 0; i < threadCount; i++) {
				this->threads.push_back(std::thread(&ThreadPool::threadEntry, this, i));
			}
		}
		else {
			this->occupied = nullptr;
		}
	}

	void addJob(const std::function<void(int)>& job, unsigned int repeat) {
		std::unique_lock<std::mutex> lock(mutex);
		this->jobs.push(Job(job, repeat, repeat));
		if (repeat >= threads.size()) {
			cv.notify_all();
		}
		else {
			for (int i = 0; i < repeat; i++) {
				cv.notify_one();
			}
		}
	}

	bool isOccupied() {
		if (occupied != nullptr) {
			for (int i = 0; i < threads.size(); i++) {
				if (occupied[i]) {
					return true;
				}
			}
		}
		return !(jobs.size() <= 0);
	}

	void wait() {
		volatile int i = 0;
		while (isOccupied()) {
			i++;
		}
	}

	~ThreadPool() {
		terminate = true;
		cv.notify_all();

		std::cout << "Waiting for all threads ...\n";
		for (std::thread& thread : threads) {
			thread.join();
		}
		delete[] occupied;
	}

private:
	std::vector<std::thread> threads;
	std::mutex mutex;
	std::queue<Job> jobs;
	std::condition_variable cv;
	bool* occupied;
	bool terminate = false;

	void threadEntry(int threadId) {
		{
			std::unique_lock<std::mutex> lock(mutex);
			std::cout << "Thread " << threadId << " started" << std::endl;
		}

		Job job;
		unsigned int repeat = 0;
		while (true) {
			if (terminate) {
				return;
			}

			{
				std::unique_lock<std::mutex> lock(mutex);
				while (jobs.size() <= 0) {
					cv.wait(lock);
				}
				occupied[threadId] = true;
				Job* front = &jobs.front();
				job = *front;
				if (job.repeatsLeft >= 1) {
					repeat = job.repeat;
					repeat /= threads.size();
					repeat = std::max(std::min(job.repeatsLeft, repeat), 1u);

					front->repeatsLeft -= repeat;
				}
				if (front->repeatsLeft <= 0) {
					jobs.pop();
				}
				//std::cout << "Thread " << threadId << " got a job of " << repeat << " repeats" << std::endl;
			}
			while (repeat > 0) {
				//std::cout << "Thread " << threadId <<  " " << job.repeatsLeft - repeat << std::endl;
				job.job(job.repeatsLeft - repeat);
				repeat--;
			}
			repeat = 0;
			occupied[threadId] = false;
		}
	}
};
