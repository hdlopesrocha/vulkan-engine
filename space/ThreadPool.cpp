#include "ThreadPool.hpp"
#include <iostream>

// Constructor: launch worker threads
ThreadPool::ThreadPool(size_t threads)
	: stop(false)
{
	for(size_t i = 0; i < threads; ++i) {
		workers.emplace_back([this] {
			for(;;) {
				std::function<void()> task;
				{
					std::unique_lock<std::mutex> lock(this->queue_mutex);
					this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
					if(this->stop && this->tasks.empty())
						return;
					task = std::move(this->tasks.front());
					this->tasks.pop();
				}
				task();
			}
		});
	}
}

// Destructor: stop all threads
ThreadPool::~ThreadPool()
{
	std::cout << "ThreadPool::~ThreadPool()" << std::endl;
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		stop = true;
	}
	condition.notify_all();
	for(auto &worker : workers) {
		if(worker.joinable())
			worker.join();
	}
}

size_t ThreadPool::threadCount() const
{
	return workers.size();
}
