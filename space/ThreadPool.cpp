#include "ThreadPool.hpp"

// Constructor: launch worker threads
ThreadPool::ThreadPool(size_t threads)
	: stopping(false)
{
	for(size_t i = 0; i < threads; ++i) {
		workers.emplace_back([this] {
			for(;;) {
				SmallFunction task;
				{
					std::unique_lock<std::mutex> lock(this->queue_mutex);
					this->condition.wait(lock, [this]{ return this->stopping || !this->tasks.empty(); });
					if(this->stopping && this->tasks.empty())
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
	stop();
}

// Explicit stop: signal workers to finish and drain remaining tasks.
// Safe to call multiple times and from any thread.
void ThreadPool::stop()
{
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		if (stopping) return;
		stopping = true;
	}
	condition.notify_all();
	for(auto &worker : workers) {
		if(worker.joinable()) {
			worker.join();
		}
	}
}

size_t ThreadPool::threadCount() const
{
	return workers.size();
}
