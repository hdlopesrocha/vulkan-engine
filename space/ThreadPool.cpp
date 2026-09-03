#include "ThreadPool.hpp"
#include <cstdio>
#include <system_error>

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
	for(size_t i = 0; i < workers.size(); ++i) {
		auto &worker = workers[i];
		if(worker.joinable()) {
			try {
				worker.join();
			} catch (const std::system_error& e) {
				// Never let a failed join escape: stop() runs from the
				// noexcept destructor, so any exception here is an
				// immediate std::terminate. Log and continue tearing down
				// the remaining workers instead (seen once as ESRCH /
				// "No such process" during TerrainStreamer pool replacement).
				fprintf(stderr, "[ThreadPool::stop] worker %zu join failed: %s\n",
				        i, e.what());
			}
		}
	}
}

// Pop and run one queued task. Mirrors the worker loop's dequeue, minus the
// condition-variable wait: cooperative waiters call this in a loop, so they
// stay productive while there is queued work that would unblock them.
bool ThreadPool::runOneTask()
{
	SmallFunction task;
	{
		std::unique_lock<std::mutex> lock(this->queue_mutex);
		if(this->tasks.empty())
			return false;
		task = std::move(this->tasks.front());
		this->tasks.pop();
	}
	task();
	return true;
}

size_t ThreadPool::threadCount() const
{
	return workers.size();
}
