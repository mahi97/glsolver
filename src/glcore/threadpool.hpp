// Persistent worker pool for the oracle's parallel regions (docs/optimizations.md O8).
//
// The EssentialOracle used to spawn std::threads for every parallel region; on instances with tens of
// thousands of small regions (one per deletion evaluation) thread creation dominated the region itself.
// A WorkerPool keeps its threads alive between regions: they sleep on a condition variable, are woken
// with a generation counter, pull indices from a shared atomic counter (dynamic load balancing, since
// per-vertex flows differ in cost) and signal completion through a second condition variable. The
// callback receives (index, worker id) and writes only into per-index slots, so the outcome is
// deterministic regardless of the thread count or the scheduling. Exceptions thrown by any worker are
// captured (the first one wins), the region drains, and the exception is rethrown on the calling thread.
// The pool is created lazily by its owner and joined in the destructor.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace glcore {

class WorkerPool {
public:
    WorkerPool() = default;
    ~WorkerPool() { shutdown(); }
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    // Execute fn(i, w) for every i in [0, count) using `nt` workers: the calling thread is worker 0 and
    // nt - 1 pool threads (created on demand) are workers 1 .. nt-1. Returns when every index has been
    // processed (or a worker threw: the region drains and the exception is rethrown here).
    void run(size_t count, int nt, const std::function<void(size_t, int)>& fn) {
        if (count == 0) return;
        if (nt < 2) {
            for (size_t i = 0; i < count; ++i) fn(i, 0);
            return;
        }
        ensure_threads(nt - 1);
        {
            std::lock_guard<std::mutex> lock(mu_);
            fn_ = &fn;
            count_ = count;
            next_.store(0, std::memory_order_relaxed);
            failed_.store(false, std::memory_order_relaxed);
            error_ = nullptr;
            wanted_ = nt - 1;
            active_ = wanted_;
            ++generation_;
        }
        cv_start_.notify_all();
        work(0);
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_done_.wait(lock, [&] { return active_ == 0; });
            fn_ = nullptr;
        }
        if (failed_.load(std::memory_order_acquire)) std::rethrow_exception(error_);
    }

    int size() const { return (int)threads_.size(); }

private:
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_start_, cv_done_;
    uint64_t generation_ = 0;   // bumped per region; workers compare against the last one they saw
    int wanted_ = 0;            // pool threads (ids 0 .. wanted_-1) participating in the current region
    int active_ = 0;            // participating pool threads that have not finished the region yet
    bool stop_ = false;
    const std::function<void(size_t, int)>* fn_ = nullptr;
    size_t count_ = 0;
    std::atomic<size_t> next_{0};
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    std::mutex err_mu_;

    void ensure_threads(int n) {
        while ((int)threads_.size() < n) {
            const int id = (int)threads_.size();
            threads_.emplace_back([this, id] { worker_loop(id); });
        }
    }

    // Shared work loop: grab indices until exhausted or a worker failed.
    void work(int w) {
        try {
            for (;;) {
                if (failed_.load(std::memory_order_relaxed)) break;
                const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
                if (i >= count_) break;
                (*fn_)(i, w);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(err_mu_);
            if (!failed_.exchange(true, std::memory_order_acq_rel)) error_ = std::current_exception();
        }
    }

    void worker_loop(int id) {
        uint64_t seen = 0;
        for (;;) {
            bool participate = false;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_start_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                participate = id < wanted_;
            }
            if (!participate) continue;
            work(id + 1);
            bool last = false;
            {
                std::lock_guard<std::mutex> lock(mu_);
                last = (--active_ == 0);
            }
            if (last) cv_done_.notify_one();
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_start_.notify_all();
        for (auto& th : threads_) th.join();
        threads_.clear();
    }
};

}  // namespace glcore
