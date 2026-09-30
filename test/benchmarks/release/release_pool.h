// Persistent worker pool for the CPU baselines' batched calls.
//
// Why not the Python ThreadPoolExecutor the first collector used: submitting
// N futures and joining them from Python costs 100-900 us per call (executor
// bookkeeping, GIL hand-offs, thread wake-ups), which is 10-90x the single
// sample work of a small robot. That made Pinocchio look 20x slower between
// B=16 and B=32 (the batch where the pool switched on) — a measurement of the
// Python executor, not of Pinocchio. Here the calling thread runs slice 0
// itself and the other slices are handed to already-running workers with one
// generation bump; one thread means no hand-off at all.
//
// Header-only and library-agnostic so the CPU tests can compile it into a
// fake translation unit without Pinocchio.
#pragma once
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class ReleasePool {
public:
    explicit ReleasePool(std::size_t helpers) : generation_(0), pending_(0), stop_(false) {
        for (std::size_t k = 0; k < helpers; ++k) workers_.emplace_back([this, k] { loop(k + 1); });
    }
    ~ReleasePool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto &w : workers_) w.join();
    }
    ReleasePool(const ReleasePool &) = delete;
    ReleasePool &operator=(const ReleasePool &) = delete;

    // Number of threads (including the caller) this pool can run at once.
    std::size_t capacity() const { return workers_.size() + 1; }

    // Run `task(slot)` for slot in [0, active): slot 0 on the calling thread, the
    // rest on the persistent helpers. Blocks until every slot has completed.
    // `active` above capacity() is clamped. Exceptions inside a task are the
    // task's responsibility (record, never throw across the pool boundary).
    void run(std::size_t active, const std::function<void(std::size_t)> &task) {
        if (active > capacity()) active = capacity();
        if (active <= 1) { task(0); return; }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = &task;
            active_ = active;
            pending_ = active - 1;
            ++generation_;
        }
        wake_.notify_all();
        task(0);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return pending_ == 0; });
        task_ = nullptr;
    }

private:
    void loop(std::size_t slot) {
        unsigned long seen = 0;
        for (;;) {
            const std::function<void(std::size_t)> *task = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                if (slot < active_) task = task_;
            }
            if (task) {
                (*task)(slot);
                std::lock_guard<std::mutex> lock(mutex_);
                if (--pending_ == 0) done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(std::size_t)> *task_ = nullptr;
    std::size_t active_ = 0;
    unsigned long generation_;
    std::size_t pending_;
    bool stop_;
};
