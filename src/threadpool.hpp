// Minimal persistent fork-join pool. No allocation after construction; the caller participates.
#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace hie {

class ThreadPool {
 public:
  explicit ThreadPool(int participants) : n_(participants < 1 ? 1 : participants) {
    for (int i = 1; i < n_; ++i) workers_.emplace_back([this, i] { loop(i); });
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
      ++gen_;
    }
    cv_start_.notify_all();
    for (auto& t : workers_) t.join();
  }
  int size() const { return n_; }

  // f(task, thread_id); tasks are claimed dynamically. thread_id in [0, size()).
  template <class F>
  void run(int tasks, F&& f) {
    if (tasks <= 0) return;
    if (n_ == 1 || tasks == 1) {
      for (int t = 0; t < tasks; ++t) f(t, 0);
      return;
    }
    using FD = typename std::remove_reference<F>::type;
    dispatch(tasks, [](void* c, int t, int tid) { (*static_cast<FD*>(c))(t, tid); },
             (void*)&f);
  }

 private:
  using Fn = void (*)(void*, int, int);

  void dispatch(int tasks, Fn fn, void* ctx) {
    {
      std::lock_guard<std::mutex> lk(m_);
      fn_ = fn;
      ctx_ = ctx;
      tasks_ = tasks;
      next_.store(0, std::memory_order_relaxed);
      active_ = n_ - 1;
      ++gen_;
    }
    cv_start_.notify_all();
    work(0);
    std::unique_lock<std::mutex> lk(m_);
    cv_done_.wait(lk, [this] { return active_ == 0; });
  }

  void work(int tid) {
    for (;;) {
      int t = next_.fetch_add(1, std::memory_order_relaxed);
      if (t >= tasks_) break;
      fn_(ctx_, t, tid);
    }
  }

  void loop(int tid) {
    unsigned seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_start_.wait(lk, [&] { return gen_ != seen; });
        seen = gen_;
        if (stop_) return;
      }
      work(tid);
      {
        std::lock_guard<std::mutex> lk(m_);
        if (--active_ == 0) cv_done_.notify_one();
      }
    }
  }

  int n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_start_, cv_done_;
  Fn fn_ = nullptr;
  void* ctx_ = nullptr;
  int tasks_ = 0;
  std::atomic<int> next_{0};
  int active_ = 0;
  unsigned gen_ = 0;
  bool stop_ = false;
};

}  // namespace hie
