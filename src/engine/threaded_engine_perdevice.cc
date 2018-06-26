/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file threaded_engine_perdevice.cc
 * \brief ThreadedEngine that uses fix amount of thread for each device.
 */
#include <dmlc/base.h>
#include <dmlc/omp.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/concurrency.h>
#include <dmlc/thread_group.h>
#include "./threaded_engine.h"
#include "./thread_pool.h"
#include "../common/lazy_alloc_array.h"
#include "../common/utils.h"

namespace mxnet {
namespace engine {
/*!
 * \brief ThreadedEngine uses per device threads.
 * The policy of this Engine:
 *  - Execute Async operation immediately if pushed from Pusher.
 *  - Use fixed amount of threads for each device.
 *  - Use special threads for copy operations.
 *  - Each stream is allocated and bound to each of the thread.
 */
class ThreadedEnginePerDevice : public ThreadedEngine {
 public:
  static auto constexpr kFIFO = dmlc::ConcurrentQueueType::kFIFO;
  static auto constexpr kPriority = dmlc::ConcurrentQueueType::kPriority;
  static auto constexpr kCopyQueue = kPriority;
  static auto constexpr kPriorityQueue = kPriority;
  static auto constexpr kWorkerQueue = kFIFO;

  ThreadedEnginePerDevice() noexcept(false) 
  {
    this->Start();
  }
  ~ThreadedEnginePerDevice() noexcept(false) 
  {
    this->StopNoWait();
  }

  void StopNoWait() {
    SignalQueuesForKill();
    gpu_normal_workers_.Clear();
    gpu_copy_workers_.Clear();
    cpu_normal_workers_.Clear();
    cpu_priority_worker_.reset(nullptr);
  }

  void Stop() override {
    if (is_worker_) return;
    WaitForAll();
    StopNoWait();
  }

  void Start() override {
    if (is_worker_) return;
   
    //gpu_worker_nthreads_ = common::GetNumThreadsPerGPU();
    gpu_worker_nthreads_=1;
    //cpu_worker_nthreads_ = dmlc::GetEnv("MXNET_CPU_WORKER_NTHREADS", 1);
    cpu_worker_nthreads_=1;
    // create CPU task
    //int cpu_priority_nthreads = dmlc::GetEnv("MXNET_CPU_PRIORITY_NTHREADS", 4);
    int  cpu_priority_nthreads=1;
    //td::unique_ptr<ThreadWorkerBlock<kPriorityQueue>> mxnet::engine::ThreadedEnginePerDevice::cpu_priority_worker_
    cpu_priority_worker_.reset(new ThreadWorkerBlock<kPriorityQueue>());

    cpu_priority_worker_->pool.reset(new ThreadPool(
        cpu_priority_nthreads,
        [this](std::shared_ptr<dmlc::ManualEvent> ready_event) 
        {
          //  线程的所作的匿名函数。
          // 
          this->CPUWorker(Context(), cpu_priority_worker_.get(), ready_event);
        },
         true
         )
         );
    // GPU tasks will be created lazily
  }

 protected:
  void PushToExecute(OprBlock *opr_block, bool pusher_thread) override 
  {

    const Context& ctx = opr_block->ctx;
    if ((opr_block->opr->prop == FnProperty::kAsync ||
         opr_block->opr->prop == FnProperty::kDeleteVar) && pusher_thread) 
    {
      if (ctx.dev_mask() == Context::kGPU) {
        #if MXNET_USE_CUDA
        MSHADOW_CATCH_ERROR(mshadow::SetDevice<gpu>(ctx.dev_id));
        #endif
      }
       LOG(INFO)<<"enter this->ExecuteOprBlock(RunContext{ctx, nullptr}, opr_block); ";
    
      this->ExecuteOprBlock(RunContext{ctx, nullptr}, opr_block);

    }
     else  
     {
      //  如果是在CPU上面执行的话。
      if (ctx.dev_mask() == Context::kCPU)
      {
        /**
         * enum class FnProperty {
              /*! \brief Normal operation 
              */
             // kNormal,
              /*! \brief Copy operation from GPU to other devices */
              //kCopyFromGPU,
              /*! \brief Copy operation from CPU to other devices */
              //kCopyToGPU,
              /*! \brief Prioritized sync operation on CPU */
              //kCPUPrioritized,
              /*! \brief Asynchronous function call */
             // kAsync,
              /*! \brief Delete variable call */
             // kDeleteVar
           // };  // enum class FnProperty
        
        // 如果是Prioritized sync operation on CPU
        if (opr_block->opr->prop == FnProperty::kCPUPrioritized) 
        { 
          //优先的队列设计
          LOG(INFO)<<"enter cpu_priority_worker_->task_queue.Push(opr_block, opr_block->priority); ";
          // 计算机学会通讯2018年第五期
          cpu_priority_worker_->task_queue.Push(opr_block, opr_block->priority);
        } 
        else 
        {
          int dev_id = ctx.dev_id;
          //
          int nthread = cpu_worker_nthreads_;
          //int nthread=1;
          // 我们其实发现
          //  这里面会先
          LOG(INFO)<<"ptr->task_queue.Push(opr_block, opr_block->priority);";
          auto ptr =
          cpu_normal_workers_.Get(dev_id, [this, ctx, nthread]() 
          {
              //   新构造一个线程workerblock
              auto blk = new ThreadWorkerBlock<kWorkerQueue>();
              //   
              blk->pool.reset(new ThreadPool(nthread,
                  [this, ctx, blk](std::shared_ptr<dmlc::ManualEvent> ready_event) 
                  {
                    this->CPUWorker(ctx, blk, ready_event);
                  }, 
              true));
            return blk;
          }
          );
          if (ptr) 
          {
            if (opr_block->opr->prop == FnProperty::kDeleteVar) 
            {
              ptr->task_queue.PushFront(opr_block, opr_block->priority);
            } 
            else 
            {
              ptr->task_queue.Push(opr_block, opr_block->priority);
            }
          }
        }
      }
      //  如果是在GPU上面。
       else 
       {
        CHECK_EQ(ctx.dev_mask(), Context::kGPU);
        // GPU execution.
        const FnProperty prop = opr_block->opr->prop;

        const bool is_copy = (prop == FnProperty::kCopyFromGPU ||
                              prop == FnProperty::kCopyToGPU);
        LOG(INFO)<<"is_copy"<<is_copy; 
        
        const size_t nthread = gpu_worker_nthreads_;

        //const size_t nthread=1;

        LOG(INFO)<<"gpu_worker_nthreads_"<<gpu_worker_nthreads_;
        if (is_copy) 
        {
          auto ptr = gpu_copy_workers_.Get(ctx.dev_id, [this, ctx, is_copy, nthread]() {
            // Signify to kernel that GPU is being used, so reserve cores as necessary
            OpenMP::Get()->set_reserve_cores(GetReserveCoreCount(true));
            auto blk = new ThreadWorkerBlock<kCopyQueue>();
              blk->pool.reset(new ThreadPool(
                nthread,
                [this, ctx, is_copy, blk]
                  (std::shared_ptr<dmlc::ManualEvent> ready_event) {
                    this->GPUWorker(ctx, is_copy, blk, ready_event);
                  }, true));
              return blk;
            });
          if (ptr) {
            if (opr_block->opr->prop == FnProperty::kDeleteVar) {
              ptr->task_queue.PushFront(opr_block, opr_block->priority);
            } else {
              ptr->task_queue.Push(opr_block, opr_block->priority);
            }
          }
        }
        else 
         {
          //      核心是得到对应的队列。
          //  
          LOG(INFO)<<"gpu_normal_workers_.Get(ctx.dev_id, [this, ctx, is_copy, nthread]()";    
          auto ptr = gpu_normal_workers_.Get(ctx.dev_id, [this, ctx, is_copy, nthread]() 
          {
            // Signify to kernel that GPU is being used, so reserve cores as necessary
              OpenMP::Get()->set_reserve_cores(GetReserveCoreCount(true));

              auto blk = new ThreadWorkerBlock<kWorkerQueue>();

              blk->pool.reset(new ThreadPool
              (
                nthread,
                [this, ctx, is_copy, blk]
                  (std::shared_ptr<dmlc::ManualEvent> ready_event) 
                  {
                    this->GPUWorker(ctx, is_copy, blk, ready_event);
                  }, 
                  true
              )
              );
              return blk;
          }
          );
          if (ptr)
           {
            if (opr_block->opr->prop == FnProperty::kDeleteVar) 
            {
              ptr->task_queue.PushFront(opr_block, opr_block->priority);
            } 
            else 
            {
              ptr->task_queue.Push(opr_block, opr_block->priority);
            }
          }
        }
      }
    }
  }

 private:
  // working unit for each of the task.
  template<dmlc::ConcurrentQueueType type>
  struct ThreadWorkerBlock
   {
    // task queue on this task
    dmlc::ConcurrentBlockingQueue<OprBlock*, type>  task_queue;
    // thread pool that works on this task
    std::unique_ptr<ThreadPool> pool;
    // constructor
    ThreadWorkerBlock() = default;
    // destructor
    ~ThreadWorkerBlock() noexcept(false) {}
  };

  /*! \brief whether this is a worker thread. */
  static MX_THREAD_LOCAL bool is_worker_;
  /*! \brief number of concurrent thread cpu worker uses */
  size_t cpu_worker_nthreads_;
  /*! \brief number of concurrent thread each gpu worker uses */
  size_t gpu_worker_nthreads_;
  // cpu worker
  common::LazyAllocArray<ThreadWorkerBlock<kWorkerQueue> > cpu_normal_workers_;
  // cpu priority worker
  std::unique_ptr<ThreadWorkerBlock<kPriorityQueue> > cpu_priority_worker_;
  // workers doing normal works on GPU
  common::LazyAllocArray<ThreadWorkerBlock<kWorkerQueue> > gpu_normal_workers_;
  // workers doing copy works from/to GPU
  common::LazyAllocArray<ThreadWorkerBlock<kCopyQueue> > gpu_copy_workers_;
  /*!
   * \brief GPU worker that performs operations on a certain device.
   * \param dev_id The device id of the worker.
   * \param is_copy_worker whether the worker only do copy job
   * \param block The task block of the worker.
   * 
   * 用所有的任务块来构造一个GPUworker，实际上是不断每一个线程都要做到做的事情。
   * 
   */
  template<dmlc::ConcurrentQueueType type>
  inline void GPUWorker(Context ctx,  
                        bool is_copy_worker,
                        ThreadWorkerBlock<type> *block,
                        const std::shared_ptr<dmlc::ManualEvent>& ready_event) 
  {
    LOG(INFO)<<"GPUWorker";
    this->is_worker_ = true;
    #if MXNET_USE_CUDA
    CHECK(block != nullptr);
    mshadow::Stream<gpu> *stream;
    //  也就是说：
    //  一个GPU总共也是只有2个流来使用。
    do 
    {

      ThreadPool::SetReadyOnDestroy setReady(ready_event);
      // allocate stream
      mshadow::SetDevice<gpu>(ctx.dev_id);
      //  第一个作业。
      LOG(INFO)<<"is_copy_worker"<<is_copy_worker;

      if (is_copy_worker)
      {
        //  c创建一个复制流
        stream = mshadow::NewStream<gpu>(false, false, ctx.dev_id);
      }
      else  //  工作worker
      {
        // 创建一个工作流
        stream = mshadow::NewStream<gpu>(true, MXNET_USE_CUDNN != 0, ctx.dev_id);
      }

    } while (false);
    // execute task

    OprBlock* opr_block;
    //  合成上下文

    RunContext run_ctx{ctx, stream};

    //  获取任务队列
    auto* task_queue = &(block->task_queue);

    // Don't eat up omp threads for GPU jobs.  They're probably best used elsewhere,
    // for example for image decoding or the optimizer pass
    OpenMP::Get()->on_start_worker_thread(false);
    //  只要任务队列部位空，那么自然
    //  从任务队列里面弹出一个OP。放到opr_block
    //  然后去执行
    //  执行完毕后，继续查看队列是不是为空
    //  如果为空，那么接着执行获取任务，实现了线程的共享。

    while (task_queue->Pop(&opr_block))
    {
      //    只要任务队列不空，不断调度执行
      //    this应该表示本线程了。
      //     
      this->ExecuteOprBlock(run_ctx, opr_block);
    }
    // Catch exception for CUDA driver shutdown
    MSHADOW_CATCH_ERROR(mshadow::DeleteStream<gpu>(stream));
    #else
    ready_event->signal();
    #endif
  }
  /*!
   * \brief CPU worker that performs operations on CPU.
   * \param block The task block of the worker.
   */
  template<dmlc::ConcurrentQueueType type>
  inline void CPUWorker(Context ctx,
                        ThreadWorkerBlock<type> *block,
                        const std::shared_ptr<dmlc::ManualEvent>& ready_event) 
  {
    LOG(INFO)<<"CPUWorker";
    this->is_worker_ = true;
    //  dmlc::ConcurrentBlockingQueue<OprBlock*,type> mxnet::engine
    auto* task_queue = &(block->task_queue);

    RunContext run_ctx{ctx, nullptr};

    // execute task

    OprBlock* opr_block;

    ready_event->signal();

    // Set default number of threads for OMP parallel regions initiated by this thread
    OpenMP::Get()->on_start_worker_thread(true);
    //  循环不断获取任务执行。
    //  
    while (task_queue->Pop(&opr_block))
    {
      LOG(INFO)<<"this->ExecuteOprBlock(run_ctx, opr_block);";
      this->ExecuteOprBlock(run_ctx, opr_block);
    }
  }

  /*!
   * \brief Get number of cores this engine should reserve for its own use
   * \param using_gpu Whether there is GPU usage
   * \return number of cores that this engine wishes to be reserved
   * \note Testing found no degradation of performance using these values
   *       running cifar10 with resnet50 on various GPU systems,
   *       including AWS p2.16xlarge, which has 16 GPU's
   */
  int GetReserveCoreCount(const bool using_gpu) const {
    int reserve = 0;
    if (using_gpu) {
      // Save at least one for GPU tasks
      ++reserve;
      // If we have 8 or more real cores, reserve another core for GPU tasks
      if (OpenMP::Get()->GetRecommendedOMPThreadCount(true) >= 8) {
        ++reserve;
      }
    }
    return reserve;
  }

  /*! \brief Signal a single queue for shutdown */
  template<typename Object>
  static inline void SignalQueueForKill(common::LazyAllocArray<Object> *array) {
    array->ForEach([](size_t i, Object *block) {
      block->task_queue.SignalForKill();
    });
  }

  /*! Signal all queues for shutdown */
  void SignalQueuesForKill() {
    SignalQueueForKill(&gpu_normal_workers_);
    SignalQueueForKill(&gpu_copy_workers_);
    SignalQueueForKill(&cpu_normal_workers_);
    if (cpu_priority_worker_) {
      cpu_priority_worker_->task_queue.SignalForKill();
    }
  }
};

Engine *CreateThreadedEnginePerDevice() {
  return new ThreadedEnginePerDevice();
}

MX_THREAD_LOCAL bool ThreadedEnginePerDevice::is_worker_ = false;

}  // namespace engine
}  // namespace mxnet
