/*!
 *  Copyright (c) 2016 by Contributors
 * \file plan_memory.cc
 * \brief Assign memory tag to each of the data entries.
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <memory>
#include "./graph_algorithm.h"

namespace nnvm {
namespace pass {
namespace {

// simple graph based allocator.
class GraphAllocator {
 public:
  // storage id equals integer.
  using StorageID = int;

  // bad storage id
  static const StorageID kBadStorageID = -1;
  // external storage id
  static const StorageID kExternalStorageID = -2;
  // dynamic storage id
  static const StorageID kDynamicStorageID = -3;

  // request a free storage
  StorageID Request(int dev_id, int dtype, TShape shape, uint32_t node_id) {
    if (shape.ndim() == 0) return kBadStorageID;
    // search memory block in [size / match_range_, size * match_range_)
    // TODO(tqchen) add size of the dtype, assume 4 bytes for now
    size_t size = shape.Size() * 4;
    if (match_range_ == 0) return this->Alloc(dev_id, size);
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // find a exact match, erase from map and return
      free_.erase(it);
      return e->id;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // erase from map and return
      free_.erase(it);
      return e->id;
    }
    // cannot find anything return a new one.
    return this->Alloc(dev_id, size);
  }
  // release a memory space.
  void Release(StorageID id, uint32_t node_id) {
    CHECK_NE(id, kBadStorageID);
    if (id == kExternalStorageID || id == kDynamicStorageID) return;
    StorageEntry *e = data_[id].get();
    e->released_by_node = node_id;
    free_.insert({e->max_bytes, e});
    //  git  使用方法。



    
  }

  // totoal number of bytes allocated
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (auto &p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  // constructor
  explicit GraphAllocator(const IndexedGraph* idx, const size_t match_range) : idx_(idx) {
    this->Init(match_range, dmlc::GetEnv("NNVM_EXEC_NUM_TEMP", 1));
  }

 private:
  // initialize the graph allocator
  void Init(const size_t match_range, const uint32_t num_match_color) {
    match_range_ = match_range;
    num_match_color_ = num_match_color;
    if (num_match_color_ > 1) {
      std::vector<uint32_t> importance(idx_->num_nodes(), 0);
      for (uint32_t nid = 0; nid < idx_->num_nodes(); ++nid) {
        if ((*idx_)[nid].source->is_variable()) continue;
        importance[nid] = 1;
      }
      num_match_color_ = pass::ColorNodeGroup(
          *idx_, importance, num_match_color_, &node_color_);
    }
  }

  StorageID Alloc(int dev_id, size_t size) {
    StorageID id = static_cast<StorageID>(data_.size());
    std::unique_ptr<StorageEntry> ptr(new StorageEntry());
    ptr->id = id;
    ptr->device_id = dev_id;
    ptr->max_bytes = size;
    data_.emplace_back(std::move(ptr));
    return id;
  }
  // internal storage entry
  struct StorageEntry {
    // the id of the entry.
    StorageID id;
    // the device id of the storage.
    int device_id;
    // maximum size of storage requested.
    size_t max_bytes{0};
    // node index that released it last time
    uint32_t released_by_node{0};
  };
  // scale used for rough match
  size_t match_range_;
  // whether use color based match algorithm
  uint32_t num_match_color_{1};
  // the size of each dtype
  std::vector<size_t> dtype_size_dict_;
  // free list of storage entry
  std::multimap<size_t, StorageEntry*> free_;
  // all the storage resources available
  std::vector<std::unique_ptr<StorageEntry> > data_;
  // color of nodes in the graph, used for auxiliary policy making.
  std::vector<uint32_t> node_color_;
  // internal indexed graph
  const IndexedGraph* idx_;
};

/*
 * Internal method to perform the memory allocation for a graph
 * 
 * 采用真正的内存分配函数
 * */
size_t AllocMemory(const Graph& ret, const IndexedGraph& idx,
                   const std::pair<uint32_t, uint32_t>& node_range,
                   StorageVector* storage_ptr,
                   std::vector<int>* storage_inplace_index_ptr,
                   const std::vector<uint32_t>& entry_ref_count,
                   GraphAllocator* allocator) {
  static auto& finplace_option =   Op::GetAttr<FInplaceOption>("FInplaceOption");
  static auto& finplace_identity = Op::GetAttr<FInplaceIdentity>("FInplaceIdentity");
  static auto& fignore_inputs =    Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");

  // Get reference
  auto &storage = *storage_ptr;
  auto &storage_inplace_index = *storage_inplace_index_ptr;
  for(int i=0;i<storage.size();i++)
  {
      LOG(INFO)<<"storage存储节点向量  "<< i<<"对应的存储类型为：："<<storage[i];
  }
  //  这里面可以发现，存储的类型为 -1  和 -2
  /*  [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  0对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  1对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  2对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  3对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  4对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  5对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  6对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  7对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  8对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  9对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  10对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  11对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  12对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  13对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  14对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  15对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  16对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  17对应的引用计数为：：-2
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  18对应的引用计数为：：-2
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  19对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  20对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  21对应的引用计数为：：-2
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  22对应的引用计数为：：-2
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  23对应的引用计数为：：-1
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  24对应的引用计数为：：-2
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  25对应的引用计数为：：-2
      [04:56:39] src/pass/plan_memory.cc:168: storage存储节点向量  26对应的引用计数为：：-2 */

  for(int i=0;i<storage_inplace_index.size();i++)
  {
      LOG(INFO)<<"storage_inplace_index存储节点向量  "<< i<<"是不是具有原地：："<<storage_inplace_index[i];
  }
  //  都为-1   都不是原地操作。
    /*  
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  0对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  1对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  2对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  3对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  4对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  5对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  6对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  7对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  8对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  9对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  10对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  11对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  12对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  13对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  14对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  15对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  16对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  17对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  18对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  19对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  20对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  21对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  22对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  23对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  24对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  25对应的引用计数为：：-1
    [04:56:39] src/pass/plan_memory.cc:172: storage_inplace_index存储节点向量  26对应的引用计数为：：-1 */



  // Get attributes from the graph
  const ShapeVector& shape_vec = ret.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = ret.GetAttr<DTypeVector>("dtype");
  const DeviceVector* device_vec = nullptr;
  for(int i=0;i<dtype_vec.size();i++)
  {
      LOG(INFO)<<"DTypeVector   "<< i<<"存储类型：："<<dtype_vec[i];
  }
  // 为什么都是0？？？？？？
    /* 
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   0存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   1存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   2存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   3存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   4存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   5存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   6存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   7存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   8存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   9存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   10存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   11存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   12存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   13存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   14存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   15存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   16存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   17存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   18存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   19存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   20存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   21存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   22存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   23存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   24存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   25存储类型：：0
    [04:56:39] src/pass/plan_memory.cc:181: DTypeVector   26存储类型：：0 */
   // 如果设备已经分配好了
  if (ret.attrs.count("device") != 0)
  {
     device_vec = &(ret.GetAttr<DeviceVector>("device"));
    
    /*  for(int i=0;i<device_vec->size();i++)
     {
      LOG(INFO)<<"device_vecr   "<< i<<"   设备为：： "<<device_vec[i];
     } */
  }


  size_t num_not_allocated = 0;
  //  27
  LOG(INFO)<<" idx.num_node_entries()  "<<idx.num_node_entries();
  // 这里面应该是27   
  std::vector<GraphAllocator::StorageID> storage_ref_count(idx.num_node_entries(), 0); 
  // 0--20 因为节点只有20个
  LOG(INFO)<<" node_range.first  "<<node_range.first<<" node_range.second "<<node_range.second;

   //  这里里面 0-20
   //  遍历每一个节点
  for (uint32_t nid = node_range.first; nid < node_range.second; ++nid)
  {
    
    LOG(INFO)<<" **********************************************************************************";
    LOG(INFO)<<"nid======="<<nid;
    const auto& inode = idx[nid];
    const std::string& arg_name = inode.source->attrs.name;
    LOG(INFO)<<"这个节点的名字为"<<arg_name;
    if (inode.source->is_variable()) continue;
    // check inplace option
    // 检查是不是可以进行原地操作。
    // 原地操作的条件是？？？？？？ 

    if (finplace_option.count(inode.source->op()) != 0) 
    {
      //  这里面不会执行 
      auto inplace_pairs = finplace_option[inode.source->op()](inode.source->attrs);
      std::vector<bool> identity;

      if (finplace_identity.count(inode.source->op()) != 0) 
      {
        identity = finplace_identity[inode.source->op()](inode.source->attrs);
        CHECK_EQ(identity.size(), inplace_pairs.size())
            << "FInplaceOption and FInplaceIdentity returned vectors of different "
            << "size for operator " << inode.source->op()->name;
      } 
      else 
      {
        identity = std::vector<bool>(inplace_pairs.size(), false);
      }
      std::vector<bool> taken(inode.inputs.size(), false);
      for (size_t ipair = 0; ipair < inplace_pairs.size(); ++ipair) 
      {
        const auto& kv = inplace_pairs[ipair];
        uint32_t eid_out = idx.entry_id(nid, kv.second);
        uint32_t eid_in = idx.entry_id(inode.inputs[kv.first]);
        auto sid_out = storage[eid_out];
        auto sid_in = storage[eid_in];
        bool ignore_all_inputs = (fignore_inputs.count(inode.source->op()) != 0 &&
                                  fignore_inputs[inode.source->op()](
                                      inode.source->attrs).size() == inode.source->num_inputs());
        if (taken[kv.first] == false &&
            sid_out == GraphAllocator::kBadStorageID &&
            sid_in >= 0 &&
            ((storage_ref_count[sid_in] == 1 && !ignore_all_inputs) || identity[ipair]) &&
            entry_ref_count[eid_out] > 0 &&
            shape_vec[eid_out].Size() == shape_vec[eid_in].Size() &&
            dtype_vec[eid_out] == dtype_vec[eid_in]) 
        {
          // inplace optimization
          taken[kv.first] = true;
          storage[eid_out] = sid_in;
          // Reuse storage for output and add ref count of output
          // to storage. This will get substracted later in free
          // input section.
          storage_ref_count[sid_in] += entry_ref_count[eid_out];
          storage_inplace_index[eid_out] = kv.first;
        }
      }
    }
    // normal allocation
    // 节点的正常分配
    const int dev_id = (device_vec != nullptr) ? device_vec->at(nid) : 0;

    // sort output nodes based on size before allocating output
    // 对于输出节点按照大小进行排序
    std::multimap<size_t, uint32_t> eids;
    //   最终得到了每一个输入实体和对应的维度的信息
    //   为这个节点的每一个输出分配内存
    //  
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) 
    {
      //
      uint32_t eid = idx.entry_id(nid, index);
      
      // only request memory for kBadStorageID

      if (storage[eid] == GraphAllocator::kBadStorageID)
      {
        auto &eshape = shape_vec[eid];
        // 数据的维度
        size_t esize = 0;
        if (eshape.ndim() != 0) 
            esize = eshape.Size();
        //  插入维度和
        eids.insert(std::make_pair(esize, eid));
      }
    }

    //    这里面遍历刚才得到的每一个输入的维度。
    //    eids上面得到了。  
    for (auto rit = eids.rbegin(); rit != eids.rend(); ++rit)
    {
        
        uint32_t eid = rit->second;
        // 向分配器请求（设备ID，类型，形状，以及节点的）
        auto sid = allocator->Request(dev_id, dtype_vec[eid], shape_vec[eid], nid);
        //  如果返回分配成功的话，返回存储号
        if (sid >= 0) 
        {
           // 这个存储号码对应有多少引用，按时这个的实体对于的引用数目。
           storage_ref_count[sid] = entry_ref_count[eid];
        }
        //  实体ID和存储ID之间的对应关系在此
        storage[eid] = sid;

    }
    // check if certain inputs is ignored.
    std::vector<uint32_t> ignore_inputs;
    if (fignore_inputs.count(inode.source->op()) != 0) 
    {
      //  如果这个节点所呼略的输入不等于o  那么进行排序
      ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
      std::sort(ignore_inputs.begin(), ignore_inputs.end());
    }
    // then free inputs
    //  一旦这个节点被正确的分配
    LOG(INFO)<<"这个节点的输入节点个数为"<<inode.inputs.size();
    for (size_t i = 0; i < inode.inputs.size(); ++i) 
    {
      // ref counter of ignored input is already decreased.
      // 如果在忽略额节点里面被找到，直接退出这一轮迭代过程
      if (std::binary_search(ignore_inputs.begin(), ignore_inputs.end(), i)) continue;
      //  每一个输入都是一个实体
      const auto& e = inode.inputs[i];
      //  说明是返回一个实体对应的编号   
      uint32_t eid = idx.entry_id(e); 
      auto sid = storage[eid];     
      //  获取到这个实体的对于的存储ID
      // storage_ref_count == 0 means it is taken by inplace op
      LOG(INFO)<<"第"<<i<<"个输入"<<"对应的输入的实体为ID为"<<eid<<"  存储的ID为   "<<sid;
      if (sid < 0) continue;
      //  if we decrease it to zero, means we are ready to relase
      //  一旦这个数据的输出节点的呗分配了内存，那么所有的输入节点的引用就会减少一个。
      //  一旦变成0 之后
      //  说明这个空间可以在输出都已经存在的情况下，输入
      --storage_ref_count[sid];
      LOG(INFO)<<"  对应的引用计数为  "<<storage_ref_count[sid];
      if (storage_ref_count[sid] == 0) 
      {

        allocator->Release(sid, nid);


      }
    }
    // check if there are outputs that can be freeded immediately
    // these output are not referenced by any operator.
    LOG(INFO)<<" 这个节点的  nid"<<nid<<"   inode.source->num_outputs()   "<< inode.source->num_outputs();
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) 
    {
      //  这个节点的所有的输出节点
      
      uint32_t eid = idx.entry_id(nid, index);
      LOG(INFO)<<nid<<"第"<<index<<"个输出节点为eid=="<<eid;
      auto sid = storage[eid];
      //  如果引用计数等于0 的话
      if (sid >= 0 && storage_ref_count[sid] == 0) 
      {
        // 直接释放
        allocator->Release(sid, nid);
        // use -2 to indicate that the node was never touched.
        storage_inplace_index[eid] = -2;
      }
      if (storage[eid] == GraphAllocator::kBadStorageID)
      {
          ++num_not_allocated;
      }
    }
  }
  //还没有进行分配 的节点数目

  return num_not_allocated;
  LOG(INFO)<< "  num_not_allocated  "<<num_not_allocated;
}


// function to plan memory
Graph PlanMemory(Graph ret)
 {

  // 进入内存规划函数。
  LOG(INFO)<<"PlanMemory(Graph ret)";
  //  setup ref counter
  
  const IndexedGraph& idx = ret.indexed_graph();
  //  获取可以忽略的节点
  static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
  //   20
  LOG(INFO)<<"idx.num_nodes()   "<<idx.num_nodes();
  std::pair<uint32_t, uint32_t> node_range = {0, idx.num_nodes()};
  // 如果指定了节点的范围
  if (ret.attrs.count("node_range")) 
  {
    node_range = ret.MoveCopyAttr<std::pair<uint32_t, uint32_t> >("node_range");
  }
  // reference counter of each node
  // 每一个节点的引用计数
  std::vector<uint32_t> ref_count;

  // step 1: initialize reference count
  int  a=ret.attrs.count("ref_count");
  LOG(INFO)<<"ret.attrs.count(ref_count   "<<a;
  //  0  表示之前没有任何的设置引用计数
  if (ret.attrs.count("ref_count") != 0) 
  { 
    ref_count = ret.MoveCopyAttr<std::vector<uint32_t> >("ref_count");
  } 
 else
 {  
    //   进入这里面。
    //   27  
    //   这里面可以看出
    //   节点的编号和实体的编号没有任何关系
    //   节点只是计算节点
    //   实体代表的是数据实体
    //   也就是说 目前的 mxnet 计算都是十分粗粒度的。
    //   27
    LOG(INFO)<<"idx.num_node_entries()"<<idx.num_node_entries();

    ref_count.resize(idx.num_node_entries(), 0);  
    //   这里面遍历每一个计算的节点
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) 
    {
     
     
      // 结果为Node
      // std::vector<Node> nodes_;
      //  inline const Node& operator[](uint32_t node_id) const {
      //明白了  重载了这个函数
      //return nodes_[node_id];
      //}
      //

       //          struct Node 
      // {
          /*! \brief pointer to the source node */
        // const nnvm::Node* source;
          /*! \brief inputs to the node */
          //array_view<NodeEntry> inputs;
          /*! \brief control flow dependencies to the node */
        // array_view<uint32_t> control_deps;
          /*! \brief weak reference to node */
          //std::weak_ptr<nnvm::Node> weak_ref;
        //};
      

             /* class NNVM_DLL Node {
                  public:
                    /*! \brief The attributes in the node. 
                    NodeAttrs attrs;
                    /*! \brief inputs to this node 
                    std::vector<NodeEntry> inputs;
                    /*!
                    * \brief Optional control flow dependencies
                    *  Gives operation must be performed before this operation.
                    
                    std::vector<NodePtr> control_deps;
                    /*! \brief additional fields for this node 
                    any info;
                    /*! \brief destructor of node 
                    ~Node();
                    /*! \return operator in this node 
                    inline const Op* op() const;
                    /*!
                    * \brief return whether node is placeholder variable.
                    *  This is equivalent to op == nullptr
                    * \return whether node is placeholder input variable
                    
                    inline bool is_variable() const;
                    /*! \return number of outputs from this node 
                    inline uint32_t num_outputs() const;
                    /*! \return number of inputs from this node 
                    inline uint32_t num_inputs() const;
                    /*!
                    * \brief create a new empty shared_ptr of Node.
                    * \return a created empty node.
                    
                    static NodePtr Create();
                  }; 
          */
      LOG(INFO)<<"  nid="<<nid; 
      const auto& inode = idx[nid];     // 真实节点的ID   1,3,2,4,5，等
      const std::string& arg_name = inode.source->attrs.name;
      LOG(INFO)<<"名字="<<arg_name;
      int a=inode.source->is_variable();
      LOG(INFO)<<"变量是"<<a;
      if (inode.source->is_variable()) continue;
      //  对于这个节点的每一个输入。
      //  其对应的引用加以
      LOG(INFO)<<"   inode.inputs.size()   "<<inode.inputs.size();
      for (const auto& e : inode.inputs)
      {

        //  对于一个节点的所有的输入，将其对应的输入的节点的计数加1
        ++ref_count[idx.entry_id(e)];
        LOG(INFO)<<" 对于输入   idx.entry_id(e) ======="<<idx.entry_id(e)<< "   对应的输入的节点的引用计数  "<<ref_count[idx.entry_id(e)] ;
        // 得到这个实体真实的ID
      }
      /* for (const auto& e : inode.outputs)
      {

        //  对于一个节点的所有的输入，将其对应的输入的节点的计数加1
        //++ref_count[idx.entry_id(e)];
        LOG(INFO)<<" 对于输出   idx.entry_id(e) ======="<<idx.entry_id(e)<< "   对应的输入的节点的引用计数  "<<ref_count[idx.entry_id(e)] ;
        // 得到这个实体真实的ID
      }
      */
      // no dataflow dependency is needed for those are ignored.
      // revoke the dependency counter.
      int  a7=0;
      if (fignore_inputs.count(inode.source->op()) != 0) 
      {
         
        auto ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
        //对于所以需要忽略的属性值，加一
        for (uint32_t i : ignore_inputs) 
         {
            --ref_count[idx.entry_id(inode.inputs[i])];
            a7++;
         }
       }
      LOG(INFO)<<" fignore_inputs.count   这个节点对应的可以被忽略的输入的节点 " <<a7;
      LOG(INFO)<<"***********************************************************************************";
     }
     //    最后对于图里面的每一输出
     //    自带的引用计数
     int  output_num=0; 
     for (const auto& e : idx.outputs())
     {
       output_num++;
       ++ref_count[idx.entry_id(e)];
       LOG(INFO)<<"对于每一个最后的输出结果   "<<idx.entry_id(e)<<"   对应的引用计数为  "<<ref_count[idx.entry_id(e)];

     }
      /*      [04:56:39] src/pass/plan_memory.cc:539: 对于每一个最后的输出结果   13   对应的引用计数为  2
              [04:56:39] src/pass/plan_memory.cc:539: 对于每一个最后的输出结果   24   对应的引用计数为  1
              [04:56:39] src/pass/plan_memory.cc:539: 对于每一个最后的输出结果   25   对应的引用计数为  1
              [04:56:39] src/pass/plan_memory.cc:539: 对于每一个最后的输出结果   26   对应的引用计数为  1
              [04:56:39] src/pass/plan_memory.cc:539: 对于每一个最后的输出结果   21   对应的引用计数为  1
              [04:56:39] src/pass/plan_memory.cc:539: 对于每一个最后的输出结果   22   对应的引用计数为  1
              [04:56:39] src/pass/plan_memory.cc:539: 对于每一个最后的输出结果   17   对应的引用计数为  1
              [04:56:39] src/pass/plan_memory.cc:539: 对于每一个最后的输出结果   18   对应的引用计数为  1 
      */
    //  8
    LOG(INFO)<<"索引图的输出结果个数为"<<output_num;
  }

  //  我们最终输出索引图std::vector<uint32_t> ref_count;
  for(int i=0;i<ref_count.size();i++)
  {
      LOG(INFO)<<"第 个节点  "<< i<<"对应的引用计数为：："<<ref_count[i];
  }
  /* 
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  0对应的引用计数为：：2
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  1对应的引用计数为：：2
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  2对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  3对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  4对应的引用计数为：：3
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  5对应的引用计数为：：2
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  6对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  7对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  8对应的引用计数为：：3
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  9对应的引用计数为：：2
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  10对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  11对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  12对应的引用计数为：：2
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  13对应的引用计数为：：2
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  14对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  15对应的引用计数为：：0
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  16对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  17对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  18对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  19对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  20对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  21对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  22对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  23对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  24对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  25对应的引用计数为：：1
    [04:56:39] src/pass/plan_memory.cc:549: 第 个节点  26对应的引用计数为：：1 
  */

  // step 2: allocate memory.
  StorageVector storage;
  if (ret.attrs.count("storage") != 0) 
  {  
    storage = ret.MoveCopyAttr<StorageVector>("storage");
  } 
  else 
  {
     // 27
     LOG(INFO)<<"idx.num_node_entries()"<<idx.num_node_entries();
     storage.resize(idx.num_node_entries(), -1);
  }

  // Search the best NNVM_EXEC_MATCH_RANGE parameter. This is turned off by default
  // 是不是按照范围进行匹配？？？？
  size_t min_allocated_bytes = -1;
  size_t max_match_range = dmlc::GetEnv("NNVM_EXEC_MATCH_RANGE", 16);
  size_t min_match_range = dmlc::GetEnv("NNVM_AUTO_SEARCH_MATCH_RANGE", false) ? 1 : max_match_range;
  //  最大是16  最小是1
  //  也就是说子啊不断的变换着分配的范围，哪一种分配需要的内存最少，就按照哪一种进行分配。
  //  按照指数的范围进行内存划分

  for (size_t match_range = min_match_range; match_range <= max_match_range; match_range *= 2) 
  {
    // Make a copy of related fields
    // 16
    // 从最小的范围到最大的范围，每次增加一倍。
    LOG(INFO)<<"match_range"<<match_range;
    StorageVector storage_vec(storage);
    LOG(INFO)<<"idx.num_node_entries()"<<idx.num_node_entries();
    std::vector<int> storage_inplace_index(idx.num_node_entries(), -1);
    // the allocator
    // 按照分配这个范围要求去分配。。。。。。
    // 构造一个分配器
    GraphAllocator allocator(&idx, match_range);
    // number of entries that are not statically allocated.
    // 存储类型，哪一些节点是原地操作？引用计数向量。分配器
    // 无法分配的节点
    size_t storage_num_not_allocated=AllocMemory(ret, idx, node_range, &storage_vec, &storage_inplace_index,ref_count, &allocator);
    // 0
    LOG(INFO)<<"storage_num_not_allocated 没有静态分配的节点"<<storage_num_not_allocated;
    size_t storage_allocated_bytes = allocator.TotalAllocBytes();
    //总的分配的内存的大小。
    // 1364
    LOG(INFO)<<"storage_allocated_bytes"<<storage_allocated_bytes;
    
    // Choose the plan which leads to minimal memory usage
    // 遍历不同的分配范围，选择分配内存需求最少的方案。
    if (min_allocated_bytes > storage_allocated_bytes) 
    {
      ret.attrs["storage_id"] = std::make_shared<any>(std::move(storage_vec));
      // 所有的原地的索引
      ret.attrs["storage_inplace_index"] = std::make_shared<any>(std::move(storage_inplace_index));
      //  所有的
      ret.attrs["storage_allocated_bytes"] = std::make_shared<any>(storage_allocated_bytes);

      ret.attrs["storage_num_not_allocated"] = std::make_shared<any>(storage_num_not_allocated);

      min_allocated_bytes = storage_allocated_bytes;
    }

    if (max_match_range == 0)
     {
      break;
    }
  }
  return ret;
}

NNVM_REGISTER_PASS(PlanMemory)
.describe("Plan the memory allocation of each node entries.")
.set_body(PlanMemory)
.set_change_graph(false)
.depend_graph_attr("dtype")
.depend_graph_attr("shape")
.provide_graph_attr("storage_id")
.provide_graph_attr("storage_inplace_index");

}  // namespace
}  // namespace pass
}  // namespace nnvm
