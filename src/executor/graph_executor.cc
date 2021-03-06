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
 *  Copyright (c) 2015 by Contributors
 * \file graph_executor.cc
 * \brief graph executor
 */
#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>
#include <vector>
#include <algorithm>

#include "./exec_pass.h"
#include "./graph_executor.h"
#include "../profiler/profiler.h"
#include "../common/utils.h"
#include "../common/exec_utils.h"

namespace mxnet {
namespace exec {

GraphExecutor::GraphExecutor() 
{
  LOG(INFO)<<"进入GraphExecutor()";
  log_verbose_ = dmlc::GetEnv("MXNET_EXEC_VERBOSE_LOGGING", false);
}

GraphExecutor::~GraphExecutor() 
{
  LOG(INFO)<<"进入~GraphExecutor()";
  for (auto& n : op_nodes_) {
    if (n.cached_opr != nullptr) {
      Engine::Get()->DeleteOperator(n.cached_opr);
    }
  }
  // clean up seg ops
  for (auto& seg : cached_seg_opr_) {
    if (seg.opr != nullptr) {
      Engine::Get()->DeleteOperator(seg.opr);
    }
  }
}

inline NDArray InitZeros(const NDArrayStorageType stype, const TShape &shape,
                                const Context &ctx, const int dtype)
 {
   LOG(INFO)<<"进入InitZerosshuch";
  // NDArray with default storage

  if (stype == kDefaultStorage)
   {
    NDArray ret(shape, ctx, false, dtype);
    ret = 0;
    return ret;
  }
   //   NDArray with non-default storage. Storage allocation is always delayed.
   //   采用延迟分布的策略
  return NDArray(stype, shape, ctx, true, dtype);
}

inline void EmplaceBackZeros(const NDArrayStorageType stype, const TShape &shape,
                             const Context &ctx, const int dtype,
                             std::vector<NDArray> *vec)
 {
  // NDArray with default storage
  if (stype == kDefaultStorage) 
  {
    // 减少一次默认的分配的问题
    vec->emplace_back(shape, ctx, false, dtype);
    vec->back() = 0;
  } else 
  {
    // NDArray with non-default storage. Storage allocation is always delayed.

    vec->emplace_back(stype, shape, ctx, true, dtype);
  }
}
void GraphExecutor::Forward(bool is_train) 
{

  RunOps(is_train, 0, num_forward_nodes_);
}

void GraphExecutor::PartialForward(bool is_train, int step, int *step_left) {
  size_t sstep = static_cast<size_t>(step);
  if (sstep >= num_forward_nodes_) {
    *step_left = 0; return;
  }
  RunOps(is_train, sstep, sstep + 1);
  *step_left = static_cast<int>(num_forward_nodes_ - sstep - 1);
}

void GraphExecutor::Backward(const std::vector<NDArray>& head_grads, bool is_train) {
  const auto& idx = graph_.indexed_graph();
  if (num_forward_inputs_ != idx.input_nodes().size()) {
    for (size_t i = 0; i < head_grad_array_.size(); ++i) {
      if (!head_grad_array_[i].is_none()) {
        CHECK(i < head_grads.size() && !head_grads[i].is_none())
            << "Because the last operator is not Loss function, "
            << "head_gradient is required when calling backward. "
            << "If you are attempting to minimize the output as "
            << "an objective, please modify your network and "
            << "pass it through the make_loss symbol.";
        CopyFromTo(head_grads[i], &(head_grad_array_[i]));
      }
    }
  }
  RunOps(is_train, num_forward_nodes_, idx.num_nodes());
}

void GraphExecutor::Print(std::ostream &os) const {  // NOLINT(*)
  nnvm::Symbol s; s.outputs = graph_.outputs;
  s.Print(os);
  // message to be backward compatible with the memonger
  size_t total_bytes = graph_.GetAttr<size_t>("storage_allocated_bytes");
  os << "Total " << (total_bytes >> 20UL) <<" MB allocated\n";
  os << "Total " << 11 << " TempSpace resource requested\n";
}

void GraphExecutor::SetMonitorCallback(const MonitorCallback& callback) {
  CHECK(callback) << "invalid callback";
  monitor_callback_ = callback;
}

const std::vector<NDArray>& GraphExecutor::outputs() const {
  return output_arrays_;
}

const std::unordered_map<std::string, NDArray>& GraphExecutor::in_arg_map() const {
  return in_arg_map_;
}

const std::unordered_map<std::string, NDArray>& GraphExecutor::arg_grad_map() const {
  return arg_grad_map_;
}

const std::unordered_map<std::string, NDArray>& GraphExecutor::aux_state_map() const {
  return aux_state_map_;
}

static nnvm::NodeEntry AttrHint(nnvm::NodeEntry src, nnvm::NodeEntry like) 
{
  static const Op* id_like = Op::Get("_identity_with_attr_like_rhs");
  nnvm::NodePtr n = nnvm::Node::Create();
  n->attrs.op = id_like;
  n->attrs.name = src.node->attrs.name + "_id";
  n->inputs = {src, like};
  return nnvm::NodeEntry{n, 0, 0};
}

nnvm::NodeEntry AggregateGradient(std::vector<nnvm::NodeEntry>&& v) {
  using nnvm::Op;
  static size_t inplace_sum_cap = dmlc::GetEnv("MXNET_EXEC_INPLACE_GRAD_SUM_CAP", 8);
  static const Op* ewise_plus_op = Op::Get("_grad_add");
  static const Op* ewise_sum_op = Op::Get("ElementWiseSum");
  static const Op* identity_op = Op::Get("identity");
  static const Op* zeros_op = Op::Get("_zeros");
  static const Op* zeros_like_op = Op::Get("zeros_like");

  if (v.empty()) 
  {
    nnvm::NodePtr ng = nnvm::Node::Create();
    ng->attrs.op = zeros_op;
    ng->attrs.name = "zeros";
    ng->attrs.op->attr_parser(&(ng->attrs));
    return nnvm::NodeEntry{ng, 0, 0};
  }

  // remove zero in the sum. at least keep 1.
  auto begin = std::remove_if(v.begin(), v.end(), [](const nnvm::NodeEntry& nodeEntry) {
     return nodeEntry.node->op() == zeros_op || nodeEntry.node->op() == zeros_like_op;
  });
  if (begin == v.begin()) ++begin;
  v.erase(begin, v.end());
  CHECK(!v.empty());

  if (v.size() == 1) {
    return std::move(v[0]);
  } else {
    if (v.size() < inplace_sum_cap) {
      nnvm::NodePtr sum_node = nnvm::Node::Create();
      sum_node->attrs.op = ewise_sum_op;
      sum_node->attrs.name = "sum_grad";
      sum_node->attrs.dict["num_args"] = std::to_string(v.size());
      sum_node->attrs.op->attr_parser(&(sum_node->attrs));
      sum_node->inputs = std::move(v);
      return nnvm::NodeEntry{sum_node, 0, 0};
    } else {
      // use a stream line of plus instead
      nnvm::NodeEntry ret = v[0];
      for (size_t i = 1; i < v.size(); ++i) {
        // Add control flow dependency from to previous node
        // This enforces the gradient sum order will be in the inverse
        // order of forward traversal
        // NOTE: adding control dependency can be dangerous and cause cycle in the dep.
        // The curent usage is correct, because of the following invariant:
        // assert: v[i-1] do not depend on v[i]
        // To put in plain text: v is gradient vector that get pushed in the order
        // that can generate them, which means if v[i] is not yet pushed,
        // all previous gradient cannot depend on it.
        // Note: For a symbol like the following:
        // data = mx.sym.Variable('data')
        // sym = data + data + data + data + data + data + data
        // the node entries v passed in here are of the same node of
        // op _identity_with_attr_like_rhs. We should skip adding a node
        // to its own control_deps.
        if (v[i-1].node != v[i].node) {
          v[i].node->control_deps.push_back(ret.node);
        }

        std::ostringstream os;
        os << "sum_grad_" << i;
        nnvm::NodePtr x = nnvm::Node::Create();
        x->attrs.op = ewise_plus_op;
        x->attrs.name = os.str();
        x->inputs = {ret, v[i]};
        ret = nnvm::NodeEntry{x, 0, 0};
      }
      // identity node is used to avoid exposure of dummy plus node
      // when its output get assigned to another space.
      nnvm::NodePtr id_node = nnvm::Node::Create();
      id_node->attrs.op = identity_op;
      id_node->attrs.name = "sum_grad_final";
      id_node->inputs = {ret};
      return nnvm::NodeEntry{id_node, 0, 0};
    }
  }
}

template<typename ValueType>
inline ValueType get_node_attr(
    const nnvm::Node& node,
    const std::string& key, ValueType default_value) {
  auto it = node.attrs.dict.find(key);
  if (it == node.attrs.dict.end()) {
    return default_value;
  } else {
    ValueType ret;
    dmlc::parameter::FieldEntry<ValueType> e;
    e.Init(key, &ret, ret);
    e.Set(&ret, it->second);
    return ret;
  }
}

/*!为反向计算构造反向图
 * \brief Create the graph for backward pass.
 * This is triggered by both simple_bind and bind flows.
 */
nnvm::Graph GraphExecutor::InitFullGraph(nnvm::Symbol symbol,
                                         const std::vector<OpReqType>& grad_req_types) 
{
  LOG(INFO)<<"进入InitFullGraph";
  //  std::shared_ptr<Node>
  //  一个节点的智能指针
  using nnvm::NodePtr;
        /*! \brief an entry that represents output data from a node */
        //struct NodeEntry {
        /*! \brief the source node of this data */
          //NodePtr node;
      /*! \brief index of output from the source. */
      //uint32_t index;
        /*!
      * \brief version of input Variable.
      *  This field can only be nonzero when this->node is a Variable node.
      *  version is increased by one each time a Variable get composed to a mutation Op.
      *  This information can be helpful to decide order of operations when sequence of mutation happens.
      */
      //uint32_t version;
      //};
      // 我们知道齐核心的本质就是：表示一个节点的输出的数据。
      // 会记录这个数据是从哪个节点得到的，节点的指针
      // 是这个节点的输出的第几个数据
  using nnvm::NodeEntry;
  // initial information
  // 注意符号的listInput函数的作用
  LOG(INFO)<<"进入InitFullGraph";
  // 最终的符号的输出
  num_forward_outputs_ = symbol.outputs.size();
  //  1
  LOG(INFO)<<"num_forward_outputs_"<<num_forward_outputs_;
  //  symbol  一般来说表示的是一个图里面的最后的一个节点，
  // 所有的输入数据，包括w b  X  label  总共8个
  num_forward_inputs_ = symbol.ListInputs(nnvm::Symbol::kAll).size();
  //  8
  LOG(INFO)<<"num_forward_inputs_"<<num_forward_inputs_;
  //  
  nnvm::Graph g;

  //  图的输出等于的符号的输出。
  g.outputs = symbol.outputs;
  bool need_grad = false;
  //  请求类型是对于参数的，而不是对于节点的。
  //  这一点需要明白
  //  只要有一个需要梯度写回南无需要反向传播
  for (OpReqType req : grad_req_types) 
  {
    
    LOG(INFO)<<"req====="<<req;
    if (req != kNullOp) 
    need_grad = true;
  }
  //
      /*     03:28:40] src/executor/graph_executor.cc:299: req=====1
            [03:28:40] src/executor/graph_executor.cc:299: req=====1
            [03:28:40] src/executor/graph_executor.cc:299: req=====1
            [03:28:40] src/executor/graph_executor.cc:299: req=====1
            [03:28:40] src/executor/graph_executor.cc:299: req=====1
            [03:28:40] src/executor/graph_executor.cc:299: req=====1
            [03:28:40] src/executor/graph_executor.cc:299: req=====1
            [03:28:40] src/executor/graph_executor.cc:299: req=====0 

            std::vector<NDArray> arg_grad_store;
            arg_grad_store.push_back(NDArray());  // we don't need the grad of the input
            arg_grad_store.push_back(array_w_1_g);
            arg_grad_store.push_back(array_b_1_g);
            arg_grad_store.push_back(array_w_2_g);
            arg_grad_store.push_back(array_b_2_g);
            arg_grad_store.push_back(NDArray());  // neither do we need the grad of the loss
      */
  //
  if (!need_grad) return g;
  //对于每一个输出分配输出节点梯度节点
  // 1 
  LOG(INFO)<<"g.outputs.size()===="<<g.outputs.size();
  for (size_t i = 0; i < g.outputs.size(); ++i) 
  {
    //  uint32_t node_id;   新建一个节点
    //  uint32_t index;     第几个输出0
    //  uint32_t version;   版本号微0
    //   创建一个节点实体
    NodeEntry ngrad{nnvm::Node::Create(), 0, 0};
    //  
    /*
    static nnvm::NodeEntry AttrHint(nnvm::NodeEntry src, nnvm::NodeEntry like) 
    {
    static const Op* id_like = Op::Get("_identity_with_attr_like_rhs");
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = id_like;
    n->attrs.name = src.node->attrs.name + "_id";
    n->inputs = {src, like};
    return nnvm::NodeEntry{n, 0, 0};
    }
    */
    head_grad_entry_.emplace_back(AttrHint(ngrad, g.outputs[i]));
    //  第几个输出的数具实体索引
    head_grad_map_[ngrad.node.get()] = i;

  }
  //  得到所有的输入参数，包括输入data   label 
  std::vector<NodePtr> args = symbol.ListInputs(nnvm::Symbol::kReadOnlyArgs);
  // 8
  LOG(INFO)<<"args = symbol.ListInputs(nnvm::Symbol::kReadOnlyArgs);之后参数的个数="<<args.size();
  std::vector<NodeEntry> xs;
  // 8
  LOG(INFO)<<"grad_req_types.size()"<<grad_req_types.size();

  //  XS表示需要梯度的参数
  //  等于7个
  //  xs 是数据实体，保存了产生他们的数据节点。
  for (size_t i = 0; i < grad_req_types.size(); ++i) 
  {
    if (grad_req_types[i] != kNullOp) 
    {
      //   XS，     包含了所有需要计算梯度的数据的实体。
      //   注意不是节点， 一定要明白的。
      //   从这里面看来。mxnet 数据是采用了区分。但是计算并没有细粒度
      //    将每一个加入到XS里面，新建一个
      xs.emplace_back(NodeEntry{args[i], 0, 0});
    }
  }

  int do_mirror = dmlc::GetEnv("MXNET_BACKWARD_DO_MIRROR", 0);
  auto need_mirror = [do_mirror](const nnvm::Node& node) -> int {
    if (node.is_variable()) return 0;
    const std::string& type = node.attrs.op->name;
    if (type == "Dropout") return false;
    if (get_node_attr(node, "__force_mirroring__", false)) return true;
    if (do_mirror == 0) return false;
    if (type == "Convolution") return false;
    if (type == "FullyConnected") return false;
    if (type == "Concat") return false;
    if (type == "SoftmaxOutput") return false;
    if (type == "BatchNorm") return false;
    if (type == "CuDNNBatchNorm") return false;
    return true;
  };

  std::vector<const nnvm::Op*> zero_ops;
  zero_ops.push_back(nnvm::Op::Get("zeros_like"));
  zero_ops.push_back(nnvm::Op::Get("_zeros"));

  // take gradient
  //  生成梯度的位置核心子啊这个函数。

  LOG(INFO)<<" nnvm::Graph g_grad = nnvm::pass::Gradient";
  //  实际上是生成梯度节点，我们只是知道对应的参数需要计算梯度
  //  那么对应的梯度是如何计算呢？
  //  反向的图是什么样子呢？？？
  //  我们尝试一下哈。
  //  根据是不是需要计算梯度。从而得到对应反向计算的图。主要是额米一个节点。
  // symbol.outputs 其实就是最后的损失。
  // XS 需要计算梯度的数据实体。
  // head_grad_entry_ 就是输出的数据的梯度实体
  //

  nnvm::Graph g_grad = nnvm::pass::Gradient( g, symbol.outputs, xs, head_grad_entry_,
      AggregateGradient, need_mirror, nullptr,
      zero_ops, "_copy");
  // 至此自动生成的完成的反向的计算的节点生成了。
  //  1
  LOG(INFO)<<"加入梯度之前"<<g.outputs.size();
  //  7
  LOG(INFO)<<"g_grad.outputs.size()"<<g_grad.outputs.size();
  //  这2个显然应该相等啊
  //  为每一个需要求导的数据的实体计算出对应的梯度的实体
  CHECK_EQ(g_grad.outputs.size(), xs.size());

  //  对于每一个梯度的输出
  //  加入到图的输出里面去
  for (const auto &e : g_grad.outputs) 
  {
    // LOG(INFO)<<"g.outputs.push_back(e)里面的const auto &e : g_grad.outputs"<<e;
    g.outputs.push_back(e);
  }
  // 图的输出由1个变成了8 包括一个输出节点损失。
  // 损失的梯度，w0 b0 w1  b1  w2  b2 总共8个输出。
  LOG(INFO)<<"加入梯度之前"<<g.outputs.size();
  return g;
}

/*!
 * \brief Assign context to the graph.
 * This is triggered by both simple_bind and bind flows.
 */
static Graph AssignContext(Graph g,
                    const Context& default_ctx,
                    const std::map<std::string, Context>& ctx_map,
                    const std::vector<Context>& in_arg_ctxes,
                    const std::vector<Context>& arg_grad_ctxes,
                    const std::vector<Context>& aux_state_ctxes,
                    const std::vector<OpReqType>& grad_req_types,
                    size_t num_forward_inputs,
                    size_t num_forward_outputs) 
{
  LOG(INFO)<<"进入AssignContexth";
  const auto& idx = g.indexed_graph();
  // 0
  const auto& mutable_nodes = idx.mutable_input_nodes();
  // default use default context.
  // 用户默认没有设置上下文
  LOG(INFO)<<"ctx_map.size()"<<ctx_map.size();
  //  刚开始===0
  //  所以实际上做到这一步就结束了。我们并没有显示的制定上下文。
  //  由于上下文数据是为空的
  if (ctx_map.size() == 0) 
  {
    // 采用默认上下文的
    g.attrs["context"] = std::make_shared<nnvm::any>(
        ContextVector(idx.num_nodes(), default_ctx));
    for (const auto& x : in_arg_ctxes)
    {
      CHECK(x == default_ctx)
        << "Input array is in " << x << " while binding with ctx=" << default_ctx
        << ". All arguments must be in global context (" << default_ctx
        << ") unless group2ctx is specified for cross-device graph.";
    }
    for (const auto& x : arg_grad_ctxes) 
    {
      CHECK(x == default_ctx)
        << "Gradient array is in " << x << " while binding with ctx="
        << default_ctx << ". All gradients must be in global context (" << default_ctx
        << ") unless group2ctx is specified for cross-device graph.";
    }
    return g;
  }

  // otherwise, use context assignment.
  std::map<Context, int>    ctx2id;                  // map ctx to device id     设备到ID GPU0--->0
  std::vector<Context>      ctx_list;                // index is device id       
  nnvm::DeviceVector device(idx.num_nodes(), -1);  // index is node id         节点和设备ID
  nnvm::DeviceAssignMap device_map;                // map arg name to device id

  //    loop through the user input ctx_map and
  //    populate maps and lists
  //    

  LOG(INFO)<<" for (auto &kv : ctx_map) ";
  // 参数名字到山下文的映射
  //const std::map<std::string, Context>& ctx_map,
  for (auto &kv : ctx_map) 
  {
    // 
    LOG(INFO)<<kv.first;

    if (ctx2id.count(kv.second) == 0) 
    { 
      // if context has no device id, create one
      // 
      ctx2id[kv.second] = static_cast<int>(ctx_list.size());  // assign device id to ctx
      ctx_list.push_back(kv.second);  // save ctx to the list
    }

    // assign device id to to the arg name with the corresponding ctx
    //
    LOG(INFO)<<"device_map[kv.first] = ctx2id.at(kv.second);"<<kv.first<<"----"<<ctx2id.at(kv.second);
    device_map[kv.first] = ctx2id.at(kv.second);


  }

  // loop through all the rest of input nodes not specified
  // in the ctx_map and populate maps and lists
  
  size_t arg_top = 0, aux_top = 0;
  for (size_t i = 0; i < num_forward_inputs; ++i) 
  {
    const uint32_t nid = idx.input_nodes().at(i);
    Context ctx;
    if (mutable_nodes.count(nid)) 
    {  // aux node is mutable
      CHECK_LT(aux_top, aux_state_ctxes.size());
      ctx = aux_state_ctxes[aux_top];
      ++aux_top;
    } else {  // regular input node is immutable
      CHECK_LT(arg_top, in_arg_ctxes.size());
      ctx = in_arg_ctxes[arg_top];
      ++arg_top;
    }
    if (ctx2id.count(ctx) == 0) {  // if the current ctx is not in the map of ctx and device id
      ctx2id[ctx] = static_cast<int>(ctx_list.size());  // assign the current ctx with device id
      ctx_list.push_back(ctx);  // save the current ctx in the list
    }
    device[nid] = ctx2id.at(ctx);  // assign device id to the current node
  }

  // loop through backward input nodes and populate maps and lists
  // the backward input nodes is the gradient of the loss wrt the output
  // 
  size_t arg_grad_offset = 0;
  // keep an offset into the arg_grad_ctxes vector,
  // since g.outputs exclude arg_grad whose req == null
  CHECK_GE(grad_req_types.size(), g.outputs.size() - num_forward_outputs)
           << "insufficient number of grad_reqs";
  for (size_t i = num_forward_outputs; i < g.outputs.size(); ++i, ++arg_grad_offset) 
  {
    while (grad_req_types[arg_grad_offset] == kNullOp) ++arg_grad_offset;
    const uint32_t nid = idx.outputs()[i].node_id;
    Context ctx = arg_grad_ctxes[arg_grad_offset];
    if (ctx2id.count(ctx) == 0) {
      ctx2id[ctx] = static_cast<int>(ctx_list.size());
      ctx_list.push_back(ctx);
    }
    int devid = ctx2id.at(ctx);
    if (device[nid] != -1) {
      CHECK_EQ(device[nid], devid) << "device of same output not equal to each other";
    }
     else 
    {
      device[nid] = devid;
    }
  }

  g.attrs["device"] = std::make_shared<dmlc::any>(std::move(device));

  g = nnvm::pass::PlaceDevice(g, "__ctx_group__", device_map, "_CrossDeviceCopy");

  const auto& assigned_device = g.GetAttr<nnvm::DeviceVector>("device");
  // 得到最终的分配结果

  ContextVector vcontext;
  LOG(INFO)<<"assigned_device.size()"<<assigned_device.size();
  for (size_t i = 0; i < assigned_device.size(); ++i) 
  {
     LOG(INFO)<<"assigned_device[i]"<<assigned_device[i];
    if (assigned_device[i] == -1) 
    {
      vcontext.push_back(default_ctx);
    } 
    else 
    {
      vcontext.push_back(ctx_list[assigned_device[i]]);
    }
  }

  // after device planning, we should check again
  // if the assigned device of gradient node
  // corresponds to storage of grads

  auto &new_idx = g.indexed_graph();

  arg_grad_offset = 0;
  
  for (size_t i = num_forward_outputs; i < g.outputs.size(); ++i, ++arg_grad_offset) 
  {
    // 跳过不需要的反响计算梯度的输入参数
    while (grad_req_types[arg_grad_offset] == kNullOp)
           ++arg_grad_offset;
    LOG(INFO)<<" arg_grad_offset"<< arg_grad_offset; 
    // 获取到这个反向节点的ID
    const uint32_t nid = new_idx.outputs()[i].node_id;

    LOG(INFO)<<"const uint32_t nid = new_idx.outputs()[i].node_id; "<<nid;

    // 获取到这个梯度的上下文。
    Context ctx = arg_grad_ctxes[arg_grad_offset];
    // 若果这个分配到的上下文和vcontext[nid]分配的到的上下文不一致的话


  //  nihao  nihao nihao 

  // 
    CHECK(ctx == vcontext[nid])
      << "Trying to save gradient to " << ctx
      << " while its source node \"" << new_idx[nid].source->attrs.name
      << "\" computes it on " << vcontext[nid]
      << ". Check your ctx in NDArray allocation.";
  }

  g.attrs["context"] = std::make_shared<nnvm::any>(std::move(vcontext));

  return g;
}

static void HandleInferShapeError(const size_t num_forward_inputs,
                           const nnvm::IndexedGraph& idx,
                           const nnvm::ShapeVector& inferred_shapes) {
  int cnt = 10;
  std::ostringstream oss;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const TShape& inferred_shape = inferred_shapes[eid];
    if (inferred_shape.ndim() == 0 || inferred_shape.Size() == 0U) {
      const std::string& arg_name = idx[nid].source->attrs.name;
      oss << arg_name << ": " << inferred_shape << ", ";
      if (--cnt == 0) {
        oss << "...";
        break;
      }
    }
  }
  LOG(FATAL) << "InferShape pass cannot decide shapes for the following arguments "
                "(0s means unknown dimensions). Please consider providing them as inputs:\n"
             << oss.str();
}

static void HandleInferTypeError(const size_t num_forward_inputs,
                          const nnvm::IndexedGraph& idx,
                          const nnvm::DTypeVector& inferred_dtypes) {
  int cnt = 10;
  std::ostringstream oss;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const int inferred_dtype = inferred_dtypes[eid];
    if (inferred_dtype == -1) {
      const std::string& arg_name = idx[nid].source->attrs.name;
      oss << arg_name << ": " << inferred_dtype << ", ";
      if (--cnt == 0) {
        oss << "...";
        break;
      }
    }
  }
  LOG(FATAL) << "InferType pass cannot decide dtypes for the following arguments "
                "(-1 means unknown dtype). Please consider providing them as inputs:\n"
             << oss.str();
}

static void HandleInferStorageTypeError(const size_t num_forward_inputs,
                                 const nnvm::IndexedGraph& idx,
                                 const StorageTypeVector& inferred_stypes) {
  int cnt = 10;
  std::ostringstream oss;
  for (size_t i = 0; i < num_forward_inputs; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const int inferred_stype = inferred_stypes[eid];
    if (inferred_stype == -1) {
      const std::string& arg_name = idx[nid].source->attrs.name;
      oss << arg_name << ": " << common::stype_string(inferred_stype) << ", ";
      if (--cnt == 0) {
        oss << "...";
        break;
      }
    }
  }
  LOG(FATAL) << "InferStorageType pass cannot decide storage type for the following arguments "
                "(-1 means unknown stype). Please consider providing them as inputs:\n"
             << oss.str();
}

/*!
 * \brief GraphExecutor initializer for regular bind flow in which
 * input arguments and gradients are provided by users. This initializer
 * uses the user provided NDArrays to populate data entries of the graph.
 */
void GraphExecutor::Init(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& ctx_map,
                         const std::vector<NDArray>& in_args,
                         const std::vector<NDArray>& arg_grad_store,
                         const std::vector<OpReqType>& grad_req_types,
                         const std::vector<NDArray>& aux_states,
                         Executor* shared_exec,
                         const nnvm::NodeEntryMap<NDArray>& feed_dict) {
  // create in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes
  LOG(INFO)<<"进入 GraphExecutor::Init  本质上进入这个。。。。。。。。。。。";
  // 下面是匿名函数。
  auto get_ctx1 = [](const NDArray& nd) { return nd.ctx(); };
  auto get_ctx2 = [default_ctx](const NDArray& nd) -> Context {
    if (nd.is_none()) return default_ctx;
    return nd.ctx();
  };
  // 参数上下文
  //  8
  LOG(INFO)<<"进入进入 GraphExecutor::Init    in_args.size()"<<in_args.size();
  std::vector<Context> in_arg_ctxes(in_args.size());
  LOG(INFO)<<"in_args.size()   "<<in_args.size();
  //  每一个参数，迭代获取上下文。
  std::transform(in_args.begin(), in_args.end(), in_arg_ctxes.begin(), get_ctx1);
  //    存储梯度的上下文
  LOG(INFO)<<"arg_grad_store.size()"<<arg_grad_store.size();
  std::vector<Context> arg_grad_ctxes(arg_grad_store.size());
  std::transform(arg_grad_store.begin(), arg_grad_store.end(), arg_grad_ctxes.begin(), get_ctx2);
  // 辅助空间上下文
  LOG(INFO)<<"aux_states.size()  "<<aux_states.size();
  std::vector<Context> aux_state_ctxes(aux_states.size());
  std::transform(aux_states.begin(), aux_states.end(), aux_state_ctxes.begin(), get_ctx1);
  // 我们看一下InitGraph
  nnvm::Graph g = InitGraph(symbol, default_ctx, ctx_map, in_arg_ctxes,
                            arg_grad_ctxes, aux_state_ctxes, grad_req_types);

  LOG(INFO)<<"InitGraph回到init";
  // create arg_shapes and arg_dtypes for shape and type inferences
  const auto& idx = g.indexed_graph();
  const auto& mutable_nodes = idx.mutable_input_nodes();
   //  0 
  LOG(INFO)<<" mutable_nodes数目为"<<mutable_nodes.size();
   //  27 个实体
  size_t arg_top = 0, aux_top = 0;
  LOG(INFO)<<"idx.num_node_entries()实体节点的数目="<<idx.num_node_entries();
  //   27
  data_entry_.resize(idx.num_node_entries());
  //   初始化参数的形状
  nnvm::ShapeVector arg_shapes;
  //   初始化参数的类型
  nnvm::DTypeVector arg_dtypes;
  //   实体的存储类型
  StorageTypeVector arg_stypes(idx.num_node_entries(), -1);
  //   便利每一个输入
  //    8 
  LOG(INFO)<<"num_forward_inputs_"<<num_forward_inputs_;
  // X ,W0,b0,w1 ,b1 ,w2, b2 ,label
  for (size_t i = 0; i < num_forward_inputs_; ++i) 
  {
    //    i= 0,1,2,3,4,5,6,7
    //    nid=0,1,2,5,6,9,10,12  //节点ID
    //    eid=0,1,2,5,6,9.10.12
    //    我们只是便利每一个输入。
    //    每一个输入的节点的第一个输出的节点的实体就是本事的参数
    //    如果这个节点是可以修改的
    //    那么这个节点的第一个输出节点就是辅助的空间的节点了。
    const uint32_t nid = idx.input_nodes().at(i);
    LOG(INFO)<<"  nid=="<<nid;
    //这个节点对应的参数的名字的。
    const std::string& arg_name = idx[nid].source->attrs.name;
    LOG(INFO)<<"对应  arg_name  "<<arg_name;
    //  获取对应的实体ID
    //  获取输出的第一个输出的节点的ID
    //  第一个输出实体
    size_t eid = idx.entry_id(nid, 0);
    LOG(INFO)<<" eid = idx.entry_id(nid, 0)对应eid=="<<eid;
    //  如这个节点需要写
    LOG(INFO)<<"mutable_nodes.count(nid)     结果为;   "<<mutable_nodes.count(nid);
    // 如果这个节点的需要写入的节点数目是？
    if (mutable_nodes.count(nid))
    {      
      CHECK_LT(aux_top, aux_states.size());
      data_entry_[eid] = aux_states[aux_top];
      arg_shapes.push_back(aux_states[aux_top].shape());
      arg_dtypes.push_back(aux_states[aux_top].dtype());
      arg_stypes[eid] = aux_states[aux_top].storage_type();
      aux_state_map_.emplace(arg_name, aux_states[aux_top]);
      ++aux_top;
    }
     else 
    {
      //  这个实体的ID就是对应于这个参数
      CHECK_LT(arg_top, in_args.size());
      // 第一个实体对应的数据就是这个参数的本身。
      data_entry_[eid] = in_args[arg_top];
      arg_shapes.push_back(in_args[arg_top].shape());
      arg_dtypes.push_back(in_args[arg_top].dtype());
      arg_stypes[eid] = in_args[arg_top].storage_type();
      LOG(INFO)<<"   arg_name    "<<arg_name<<"        arg_top     "<<arg_top;
      in_arg_map_.emplace(arg_name, in_args[arg_top]);
      int  aa=0;   
      if (kNullOp != grad_req_types[arg_top])  aa=1;
      LOG(INFO)<<"kNullOp != grad_req_types[arg_top]      "<<aa;
      // 如果这个参数需要计算梯
      if (kNullOp != grad_req_types[arg_top]) 
      { 
        //   依次遍历每一个输出，也就是需要计算的每一个梯度值
        auto grad_oid = grad_store_.size() + num_forward_outputs_;
        //   
        auto grad_eid = idx.entry_id(idx.outputs()[grad_oid]);
        //   让梯度的存储的类型和参数的类型一致。
        arg_stypes[grad_eid] = arg_grad_store[arg_top].storage_type();
        //    请求类型和数组之间的关联。
        grad_store_.emplace_back(grad_req_types[arg_top], arg_grad_store[arg_top]);     
        arg_grad_map_.emplace(arg_name, arg_grad_store[arg_top]);
        LOG(INFO)<<"   arg_top      "<<arg_top<<" grad_oid  "<<grad_oid<<"  grad_eid  "<<grad_eid<<"  grad_store_.size "<<grad_store_.size()<<"  arg_grad_map_  "<<arg_grad_map_.size();
        if (log_verbose_) 
        {
          LOG(INFO) << "\tassign data entry\t" << grad_eid << " as "
                    << common::stype_string(arg_stypes[grad_eid]) << " (grad)";
        }
      }
      ++arg_top;
      LOG(INFO)<<"**********************************************************************************";
    }
    if (log_verbose_) 
    {
      LOG(INFO) << "\tassign data entry\t" << eid << " as "
                << common::stype_string(data_entry_[eid].storage_type()) << " (input)";
    }
  }
  // expand arg_shapes and arg_dtypes to contain backward inputs
  arg_shapes.resize(idx.input_nodes().size(), TShape());
  //  8
  LOG(INFO)<<"  idx.input_nodes().size()     "<<idx.input_nodes().size();
  //  推测每一个中间的节点的数据的形状。
  g = InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) 
  {
    HandleInferShapeError(num_forward_inputs_, g.indexed_graph(),
                          g.GetAttr<nnvm::ShapeVector>("shape"));
  }
  // 参数类型
  // 8
  arg_dtypes.resize(idx.input_nodes().size(), -1);
  // 推测中间数据类型
  g = InferType(std::move(g), std::move(arg_dtypes), "__dtype__"); 
  if (g.GetAttr<size_t>("dtype_num_unknown_nodes") != 0U) 
  {
    HandleInferTypeError(num_forward_inputs_, g.indexed_graph(),
                         g.GetAttr<nnvm::DTypeVector>("dtype"));
  }

  g.attrs["storage_type"] = std::make_shared<dmlc::any>(std::move(arg_stypes));
  
 
  //  推测存储类型
  g = InferStorageType(std::move(g), StorageTypeVector(), "");

  if (g.GetAttr<size_t>("storage_type_num_unknown_nodes") != 0U) {
    HandleInferStorageTypeError(num_forward_inputs_, g.indexed_graph(),
                                g.GetAttr<StorageTypeVector>("storage_type"));
  }
  //  至此我们都知道了。
  //  所有的正向，反向，以及数据，输入输出节点的类型，形状，存储类型都的出来了。

  // Initialize the rest attributes of the graph.
  // This function can be called by regular bind
  // operation flow as well.
  
  FinishInitGraph(symbol, g, shared_exec, feed_dict);


}

/*!
 * \brief Initialize in_args, arg_grads, and aux_states
 * and their data_entry_ of the executor. This function
 * is called for regular simple_bind flow, i.e. no
 * shared data arrays are provided.
 */
void GraphExecutor::InitArguments(const nnvm::IndexedGraph& idx,
                                  const nnvm::ShapeVector& inferred_shapes,
                                  const nnvm::DTypeVector& inferred_dtypes,
                                  const StorageTypeVector& inferred_stypes,
                                  const std::vector<Context>& in_arg_ctxes,
                                  const std::vector<Context>& arg_grad_ctxes,
                                  const std::vector<Context>& aux_state_ctxes,
                                  const std::vector<OpReqType>& grad_req_types,
                                  std::vector<NDArray>* in_arg_vec,
                                  std::vector<NDArray>* arg_grad_vec,
                                  std::vector<NDArray>* aux_state_vec) {
   LOG(INFO)<<"进入  GraphExecutor::InitArguments";                                 
  // initialize in_args, arg_grads, and aux_states
  // populate grad_store_
  data_entry_.resize(idx.num_node_entries());
  size_t arg_top = 0, aux_top = 0;
  const auto& mutable_nodes = idx.mutable_input_nodes();
  for (size_t i = 0; i < num_forward_inputs_; ++i)
   {
     
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const TShape& inferred_shape = inferred_shapes[eid];
    const int inferred_dtype = inferred_dtypes[eid];
    const NDArrayStorageType inferred_stype = (NDArrayStorageType) inferred_stypes[eid];
    const std::string& arg_name = idx[nid].source->attrs.name;
    if (mutable_nodes.count(nid)) {  // aux_states
      EmplaceBackZeros(inferred_stype, inferred_shape, aux_state_ctxes[aux_top],
                       inferred_dtype, aux_state_vec);
      data_entry_[eid] = aux_state_vec->back();
      aux_state_map_.emplace(arg_name, aux_state_vec->back());
      ++aux_top;
      if (log_verbose_) {
        LOG(INFO) << "\tassign aux entry\t" << eid << "\t as "
                  << common::stype_string(inferred_stype);
      }
    } else {  // in_args
      EmplaceBackZeros(inferred_stype, inferred_shape, in_arg_ctxes[arg_top],
                       inferred_dtype, in_arg_vec);
      data_entry_[eid] = in_arg_vec->back();
      if (log_verbose_) {
        LOG(INFO) << "\tassign data entry\t" << eid << "\tas "
                  << common::stype_string(inferred_stype);
      }
      // Get the storage type for grad
      if (kNullOp == grad_req_types[arg_top]) {
        arg_grad_vec->emplace_back();
      } else {
        // Init based on storage type
        auto grad_oid = grad_store_.size() + num_forward_outputs_;
        auto grad_eid = idx.entry_id(idx.outputs()[grad_oid]);
        auto grad_stype = (NDArrayStorageType) inferred_stypes[grad_eid];
        EmplaceBackZeros(grad_stype, inferred_shape, arg_grad_ctxes[arg_top],
                         inferred_dtype, arg_grad_vec);
        if (log_verbose_) {
          LOG(INFO) << "\tassign grad entry\t" << grad_eid << "\tas "
                    << common::stype_string(grad_stype);
        }
        grad_store_.emplace_back(grad_req_types[arg_top], arg_grad_vec->back());
        arg_grad_map_.emplace(arg_name, arg_grad_vec->back());
      }
      in_arg_map_.emplace(arg_name, in_arg_vec->back());
      ++arg_top;
    }
  }
}

/*!
 * \brief If the requested ndarray's shape size is less than
 * the corresponding shared_data_array's shape size and the
 * storage type is shareable, reuse the memory allocation
 * in shared_buffer; otherwise, create a zero ndarray.
 * Shareable storages include both default storage and row_sparse storage
 * if enable_row_sparse_sharing is `True`, otherwise default storage only.
 */
static NDArray ReshapeOrCreate(const std::string& name,
                        const TShape& dest_arg_shape,
                        const int dest_arg_dtype,
                        const NDArrayStorageType dest_arg_stype,
                        const Context& ctx,
                        std::unordered_map<std::string, NDArray>* shared_buffer,
                        bool enable_row_sparse_sharing) {
  bool stype_shareable = dest_arg_stype == kDefaultStorage;
  if (enable_row_sparse_sharing) {
    stype_shareable = stype_shareable || dest_arg_stype == kRowSparseStorage;
  }
  auto it = shared_buffer->find(name);
  if (it != shared_buffer->end()) {
    // check if size is large enough for sharing
    bool size_shareable = it->second.shape().Size() >= dest_arg_shape.Size();
    if (size_shareable && stype_shareable) {  // memory can be reused
      CHECK_EQ(it->second.dtype(), dest_arg_dtype)
        << "Requested arg array's dtype does not match that of the reusable ndarray";
      CHECK_EQ(it->second.storage_type(), dest_arg_stype)
        << "Requested arg array's stype does not match that of the reusable ndarray";
      return it->second.Reshape(dest_arg_shape);
    } else if (stype_shareable) {
      LOG(WARNING) << "Bucketing: data " << name << " has a shape " << dest_arg_shape
                   << ", which is larger than already allocated shape " << it->second.shape()
                   << ". Need to re-allocate. Consider putting default bucket key to be "
                   << "the bucket taking the largest input for better memory sharing.";
      // size is not large enough, creating a larger one for sharing
      // the NDArrays in shared_buffer are guaranteed to be of shareable storages
      it->second = InitZeros(dest_arg_stype, dest_arg_shape, ctx, dest_arg_dtype);
      return it->second;
    } else {
      // not shareable storage
      return InitZeros(dest_arg_stype, dest_arg_shape, ctx, dest_arg_dtype);
    }
  } else {
    auto ret = InitZeros(dest_arg_stype, dest_arg_shape, ctx, dest_arg_dtype);
    if (stype_shareable) {
      shared_buffer->emplace(name, ret);
    }
    return ret;
  }  // if (it != shared_buffer->end())
}

/*!
 * \brief Initialize in_args, arg_grads, and aux_states
 * and their data_entry_ of the executor using
 * shared_buffer from DataParallelExecutorGroup
 * and shared_exec if available.
 */
void GraphExecutor::InitArguments(const nnvm::IndexedGraph& idx,
                                  const nnvm::ShapeVector& inferred_shapes,
                                  const nnvm::DTypeVector& inferred_dtypes,
                                  const StorageTypeVector& inferred_stypes,
                                  const std::vector<Context>& in_arg_ctxes,
                                  const std::vector<Context>& arg_grad_ctxes,
                                  const std::vector<Context>& aux_state_ctxes,
                                  const std::vector<OpReqType>& grad_req_types,
                                  const std::unordered_set<std::string>& shared_arg_names,
                                  const Executor* shared_exec,
                                  std::unordered_map<std::string, NDArray>* shared_buffer,
                                  std::vector<NDArray>* in_arg_vec,
                                  std::vector<NDArray>* arg_grad_vec,
                                  std::vector<NDArray>* aux_state_vec) {
  LOG(INFO)<<"进入  GraphExecutor::InitArguments";  
  // initialize in_args, arg_grads, and aux_states and populate grad_store_
  data_entry_.resize(idx.num_node_entries());
  size_t arg_top = 0, aux_top = 0;
  const auto& mutable_nodes = idx.mutable_input_nodes();
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const uint32_t eid = idx.entry_id(nid, 0);
    const TShape& inferred_shape = inferred_shapes[eid];
    const int inferred_dtype = inferred_dtypes[eid];
    const NDArrayStorageType inferred_stype = (NDArrayStorageType) inferred_stypes[eid];
    const std::string& arg_name = idx[nid].source->attrs.name;
    // aux_states
    if (mutable_nodes.count(nid)) {
      if (nullptr != shared_exec) {
        const NDArray& aux_nd = shared_exec->aux_state_map().at(arg_name);
        CHECK(inferred_stype == kDefaultStorage && aux_nd.storage_type() == kDefaultStorage)
          << "Non-default storage type detected when creating auxilliary NDArray. The allocated "
          << "memory of shared_exec.aux_array cannot be resued for argument: "
          << arg_name << " for the current executor";
        CHECK_EQ(inferred_shape, aux_nd.shape())
          << "Inferred shape does not match shared_exec.aux_array's shape."
             " Therefore, the allocated memory for shared_exec.aux_array cannot"
             " be resued for creating auxilliary NDArray of the argument: "
          << arg_name << " for the current executor";
        CHECK_EQ(inferred_dtype, aux_nd.dtype())
          << "Inferred dtype does not match shared_exec.aux_array's dtype."
             " Therefore, the allocated memory for shared_exec.aux_array cannot"
             " be resued for creating auxilliary NDArray of the argument: "
          << arg_name << " for the current executor";
        aux_state_vec->emplace_back(aux_nd);
      } else {
        EmplaceBackZeros(inferred_stype, inferred_shape, aux_state_ctxes[aux_top],
                         inferred_dtype, aux_state_vec);
      }  // if (has_shared_exec)
      data_entry_[eid] = aux_state_vec->back();
      aux_state_map_.emplace(arg_name, aux_state_vec->back());
      ++aux_top;
    } else {  // in_args and grad for in_args
      if (shared_arg_names.count(arg_name)) {  // model parameter
        // model parameter
        if (nullptr != shared_exec) {
          const NDArray& in_arg_nd = shared_exec->in_arg_map().at(arg_name);
          auto arg_nd_stype = in_arg_nd.storage_type();
          // for model parameter, both default storage and row_sparse storage can be shared
          bool shareable_arg_stype = inferred_stype == kDefaultStorage ||
                                     inferred_stype == kRowSparseStorage;
          // try to reuse memory from shared_exec
          CHECK(shareable_arg_stype) << "Inferred storage type "
            << common::stype_string(inferred_stype)
            << " does not support memory sharing with shared_exec.arg_array";
          CHECK_EQ(inferred_stype, arg_nd_stype)
            << "Inferred stype does not match shared_exec.arg_array's stype"
               " Therefore, the allocated memory for shared_exec.arg_array cannot"
               " be resued for creating NDArray of the argument "
            << arg_name << " for the current executor";
          CHECK_EQ(inferred_shape, in_arg_nd.shape())
            << "Inferred shape does not match shared_exec.arg_array's shape"
               " Therefore, the allocated memory for shared_exec.arg_array cannot"
               " be resued for creating NDArray of the argument "
            << arg_name << " for the current executor";
          CHECK_EQ(inferred_dtype, in_arg_nd.dtype())
            << "Inferred dtype does not match shared_exec.arg_array's dtype"
               " Therefore, the allocated memory for shared_exec.arg_array cannot"
               " be resued for creating NDArray of the argument "
            << arg_name << " for the current executor";
          in_arg_vec->emplace_back(in_arg_nd);
        } else {
          // doesn't have shared_exec, or non-default storage
          EmplaceBackZeros(inferred_stype, inferred_shape, in_arg_ctxes[arg_top],
                           inferred_dtype, in_arg_vec);
        }
        // gradient for model parameter
        if (kNullOp == grad_req_types[arg_top]) {
          arg_grad_vec->emplace_back();
        } else {
          auto grad_oid = grad_store_.size() + num_forward_outputs_;
          auto grad_eid = idx.entry_id(idx.outputs()[grad_oid]);
          auto grad_stype = (NDArrayStorageType) inferred_stypes[grad_eid];
          if (nullptr != shared_exec && grad_stype == kDefaultStorage &&
              shared_exec->arg_grad_map().at(arg_name).storage_type() == kDefaultStorage) {
            // try to reuse memory from shared_exec
            arg_grad_vec->emplace_back(shared_exec->arg_grad_map().at(arg_name));
          } else {
            // no need to reuse memory from shared_exec for gradient of non-default storage
            EmplaceBackZeros(grad_stype, inferred_shape, arg_grad_ctxes[arg_top],
                             inferred_dtype, arg_grad_vec);
          }
          grad_store_.emplace_back(grad_req_types[arg_top], arg_grad_vec->back());
        }
      } else {  // !shared_arg_names.count(arg_name)
        // model parameter, row_sparse ndarray sharing enabled
        bool enable_row_sparse_sharing = true;
        in_arg_vec->emplace_back(ReshapeOrCreate(arg_name, inferred_shape, inferred_dtype,
                                                 inferred_stype, in_arg_ctxes[arg_top],
                                                 shared_buffer, enable_row_sparse_sharing));
        // gradient for model parameter, row_sparse ndarray sharing disabled
        if (kNullOp == grad_req_types[arg_top]) {
          arg_grad_vec->emplace_back();
        } else {
          auto grad_oid = grad_store_.size() + num_forward_outputs_;
          auto grad_eid = idx.entry_id(idx.outputs()[grad_oid]);
          auto grad_stype = (NDArrayStorageType) inferred_stypes[grad_eid];
          bool enable_row_sparse_sharing = false;
          arg_grad_vec->emplace_back(ReshapeOrCreate("grad of " + arg_name, inferred_shape,
                                                     inferred_dtype, grad_stype,
                                                     arg_grad_ctxes[arg_top], shared_buffer,
                                                     enable_row_sparse_sharing));
          grad_store_.emplace_back(grad_req_types[arg_top], arg_grad_vec->back());
        }  // if (kNullOp == grad_req_types[arg_top])
      }  // if (shared_arg_names.count(arg_name))
      in_arg_map_.emplace(arg_name, in_arg_vec->back());
      if (!arg_grad_vec->back().is_none()) {
        arg_grad_map_.emplace(arg_name, arg_grad_vec->back());
      }
      data_entry_[eid] = in_arg_vec->back();
      ++arg_top;
    }
  }
}

/*!
 * \brief Finish graph initialization after shape and dtype inferences.
 * This function is used by both simple_bind and bind flows.
 */
void GraphExecutor::FinishInitGraph(nnvm::Symbol symbol,
                                    nnvm::Graph g,
                                    Executor* shared_exec,
                                    const nnvm::NodeEntryMap<NDArray>& feed_dict)
 {
  LOG(INFO)<<"进入  GraphExecutor::FinishInitGraph(nnvm::Symbol symbol,"; 
  const auto& idx = g.indexed_graph();
  //  图的属性这样来获取哈。
  const auto& vstorage_type = g.GetAttr<StorageTypeVector>("storage_type");

  //   data entries for output gradients
  //   得到每一个输出梯度的
  //   1  8  表示一个损失的输出，以及7个梯度输出。
  //   本质上，绑定数据实体和对应的存储的数组。
  //   对应需要计算的7个梯度的值
  LOG(INFO)<<"  num_forward_outputs_   "<<num_forward_outputs_<<" idx.outputs().size() "<<idx.outputs().size();
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) 
  {
     LOG(INFO)<< "  j===  "<<j;
     //std::vector<std::pair<OpReqType,NDArray>>
     //std::vector<std::pair<OpReqType,NDArray>> mxnet::exec::GraphExecutor::grad_store_
     //  second  是数组，first   是  对应的类型
     // std::vector<DLTensor> tvm::runtime::GraphRuntime::data_entry_
     // 是是属于tensorflow向量
     //idx.outputs()[j]  第j个输出的实体;
     // 这个ID的实体对应于梯度的存储。
     data_entry_[idx.entry_id(idx.outputs()[j])] = grad_store_[j - num_forward_outputs_].second;
     // 相当于将对于梯度的实体和对应的存储结构对应起来了。
     LOG(INFO)<<" idx.entry_id(idx.outputs()[j])===="<<idx.entry_id(idx.outputs()[j]);
     LOG(INFO)<<"   data_entry_  "<< data_entry_.size();
  }

  {
    // memory allocator
    //  27
    LOG(INFO)<<"  idx.num_node_entries()   "<<idx.num_node_entries();
    nnvm::StorageVector arg_storage_id(idx.num_node_entries(), kBadStorageID);
    // 1
    for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) 
    {
       // 表示每一个梯度表示其存储来自外部，而不是动态分配
       arg_storage_id[idx.entry_id(idx.outputs()[j])] = kExternalStorageID;
    }
    //  对于外部输入的数据。
    int  feednum=0;
    for (const auto& kv : feed_dict)
    {
       uint32_t eid = idx.entry_id(kv.first);
       LOG(INFO)<<" eid =="<<eid;
       data_entry_[eid] = kv.second;
      //
      //LOG(INFO)<<" arg_storage_id 的大小"<<arg_storage_id.size();
      // 这些来自于外部的存储。
      arg_storage_id[eid] = kExternalStorageID;
      feednum++;
    }
    //  =2
    LOG(INFO)<<" feed_dict  数目 =="<<feednum;
    // 27
    LOG(INFO)<<" idx.num_node_entries()  "<<idx.num_node_entries();
    //  不属于默认存储都被认为是动态存储。
    //   采用动态分配的方式。
    for (size_t i = 0; i < idx.num_node_entries(); i++) 
    {
       if (vstorage_type[i] != kDefaultStorage)
            arg_storage_id[i] = kDynamicStorageID;
    }
    //  设定了存储的类型
    g.attrs["storage"] = std::make_shared<dmlc::any>(std::move(arg_storage_id));
    
    //    调用内存规划，内存规划我们知道是采用了引用计数
    //    原地计算， 引用计数的方法。
    //    需要找不不同的关键路径，这些路径可以子啊执行的时候并行执行。

    // 进行内存分配
    LOG(INFO)<<" g = nnvm::ApplyPass(g, planmemory)";
    g = nnvm::ApplyPass(g, "PlanMemory");
  }
   LOG(INFO)<<" g = DetectInplaceAddTo(g);";
   g = DetectInplaceAddTo(g);

  // log the static memory plan of the graph
  static bool mem_log_verbose = dmlc::GetEnv("MXNET_MEM_PLAN_VERBOSE_LOGGING", false);

   /* if (mem_log_verbose) 
   {
    common::LogMemoryPlan(g);
   } */

  common::LogMemoryPlan(g);


  
  LOG(INFO)<< "  g = AttachOpExecs(g);";
  g = AttachOpExecs(g);

  
  LOG(INFO)<<"   AttachOpResources(g);";
  AttachOpResources(g);

  graph_ = std::move(g);


  if (shared_exec != nullptr)
  {
    LOG(INFO)<<"this->InitDataEntryMemory(&(dynamic_cast<GraphExecutor*>(shared_exec)->data_pool_));";
    this->InitDataEntryMemory(&(dynamic_cast<GraphExecutor*>(shared_exec)->data_pool_));
  }
   else 
  {
    LOG(INFO)<<"this->InitDataEntryMemory(nullptr);";
    this->InitDataEntryMemory(nullptr);
  }

  {
    // initialize output arrays
    auto& idx = graph_.indexed_graph();
    for (size_t i = 0; i < num_forward_outputs_; ++i) 
    {
      auto& e = idx.outputs()[i];
      output_arrays_.push_back(data_entry_[idx.entry_id(e)]);
    }
    // initialize head gradient array
    head_grad_array_.resize(symbol.outputs.size());
    for (size_t i = num_forward_inputs_; i < idx.input_nodes().size(); ++i)
    {
       uint32_t nid = idx.input_nodes().at(i);
       uint32_t oid = head_grad_map_.at(idx[nid].source);
       head_grad_array_[oid] = data_entry_[idx.entry_id(nid, 0)];
    }
  }


  LOG(INFO)<<"this->InitCachedOps()";
  //  看看核心这个函数。

  this->InitCachedOps();

  LOG(INFO)<<"this->InitOpSegs();";
  this->InitOpSegs();
}

/*!
 * \brief GraphExecutor initializer for simple bind flow in
 * which only certain input shapes and dtypes are provided by users.
 * The initializer uses these shapes and dtypes to perform
 * shape and dtype inferences, and then create NDArrays
 * to populate data entries of the graph. The created NDArrays
 * for in_args, arg_grads and aux_states are passed to the
 * front end to attach the created executor.
 * In front end, if the simple_bind flow is trigger by
 * _bind_ith_exec, the shared data arrays of DataParallelExecutorGroup
 * and shared executor will be taken into account in creating
 * NDArrays for in_args, arg_grads, and aux_states for resuing
 * already allocated memory.
 */
void GraphExecutor::Init(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& ctx_map,
                         const std::vector<Context>& in_arg_ctxes,
                         const std::vector<Context>& arg_grad_ctxes,
                         const std::vector<Context>& aux_state_ctxes,
                         const std::unordered_map<std::string, TShape>& arg_shape_map,
                         const std::unordered_map<std::string, int>& arg_dtype_map,
                         const std::unordered_map<std::string, int>& arg_stype_map,
                         const std::vector<OpReqType>& grad_req_types,
                         const std::unordered_set<std::string>& shared_arg_names,
                         std::vector<NDArray>* in_arg_vec,
                         std::vector<NDArray>* arg_grad_vec,
                         std::vector<NDArray>* aux_state_vec,
                         std::unordered_map<std::string, NDArray>* shared_buffer,
                         Executor* shared_exec,
                         const nnvm::NodeEntryMap<NDArray>& feed_dict) 
                         {
   LOG(INFO)<<"进入 GraphExecutor::Init"; 
   nnvm::Graph g = InitGraph(symbol, default_ctx, ctx_map, in_arg_ctxes, arg_grad_ctxes,
                            aux_state_ctxes, grad_req_types);
  
  // The following code of shape and dtype inferences and argument
  // initialization is for simple_bind only. Regular bind operation
  // should do this differently.

  // Initialize arg_shapes and arg_dtypes for shape and type inferences.
  // It contains all in_args and aux_states' shapes and types in a certain order.

  LOG(INFO)<<"由InitGraph回到init";
  const nnvm::IndexedGraph& idx = g.indexed_graph();
  // 索引图是如何构造的？？？？？
  nnvm::ShapeVector arg_shapes(idx.input_nodes().size(), TShape());
  nnvm::DTypeVector arg_dtypes(idx.input_nodes().size(), -1);
  StorageTypeVector arg_stypes(idx.input_nodes().size(), kUndefinedStorage);

  for (size_t i = 0; i < num_forward_inputs_; ++i) 
  {
    const uint32_t nid = idx.input_nodes().at(i);
    const std::string& name = idx[nid].source->attrs.name;
    LOG(INFO)<<"输入的参数的名字为"<<name;

    auto it1 = arg_shape_map.find(name);
    
    if (arg_shape_map.end() != it1)
     {
      arg_shapes[i] = it1->second;
    }
    auto it2 = arg_dtype_map.find(name);

    if (arg_dtype_map.end() != it2)
     {
      arg_dtypes[i] = it2->second;
    }
    auto it3 = arg_stype_map.find(name);
    if (arg_stype_map.end() != it3) 
    {
      arg_stypes[i] = it3->second;
    }
  }
  // 推测参数的形状。
  g = InferShape(std::move(g), std::move(arg_shapes), "__shape__");

  if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) 
  {
    HandleInferShapeError(num_forward_inputs_, g.indexed_graph(),
                          g.GetAttr<nnvm::ShapeVector>("shape"));
  }
  // 推测数据类型
  g = InferType(std::move(g), std::move(arg_dtypes), "__dtype__");
  if (g.GetAttr<size_t>("dtype_num_unknown_nodes") != 0U) 
  {
    HandleInferTypeError(num_forward_inputs_, g.indexed_graph(),
                         g.GetAttr<nnvm::DTypeVector>("dtype"));
  }
  // 推测存储类型
  g = InferStorageType(std::move(g), std::move(arg_stypes), "__storage_type__");

  if (g.GetAttr<size_t>("storage_type_num_unknown_nodes") != 0U) 
  {
    HandleInferStorageTypeError(num_forward_inputs_, g.indexed_graph(),
                                g.GetAttr<StorageTypeVector>("storage_type"));
  }

  // Create in_args, arg_grads, and aux_states using
  // the inferred shapes and dtypes.
  if (nullptr == shared_buffer)
  {  // regular simple bind
    LOG(INFO)<<"进入核心InitArguments   对于regular simple bind";
    InitArguments(idx, 
                  g.GetAttr<nnvm::ShapeVector>("shape"),
                  g.GetAttr<nnvm::DTypeVector>("dtype"),
                  g.GetAttr<StorageTypeVector>("storage_type"),
                  in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes,
                  grad_req_types, in_arg_vec, arg_grad_vec, aux_state_vec);
  } 
  else 
  {  // simple bind using shared data arrays and shared_exec
    InitArguments(idx, g.GetAttr<nnvm::ShapeVector>("shape"),
                  g.GetAttr<nnvm::DTypeVector>("dtype"),
                  g.GetAttr<StorageTypeVector>("storage_type"),
                  in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes,
                  grad_req_types, shared_arg_names, shared_exec,
                  shared_buffer, in_arg_vec, arg_grad_vec, aux_state_vec);
  }
  // The above code of shape and dtype inferences and argument
  // initialization is for simple_bind only. Regular bind operation
  // should do this differently.

  // Initialize the rest attributes of the graph.
  // This function can be called by regular bind
  // operation flow as well.
  LOG(INFO)<<"进入FinishInitGraph(symbol, g, shared_exec, feed_dict);";
  FinishInitGraph(symbol, g, shared_exec, feed_dict);
}

/*!
 * \brief Return a new executor with the same symbol and shared memory,
 * but different input/output shapes.
 * For runtime reshaping, variable length sequences, etc.
 * The returned executor shares state with the current one,
 * and cannot be used in parallel with it.
 */
Executor* GraphExecutor::Reshape(const bool partial_shaping,
                                 const bool allow_up_sizing,
                                 const Context& default_ctx,
                                 const std::map<std::string, Context>& ctx_map,
                                 const std::unordered_map<std::string, TShape>&
                                   provided_arg_shapes,
                                 std::vector<NDArray>* in_args,
                                 std::vector<NDArray>* arg_grads,
                                 std::vector<NDArray>* aux_states) {
  nnvm::Graph g;
  g.outputs = std::vector<nnvm::NodeEntry>(graph_.outputs.begin(),
    graph_.outputs.begin() + num_forward_outputs_);
  nnvm::Symbol symbol;
  symbol.outputs = g.outputs;
  const nnvm::IndexedGraph& idx = g.indexed_graph();
  nnvm::ShapeVector arg_shapes(idx.input_nodes().size(), TShape());
  for (size_t i = 0; i < num_forward_inputs_; ++i) {
    const uint32_t nid = idx.input_nodes().at(i);
    const std::string& name = idx[nid].source->attrs.name;
    auto it = provided_arg_shapes.find(name);
    if (provided_arg_shapes.end() != it) {
      arg_shapes[i] = it->second;
    }
  }
  g = InferShape(std::move(g), std::move(arg_shapes), "__shape__");
  if (g.GetAttr<size_t>("shape_num_unknown_nodes") != 0U) {
    HandleInferShapeError(num_forward_inputs_, g.indexed_graph(),
                          g.GetAttr<nnvm::ShapeVector>("shape"));
  }
  const nnvm::ShapeVector& shape_vec = g.GetAttr<nnvm::ShapeVector>("shape");
  std::vector<OpReqType> grad_req_types;
  size_t grad_top = 0;
  const size_t num_args = in_arg_map_.size();
  const size_t num_aux = aux_state_map_.size();
  in_args->reserve(num_args);
  grad_req_types.reserve(num_args);
  arg_grads->reserve(num_args);
  aux_states->reserve(num_aux);
  for (uint32_t nid : idx.input_nodes()) {
    std::string name = idx[nid].source->attrs.name;
    const TShape& new_shape = shape_vec[idx.entry_id(nid, 0)];
    if (idx.mutable_input_nodes().count(nid) == 0) {
      NDArray& arr = in_arg_map_.at(name);
      auto it = arg_grad_map_.find(name);
      if (partial_shaping || provided_arg_shapes.count(name) || new_shape == arr.shape()) {
        if (new_shape.Size() > arr.shape().Size()) {
          CHECK(allow_up_sizing) << "New shape of arg: " << name << " is larger than original."
            << "First making a big executor and then down sizing it "
            << "is more efficient than the reverse."
            << "If you really want to up size, set allow_up_sizing=True "
            << "to enable allocation of new arrays.";
          in_args->emplace_back(new_shape, arr.ctx(), false, arr.dtype());
          if (it != arg_grad_map_.end()) {
            NDArray& darr = it->second;
            arg_grads->emplace_back(new_shape, darr.ctx(), false, darr.dtype());
            grad_req_types.push_back(grad_store_.at(grad_top++).first);
          } else {
            arg_grads->emplace_back();
            grad_req_types.push_back(kNullOp);
          }
        } else {
          in_args->push_back(arr.Reshape(new_shape));
          if (it != arg_grad_map_.end()) {
            NDArray& darr = it->second;
            arg_grads->push_back(darr.Reshape(new_shape));
            grad_req_types.push_back(grad_store_.at(grad_top++).first);
          } else {
            arg_grads->emplace_back();
            grad_req_types.push_back(kNullOp);
          }
        }
      } else {
        LOG(FATAL) << "Shape of unspecifie arg: " << name << " changed. "
          << "This can cause the new executor to not share parameters "
          << "with the old one. Please check for error in network."
          << "If this is intended, set partial_shaping=True to suppress this warning.";
      }
    } else {
      NDArray& arr = aux_state_map_.at(name);
      if (partial_shaping || new_shape == arr.shape()) {
        if (new_shape.Size() > arr.shape().Size()) {
          CHECK(allow_up_sizing) << "New shape of arg: " << name << " is larger than original."
            << "First making a big executor and then down sizing it "
            << "is more efficient than the reverse."
            << "If you really want to up size, set allow_up_sizing=True "
            << "to enable allocation of new arrays.";
          aux_states->emplace_back(new_shape, arr.ctx(), false, arr.dtype());
        } else {
          aux_states->push_back(arr.Reshape(new_shape));
        }
      } else {
        LOG(FATAL) << "Shape of unspecifie arg: " << name << " changed. "
          << "This can cause the new executor to not share parameters "
          << "with the old one. Please check for error in network."
          << "If this is intended, set partial_shaping=True to suppress this warning.";
      }
    }
  }
  auto exec = new GraphExecutor();
  exec->Init(symbol, default_ctx, ctx_map,
             *in_args, *arg_grads, grad_req_types, *aux_states,
             this);
  return exec;
}
/*!
 * \brief This function is triggered by both simple_bind
 * and bind flows.
 * Setup backward graph, create device and context
 * attributes in the graph, and calculate the number
 * of forward nodes.
 */
Graph GraphExecutor::InitGraph(nnvm::Symbol symbol,
                               const Context& default_ctx,
                               const std::map<std::string, Context>& ctx_map,
                               const std::vector<Context>& in_arg_ctxes,
                               const std::vector<Context>& arg_grad_ctxes,
                               const std::vector<Context>& aux_state_ctxes,
                               const std::vector<OpReqType>& grad_req_types) {
  // setup gradient
  //  进入
  LOG(INFO)<<"进入initgraph（）";
  // 调用初始化全图
  // 这里面应该包括自动生成反向图
  nnvm::Graph g = InitFullGraph(symbol, grad_req_types);
  // create "device" and "context" attrs for the graph
  //  为每一个Op分配上下文
  g = AssignContext(g, default_ctx, ctx_map,
                    in_arg_ctxes,
                    arg_grad_ctxes,
                    aux_state_ctxes,
                    grad_req_types,
                    num_forward_inputs_,
                    num_forward_outputs_);
  //   获取到g 的索引图
  //   获取索引图
  const auto& idx = g.indexed_graph();
  //  get number of nodes used in forward pass
  num_forward_nodes_ = 0;
  //   1   
  LOG(INFO)<<"num_forward_outputs_"<<num_forward_outputs_;
  for (size_t i = 0; i < num_forward_outputs_; ++i) 
  {
    // 对于所有的输出节点
    //  哪一个节点的ID最大，那么总的界前向节点的数目就是这个+1  这个很容易理解的。
    num_forward_nodes_ = std::max(
                    num_forward_nodes_, static_cast<size_t>(idx.outputs()[i].node_id + 1));
  }
  //  14  结果为，包括参数和中间的节点。
  LOG(INFO)<<"num_forward_nodes_"<<num_forward_nodes_;
  return g;
}

// initialize the memory of each entries//
//  为每一个实体分配内存
void GraphExecutor::InitDataEntryMemory(std::vector<NDArray>* shared_pool)
{
  LOG(INFO)<<"进入InitDataEntryMemory";
  using nnvm::DTypeVector;
  using nnvm::ShapeVector;
  using nnvm::StorageVector;
  // get the graph
  const auto& idx = graph_.indexed_graph();
  // get the storage
  //


  const auto& vdtype = graph_.GetAttr<DTypeVector>("dtype");
  const auto& vshape = graph_.GetAttr<ShapeVector>("shape");
  const auto& vstorage = graph_.GetAttr<StorageVector>("storage_id");
  const auto& vstorage_type = graph_.GetAttr<StorageTypeVector>("storage_type");
  const auto& vctx = graph_.GetAttr<ContextVector>("context");
  CHECK_EQ(idx.num_node_entries(), vshape.size());
  CHECK_EQ(idx.num_node_entries(), vdtype.size());
  CHECK_EQ(idx.num_node_entries(), vstorage.size());
  CHECK_EQ(data_entry_.size(), vshape.size());
  
  std::vector<Context> data_context(idx.num_node_entries());
  std::vector<NDArrayStorageType> data_storage_type(idx.num_node_entries(), kUndefinedStorage);
  
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) 
  {
    for (uint32_t i = 0; i < idx[nid].source->num_outputs(); ++i)
     {

      auto eid = idx.entry_id(nid, i);
      data_context[eid] = vctx[nid]; //  这个输出节点的上下文
      CHECK_NE(vstorage_type[nid], kUndefinedStorage);// 
      data_storage_type[eid] = (NDArrayStorageType) vstorage_type[nid];
      //  如果这个节点的存储类型
    }
  }

  // information about the pool
  //  存储池子的实体
  struct PoolEntry 
  {
    Context ctx;
    size_t bytes;
    NDArrayStorageType stype;
  };
  //  池子实体向量
  std::vector<PoolEntry> pool_info;

  // assign array to head gradient
  LOG(INFO)<<"num_forward_inputs_=="<<num_forward_inputs_<<"    idx.input_nodes().size（）=="<<idx.input_nodes().size();
  for (size_t i = num_forward_inputs_; i < idx.input_nodes().size(); ++i) 
  {
    uint32_t nid = idx.input_nodes().at(i);
    
    uint32_t oid = head_grad_map_.at(idx[nid].source);

    uint32_t eid = idx.entry_id(idx.outputs()[oid]);

    NDArrayStorageType stype = (NDArrayStorageType) vstorage_type[eid];

    CHECK_NE(vshape[eid].ndim(), 0U);
    CHECK_NE(vdtype[eid], -1);
    auto data_eid = idx.entry_id(nid, 0);
    // initialize based on storage_type
    if (stype != kDefaultStorage) 
    {
      //  为节点的输入的实体分配内存
      data_entry_[data_eid] = NDArray(stype, vshape[eid], data_context[eid], true, vdtype[eid]);
    } 
    else 
    {
      data_entry_[data_eid] = NDArray(vshape[eid], data_context[eid], false, vdtype[eid]);
    }
    if (log_verbose_) {
      LOG(INFO) << "\tinit head_grad entry\t" << data_eid << "\tas "
                << common::stype_string(stype);
    }
  }
  // get maximum bytes in each pool
  for (size_t i = 0; i < vshape.size(); ++i) 
  {
    if (!data_entry_[i].is_none()) continue;
    size_t bytes = vshape[i].Size() * mshadow::mshadow_sizeof(vdtype[i]);
    int storage_id = vstorage[i];

    // skip pool allocation for kBadStorageID, kExternalStorageID and kDynamicStorageID
    if (storage_id < 0) continue;

    size_t sid = static_cast<size_t>(storage_id);

    if (sid >= pool_info.size()) 
    {
      pool_info.resize(sid + 1, PoolEntry{Context::CPU(), size_t(0), kUndefinedStorage});
    }
    PoolEntry& info = pool_info[sid];

    if (info.bytes == 0) 
    {
      info = PoolEntry{data_context[i], bytes, data_storage_type[i]};
    } 
    else
    {
      info.bytes = std::max(info.bytes, bytes);
    }
  }
  // construct the re-use pool, if needed
  std::multimap<size_t, NDArray> free_pool;
  if (shared_pool != nullptr) 
  {
    for (const NDArray& nd : *shared_pool)
    {
      size_t bytes = nd.shape().Size() * mshadow::mshadow_sizeof(nd.dtype());
      free_pool.insert(std::make_pair(bytes, nd));
    }
  }
  // remake the data pool
  data_pool_.clear();
  data_pool_.resize(pool_info.size());

  // sort the pool info the descending order before allocating memory
  std::vector<size_t> sorted_pool_index;
  for (size_t i = 0; i < pool_info.size(); i++)
  {
      sorted_pool_index.push_back(i);
  }
  auto pool_comparator = [&pool_info](int lhs, int rhs)
  {
      return pool_info[lhs].bytes > pool_info[rhs].bytes;
  };
  //  按照标准的对比器件。
  std::sort(sorted_pool_index.begin(), sorted_pool_index.end(), pool_comparator);

  for (size_t i : sorted_pool_index) 
  {
    const Context& ctx = pool_info[i].ctx;
    size_t bytes = pool_info[i].bytes;
    bool allocated = false;
    for (auto it = free_pool.lower_bound(bytes); it != free_pool.end(); ++it) 
    {
      if (it->second.ctx() == ctx && it->first >= bytes) 
      {
        data_pool_[i] = it->second;
        free_pool.erase(it);
        allocated = true;
        break;
      }
    }
    if (!allocated)
     {
      size_t nword = (bytes + 3) / 4;
      CHECK_LE(nword, std::numeric_limits<nnvm::dim_t>::max());
      // allocate float arrays
      TShape shape{static_cast<nnvm::dim_t>(nword)};
      // TODO(junwu): adding delay_alloc=true to create nd
      // is a temporary solution.
      NDArray nd(shape, ctx, true);
      data_pool_[i] = nd;
      // put the new allocated arrays to shared pool
      if (shared_pool != nullptr)  {
        shared_pool->push_back(nd);
      }
    }
  }
  CHECK_EQ(data_pool_.size(), pool_info.size());
  // assign the data entries
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    // avoid pre-allocated arrays
    if (!data_entry_[i].is_none()) continue;
    // assign allocated array by storage id
    int storage_id = vstorage[i];
    auto storage_type = (NDArrayStorageType) vstorage_type[i];
    if (storage_type == kDefaultStorage) {
      CHECK_GE(storage_id, 0) << "Do not support runtime shape op yet";
      const NDArray& src = data_pool_.at(storage_id);
      data_entry_[i] = src.AsArray(vshape[i], vdtype[i]);
    } else {
      data_entry_[i] = NDArray(storage_type, vshape[i], data_context[i],
                               true, vdtype[i]);
    }
    if (log_verbose_) {
      LOG(INFO) << "\tinit data entry\t" << i << "\tas " << common::stype_string(storage_type);
    }
  }
}


void GraphExecutor::InitCachedOps()
 {
  // get the graph
  LOG(INFO)<<"进入InitCachedOps()内部%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
  const auto& idx = graph_.indexed_graph();
  const auto& vstorage_inplace =
      graph_.GetAttr<std::vector<int> >("storage_inplace_index");
  const auto& op_execs = graph_.GetAttr<OpExecVector>("op_execs");
  const auto& vctx = graph_.GetAttr<ContextVector>("context");
  const auto& addto_entry = graph_.GetAttr<std::vector<int> >("addto_entry");
  const auto& skip_plus_node = graph_.GetAttr<std::vector<int> >("skip_plus_node");
  // 20
  LOG(INFO)<<"idx.num_nodes()===="<<idx.num_nodes();
  op_nodes_.resize(idx.num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid)
   {
    LOG(INFO)<<"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    LOG(INFO)<<"遍历nid====="<<nid;
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    // 设置节点的名字
    op_nodes_[nid].opr_name = inode.source->op()->name.c_str();
    LOG(INFO)<<"node 对应OP的名字为："<<inode.source->op()->name.c_str();
    int sk=skip_plus_node.at(nid)?1:0;
    LOG(INFO)<<"skip_plus_node.at(nid)   "<<sk;
    // 这个节点是不是跳过执行
    if (skip_plus_node.at(nid))
    {
       op_nodes_[nid].skip_exec_node = true; continue;
    }
    //  分配执行器和上下文。
    op_nodes_[nid].exec = op_execs[nid];
    op_nodes_[nid].ctx = vctx[nid];
    // 得到这个节点的执行器

       /*  class OpExecutor {
        public:
          /*! \brief input data arrays, which may be either input or aux */
        // std::vector<NDArray> in_array;
          /*! \brief output data arrays */
          //std::vector<NDArray> out_array;
          /*! \brief output requirement on each array */
          //std::vector<OpReqType> req;
          /*! \brief runtime op context, contains allocated resources */
        // OpContext op_ctx;
          /*! \brief virtual destructor */
        // virtual ~OpExecutor() {} */



    auto& exec = op_nodes_[nid].exec;
    CHECK_EQ(exec->in_array.size(), 0U);
    CHECK_EQ(exec->out_array.size(), 0U);
    //  遍历每一个节点的输入
    LOG(INFO)<<"对于每一个输入";
    int  p=0;
    /* for (const auto& e : inode.inputs)
    {
      //由对应的实体得到对应的ID
      //LOG(INFO)<<"idx.entry_id(e)  "<<idx.entry_id(e);
      //  加入到执行器对应的输入的数组里面
      p++;
      //exec->in_array.push_back(data_entry_[idx.entry_id(e)]);
    }
    LOG(INFO)<<"输入实体的个数为  "<<p; */
    for (const auto& e : inode.inputs)
    {
      //  由对应的实体得到对应的ID
      //  LOG(INFO)<<"idx.entry_id(e)  "<<idx.entry_id(e);
      //  加入到执行器对应的输入的数组里面
      p++;
      exec->in_array.push_back(data_entry_[idx.entry_id(e)]);
    }

    //   detect inplace requirement
    //
    LOG(INFO)<<"输出实体的个数为  inode.source->num_outputs()"<<inode.source->num_outputs();
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index)
    {  
      uint32_t eid = idx.entry_id(nid, index);
      //   LOG(INFO)<<" 遍历eid=====   "<<eid;
      exec->out_array.push_back(data_entry_[eid]);
      //   如果这个实体是aad_to
      // 
      //int  add_to=addto_entry.at(eid)！=0? 1:0;
      //LOG(INFO)<<"节点的ID为  "<< "";
      if (addto_entry.at(eid) != 0) 
      {
        exec->req.push_back(kAddTo);
        LOG(INFO)<<" kAddTo   ";
      } 
      else if 
      (vstorage_inplace[eid] >= 0) 
      {
        exec->req.push_back(kWriteInplace);
        LOG(INFO)<<"  kWriteInplace   ";
      } 
      else if 
      (vstorage_inplace[eid] == -2)
      {
        // -2 indicate that the entry is never referenced.
        exec->req.push_back(kNullOp);
        LOG(INFO)<<"kNullOp   ";
      } 
      else
      {
        exec->req.push_back(kWriteTo);
        LOG(INFO)<<"kWriteTo   ";
      }
    }
   }
  // Note that this modifies the requirment of kWriteInplace
  // 1  8 
  LOG(INFO)<<"num_forward_outputs_===="<<num_forward_outputs_;
  LOG(INFO)<<"idx.outputs().size()===="<<idx.outputs().size();
  // 找到对应的梯度的存储
  //  1    8
  //  对于每一个输出的计算的梯度
  // 直接来遍历梯度
  for (size_t j = num_forward_outputs_; j < idx.outputs().size(); ++j) 
  {
    // 
    LOG(INFO)<<"j "<<j;
    // 得到6个梯度实体
    auto& e = idx.outputs()[j];
    //   这个输出梯度的节点。对应的梯度的存储。
    //   std::vector<std::pair<OpReqType, NDArray> > grad_store_;
    //   输出梯度的对应的节点的请求类型和梯度存储里面的情趣类型一致。
    //   这里面只是照搬过来而已。
    op_nodes_[e.node_id].exec->req[e.index] =
        grad_store_[j - num_forward_outputs_].first;

  }
  //  20
  LOG(INFO)<<"idx.num_nodes()"<<idx.num_nodes();
  
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) 
  {
    const auto& inode = idx[nid];
    if (inode.source->is_variable())    continue; 
    if (op_nodes_[nid].skip_exec_node)  continue;
    auto& exec = op_nodes_[nid].exec;      //  对于一个固定节点的执行器。
    bool  is_async = op_nodes_[nid].exec->exec_type() == ExecType::kAsync;
    bool  is_gpu = op_nodes_[nid].ctx.dev_mask() == gpu::kDevMask;

    // the variables
    // 读取的变量和写变量
    std::vector<Engine::VarHandle> use_vars, mutate_vars;
    // 输入变量。
    for (size_t i = 0; i < exec->in_array.size(); ++i) 
    {
      auto& nd = exec->in_array[i];
      use_vars.push_back(nd.var());
    }  
    for (auto& r : exec->op_ctx.requested) 
    {
      mutate_vars.push_back(r.var);
    }
    for (auto& nd : exec->out_array) 
    {
      mutate_vars.push_back(nd.var());
    }
    if (exec->var() != nullptr) 
    {
      mutate_vars.push_back(exec->var());
    }
    // dedup vars
    // 变量去重，去除写变量之后的就是读变量了
    Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);

    // all vars include both mutate vars and use vars

    std::vector<Engine::VarHandle> all_vars(use_vars);

    std::copy(mutate_vars.begin(), mutate_vars.end(),
              std::inserter(all_vars, all_vars.end()));
    //   setup exec vars
    //   交给执行引擎执行的第一个任务
    //   就是变量的设置问题。
    //   
    Engine::Get()->PushAsync(
      [exec](RunContext rctx, Engine::CallbackOnComplete on_complete) 
      {
        exec->Setup();
        on_complete();
      }, 
      Context::CPU(),
      {},
      all_vars,
      FnProperty::kNormal,
      0,
      "SetupExec");
    auto exec_fun = [exec, is_async, is_gpu] (
      RunContext ctx, Engine::CallbackOnComplete on_complete) 
      {
      if (is_async) 
      {
        exec->op_ctx.async_on_complete = on_complete;
      }
      exec->Run(ctx, is_gpu);
      // call on complete only if it is async op
      LOG(INFO)<<"is_async==="<<is_async;
      if (!is_async)
      {
        if (is_gpu)
         {
           #if MXNET_USE_CUDA
          // Wait GPU kernel to finish.
          ctx.get_stream<gpu>()->Wait();

          #else
          LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
          #endif
        }
        on_complete();

      }
    };
     //  setup the vars
     //  每一个节点cache 的OPT
     //  为节点构造完成的实例化的OP
     //  为什么叫做cache-op呢？
     //  明白了，按照执行器构造一个operator
     //
          /* struct OpNode {
        // The name of the operator
        const char* opr_name;
        // the context of the node
        Context ctx;
        // The executor
        std::shared_ptr<OpExecutor> exec;
        // skip the execution of this node
        bool skip_exec_node{false};
        // cached operator handle
        Engine::OprHandle cached_opr{nullptr};
        // cached const vars, used for seg ops creation
        std::vector<Engine::VarHandle> use_vars;
        // cached mutate vars, used for seg ops creation
        std::vector<Engine::VarHandle> mutate_vars;
      }; 
      */
    op_nodes_[nid].cached_opr = Engine::Get()->NewOperator(
        exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
        op_nodes_[nid].opr_name);

    op_nodes_[nid].mutate_vars = mutate_vars;

    op_nodes_[nid].use_vars = use_vars;
  }
}

void GraphExecutor::InitOpSegs()
 {
  LOG(INFO)<<"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC";
  LOG(INFO)<<"进入到InitOpSegs()内部";
  //  20个
  size_t total_num_nodes = graph_.indexed_graph().num_nodes();

  //std::vector<CachedSegOpr> mxnet::exec::GraphExecutor::cached_seg_opr_
  cached_seg_opr_.clear();
    //  也就是说一段的cache的操作。
    // a cached segment operator that executes a segment
    //   struct CachedSegOpr {
    //  context of the operator
    //  Context ctx;
    //  begin in topo order
    //  size_t topo_start;
    //  end in topo order
    //  size_t topo_end;
    //  the cached operator
    //  Engine::OprHandle opr = nullptr;
    //  list of op executors
    //  std::vector<std::shared_ptr<OpExecutor> > exec_list;
    // };
  CachedSegOpr p;
  //  初始化为全部的节点的数目的
  //  默认和节点的数目保持一致。
  //  20
  LOG(INFO)<<"总的节点数目为="<<total_num_nodes;
  // 
  cached_seg_opr_.resize(total_num_nodes, p);
  if (monitor_callback_) return;

  // Generate segments based on the graph structure
  //  是不是批量执行推测？？？？？ 如何分段？
  bool prefer_bulk_exec_inference = dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_INFERENCE", true);
  // Whether to perform bulk exec for training
  const profiler::Profiler *prof = profiler::Profiler::Get();
  // 是不是批量执行训练？？？
  bool prefer_bulk_exec = dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_TRAIN", 1)
                          && (!prof || !prof->AggregateEnabled());
  //  这个用来判断训练还是推测挺好的
  bool is_training = num_forward_nodes_ != total_num_nodes;
  // 批次训练
  if (prefer_bulk_exec  && is_training) 
  {
    LOG(INFO)<<"进入到BulkTrainingOpSegs(total_num_nodes)";
    this->BulkTrainingOpSegs(total_num_nodes);
  }
  //  批次推测
  if (prefer_bulk_exec_inference && !is_training) 
  {
    LOG(INFO)<<"进入到BulkInferenceOpSegs";
    this->BulkInferenceOpSegs();
  }
}

//    这里面总算可以理解了，分段的目的是为了将变量和对应的计算合在一起进行计算。
//    
void GraphExecutor::BulkTrainingOpSegs(size_t total_num_nodes)
 {
   LOG(INFO)<<"DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD";
   LOG(INFO)<<"DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD";
   LOG(INFO)<<"DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD";
   LOG(INFO)<<"进入BulkTrainingOpSegs";
  // The maximum number of node in a segment executed in bulk
  size_t num_nodes_threshold = dmlc::GetEnv("MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN", 15);
  //  15
  LOG(INFO)<<"num_nodes_threshold    "<<num_nodes_threshold;
  // create forward segments for training
  size_t topo_start = 0;
  //  14
  LOG(INFO)<<"num_forward_nodes_"<<num_forward_nodes_;
  //  进行正向的过程。
  //  我们可以看出
  //  0-0  1-1   2-2   3-5   6-6  7-9
  //  变量单独一个段
  //  以及连续的非变量加上第一个变量构成段

  for (size_t nid = 0; nid < num_forward_nodes_; nid++)
  {
    // 按照拓扑排序好的性顺序来获取节点
    auto &node = graph_.indexed_graph()[nid].source;
    auto &op_node = op_nodes_[nid];
    // check if the segment relies on external input, or exceeds maxinum number of node,
    // or requires async ops
    //  采用分段的方式。
    //  如果是一个变量或者  长度太长了。。或者是异步执行。
    int  is_a=node->is_variable()?1:0;
    LOG(INFO)<<"nid====="<<nid<<"   是不是变量== "<<is_a;
    //  只要3折有一个满足即可。

    if (node->is_variable() || nid - topo_start > num_nodes_threshold ||
    op_node.exec->exec_type() != ExecType::kSync) 
    {
      // create a new segment for the previous nodes if the current one cannot be bulked
      // num_nodes_threshold  每隔这么多次进行一次成段构造。
      LOG(INFO)<<"topo_start  "<<topo_start<<"   nid   "<<nid;
      cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, nid);
      topo_start = nid + 1;
    }
  }
  //  the last segment
  //  负责构造最后一个段
  if (topo_start != num_forward_nodes_) 
  {
    LOG(INFO)<<"topo_start  "<<topo_start<<"   num_forward_nodes_   "<<num_forward_nodes_;
    cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, num_forward_nodes_);
  }

  // create backward segments for training
  // get all gradient variables
  
  std::unordered_set<engine::VarHandle> grad_vars;

  //std::vector<std::pair<OpReqType,NDArray>> mxnet::exec::GraphExecutor::grad_store_
  //  second 是梯度的存储的数组
  for (auto &kv : grad_store_) 
  {
    grad_vars.insert(kv.second.var());
  }
  //   等于  7   
  LOG(INFO)<<"grad_vars.size()  "<<grad_vars.size();
  //
  auto &idx = graph_.indexed_graph();
  //  14
  topo_start = num_forward_nodes_;
  //    20
  LOG(INFO)<<"total_num_nodes"<<total_num_nodes;
  //  进行反向的过程。
  //  从第一个反向的节点开始
  for (size_t nid = num_forward_nodes_; nid < total_num_nodes; nid++) 
  {
    auto &op_node = op_nodes_[nid];
    if (op_node.skip_exec_node || op_node.exec == nullptr) 
    {
      continue;
    }
    int  is_B=idx[nid].source->is_variable()?1:0;
    LOG(INFO)<<"nid="<<nid<<"是不是变量？？？？ "<<is_B;
    if (idx[nid].source->is_variable() || nid - topo_start > num_nodes_threshold ||
        op_node.exec->exec_type() != ExecType::kSync)
    {
      //std::vector<CachedSegOpr> mxnet::exec::GraphExecutor::cached_seg_opr_
      LOG(INFO)<<"topo_start  "<<topo_start<<"   nid   "<<nid;
      cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, nid);
      topo_start = nid + 1;
    } 
    else 
    {
      // 如果一个node 产生输出的梯度。那么立即产生一个新的段
      // If it produces output gradient, don't include it in the segment
      bool output_gradient = false;
      //  遍历这个节点的每一个输出的结果。
      for (auto &out_arr : op_node.exec->out_array)
       {
        //  如果输出的梯度变量发现了这个节点
        //  也就是说这个节点如果是产生进行梯度计算的节点，那么也应该单独列出。
        if (grad_vars.find(out_arr.var()) != grad_vars.end()) 
        {
          output_gradient = true;
        }
      }
      if (output_gradient) 
      {
        LOG(INFO)<<"output_gradient对吗  "<<output_gradient;
        LOG(INFO)<<"topo_start  "<<topo_start<<"   nid   "<<nid;
        cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, nid);
        topo_start = nid + 1;
      }
    }
  }
  // last segment for backward
  if (topo_start < total_num_nodes)
 {
    LOG(INFO)<<"topo_start  "<<topo_start<<"   total_num_nodes   "<<total_num_nodes;
    cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, total_num_nodes);
  }
}

void GraphExecutor::BulkInferenceOpSegs() {
  // Attempt to bulk the whole graph for inference.  We will only create new segments when
  // required for non-kSync operations.
  size_t topo_start = 0;
  for (size_t nid = 0; nid < num_forward_nodes_; nid++) {
    auto &node = graph_.indexed_graph()[nid].source;
    auto &op_node = op_nodes_[nid];

    // Variables do not need to be segmented at inference time.
    if (node->is_variable()) continue;

    if (op_node.exec->exec_type() != ExecType::kSync) {
      cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, nid);
      topo_start = nid + 1;
    }
  }
  // The last segment
  if (topo_start != num_forward_nodes_) {
    cached_seg_opr_[topo_start] = this->CreateCachedSegOpr(topo_start, num_forward_nodes_);
  }
}

void GraphExecutor::ExecuteMonCallback(size_t nid) {
  static const auto& flist_outputs =
      nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  const auto& idx = graph_.indexed_graph();
  std::vector<std::string> output_names;
  OpNode& opnode = op_nodes_[nid];
  const auto& inode = idx[nid];
  const auto& node = idx[nid].source;
  if (flist_outputs.count(node->op())) {
    output_names = flist_outputs[node->op()](node->attrs);
  } else {
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      output_names.emplace_back(std::to_string(i));
    }
  }
  CHECK_EQ(opnode.exec->out_array.size(), output_names.size());
  for (index_t i = 0; i < opnode.exec->out_array.size(); ++i) {
    NDArray *cpy = new NDArray(opnode.exec->out_array[i]);
    std::string name = inode.source->attrs.name + "_" + output_names[i];
    this->monitor_callback_(name.c_str(), reinterpret_cast<void*>(cpy));
  }
}

void GraphExecutor::RunOps(bool is_train, size_t topo_start, size_t topo_end) 
{
  LOG(INFO)<<"进入RunOP  "<<"is_train   "<<is_train<<"topo_start  "<<topo_start<<"topo_end"<<topo_end;
  // Update context

  //  从拓扑的开始的结束。
  const auto& idx = graph_.indexed_graph();
  //  设置执行的环境。
  for (size_t nid = topo_start; nid < topo_end; ++nid) 
  {
    OpNode& opnode = op_nodes_[nid];
    // 如果跳过执行？
    if (opnode.skip_exec_node) continue;

    const auto& inode = idx[nid];

    if (inode.source->is_variable()) continue;
    // 处于是否训练过程。
    opnode.exec->op_ctx.is_train = is_train;

  }

  // Push Ops
  for (size_t nid = topo_start; nid < topo_end; ++nid) 
  {
    //  这个是干嘛的？？？？？？？
    //  这段世他的变量和操作在一起进行。
    //
    auto seg_op = cached_seg_opr_[nid];
    //  Check segments firs
    //  我们看下哪一些条件存在哈。
    //  说明是按照段进行执行的。好
    if (monitor_callback_ == nullptr && seg_op.opr != nullptr && seg_op.topo_end <= topo_end) 
    {
      //LOG(INFO)<<"进入段模式";
      bool profiling = profiler::Profiler::Get()->GetState() == profiler::Profiler::kRunning;
      Engine::Get()->Push(seg_op.opr, seg_op.ctx, 0, profiling);
      //  nid 直接可以加上。
      nid = seg_op.topo_end - 1;
      continue;
    }
    // Normal mode
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    OpNode& opnode = op_nodes_[nid];
    if (op_nodes_[nid].skip_exec_node) continue;
    opnode.exec->op_ctx.is_train = is_train;
    //  如果OP是跨设备的copy
    //  也就是说是复制节点
    if (opnode.exec->exec_type() == ExecType::kCrossDeviceCopy)
    {
      CHECK_EQ(inode.inputs.size(), 1U);
      CHECK_EQ(opnode.exec->in_array.size(), 1U);
      CHECK_EQ(opnode.exec->out_array.size(), 1U);
      //  直接进行数据复制
      CopyFromTo(opnode.exec->in_array[0], &(opnode.exec->out_array[0]));
    }
    else if (opnode.cached_opr != nullptr) 
    {
      bool profiling = profiler::Profiler::Get()->GetState() == profiler::Profiler::kRunning;
      LOG(INFO)<<"Engine::Get()->Push(opnode.cached_opr, opnode.ctx, 0, profiling);";
      // 讲对应的OP送到队列去执行。
      //
      Engine::Get()->Push(opnode.cached_opr, opnode.ctx, 0, profiling);
    } 
    else 
    {
      LOG(FATAL) << "Not accessed";
    }
    // Monitor callbacks
    if (monitor_callback_) 
    {
      ExecuteMonCallback(nid);
    }
  }
}

GraphExecutor::CachedSegOpr GraphExecutor::CreateCachedSegOpr(size_t topo_start, size_t topo_end)
 {
  LOG(INFO)<<"CreateCachedSegOpreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
  std::vector<Engine::VarHandle>  use_vars;
  std::vector<Engine::VarHandle>  mutate_vars;
  Context *pctx = nullptr;
  GraphExecutor::CachedSegOpr    ret;
  ret.topo_start = topo_start;
  ret.topo_end = topo_end;
  auto& exec_list = ret.exec_list;
  // invalid segment
  if (topo_end <= topo_start) 
  {
    return ret;
  }
  std::string opr_names = "[";

  const auto& idx = graph_.indexed_graph();
   // 遍历每一个节点
  for (size_t nid = topo_start; nid < topo_end; ++nid)
   {
    std::vector<Engine::VarHandle> all_vars;
    const auto& inode = idx[nid];
    OpNode& op_node = op_nodes_[nid];
    if (op_node.skip_exec_node)       continue;
    if (inode.source->is_variable())  continue;
    if (op_node.exec->exec_type() != ExecType::kSync)
    {
      return ret;
    }
    if (pctx == nullptr)  pctx = &(op_node.ctx);
    if (*pctx != op_node.ctx) 
    {
      return ret;
    }

    auto& exec = op_nodes_[nid].exec;
    //将的那个节点的依赖的变量复制到全局的变量里面
    std::copy(op_node.mutate_vars.begin(), op_node.mutate_vars.end(),
              std::inserter(mutate_vars, mutate_vars.end()));

    std::copy(op_node.use_vars.begin(), op_node.use_vars.end(),
              std::inserter(use_vars, use_vars.end()));
    //  添加这个节点的执行器
    ret.exec_list.push_back(exec);
    
    opr_names += inode.source->op()->name + ",";
    LOG(INFO)<<"节点的名字为"<<inode.source->op()->name;

  }

  if (pctx == nullptr) return ret;
  ret.ctx = *pctx;
  // 变量去重
  Engine::Get()->DeduplicateVarHandle(&use_vars, &mutate_vars);

  bool is_gpu = pctx->dev_mask() == gpu::kDevMask;
  // 获取到每一个和上下文相关的执行的函数，也就是说得到全局的函数
  auto exec_fun = [exec_list, is_gpu] (
      RunContext ctx, Engine::CallbackOnComplete on_complete) {
    // Run all opr in the sub-graph
    for (auto &exec : exec_list) {
      exec->Run(ctx, is_gpu);
    }
    if (is_gpu) {
    #if MXNET_USE_CUDA
          // Wait GPU kernel to finish.
          ctx.get_stream<gpu>()->Wait();
    #else
      LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
    #endif
    }
    on_complete();
  };
  opr_names.pop_back();
  opr_names += "]";
  auto iter = cached_seg_opr_names_.insert(opr_names).first;
  //  返回的值GraphExecutor::CachedSegOpr    ret;
  //  .opr=全部的OPr
  ret.opr = Engine::Get()->NewOperator(
    exec_fun, use_vars, mutate_vars, FnProperty::kNormal,
    iter->c_str());
  return ret;
}
}  // namespace exec



Executor *Executor::SimpleBind(nnvm::Symbol symbol,
                               const Context& default_ctx,
                               const std::map<std::string, Context>& group2ctx,
                               const std::vector<Context>& in_arg_ctxes,
                               const std::vector<Context>& arg_grad_ctxes,
                               const std::vector<Context>& aux_state_ctxes,
                               const std::unordered_map<std::string, TShape>& arg_shape_map,
                               const std::unordered_map<std::string, int>& arg_dtype_map,
                               const std::unordered_map<std::string, int>& arg_stype_map,
                               const std::vector<OpReqType>& grad_req_types,
                               const std::unordered_set<std::string>& shared_arg_names,
                               std::vector<NDArray>* in_args,
                               std::vector<NDArray>* arg_grads,
                               std::vector<NDArray>* aux_states,
                               std::unordered_map<std::string, NDArray>* shared_buffer,
                               Executor* shared_exec) 
  {
  //  这里面自然会调用GraphExecutor()的构造函数。
  //  构造函数啥都没有做，看看，什么参数都不用传递
  auto exec = new exec::GraphExecutor();
  //  然后会调用Init函数。
  //  实现形状的推断
  //  反向计算的规划
  //  中间结果的内存的规划
  //  设备上下文的分配
  //  实体内存的分配
  // 这里面的symbol 其实就是最后的一个节点的。

  exec->Init(symbol, default_ctx, group2ctx,                 // 
             in_arg_ctxes, arg_grad_ctxes, aux_state_ctxes,  // 上下文
             arg_shape_map, arg_dtype_map, arg_stype_map,
             grad_req_types, shared_arg_names,
             in_args, arg_grads, aux_states,
             shared_buffer, shared_exec);
  return exec;
}

Executor *Executor::Bind(nnvm::Symbol symbol,
                         const Context& default_ctx,
                         const std::map<std::string, Context>& group2ctx,
                         const std::vector<NDArray> &in_args,
                         const std::vector<NDArray> &arg_grad_store,
                         const std::vector<OpReqType> &grad_req_type,
                         const std::vector<NDArray> &aux_states,
                         Executor* shared_exec) {
  auto exec = new exec::GraphExecutor();
  exec->Init(symbol, default_ctx, group2ctx,
             in_args, arg_grad_store, grad_req_type, aux_states,
             reinterpret_cast<Executor*>(shared_exec));
  return exec;
}
}  // namespace mxnet
