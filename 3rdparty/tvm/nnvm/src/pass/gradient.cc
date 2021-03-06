/*!
 *  Copyright (c) 2016 by Contributors
 * \file gradients.cc
 * \brief Passes that takes gradient of the graph
 * This code code was modified based on mxnet codebase by Min Lin
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <algorithm>
#include <functional>

namespace nnvm {
namespace pass {
namespace {

// default aggregate gradient function
// require operator zeros and elemwise_sum to be presented.
NodeEntry DefaultAggregateGradient(std::vector<NodeEntry>&& v) 
{
  if (v.size() == 1) {
    return std::move(v[0]);
  } else if (v.size() == 0) {
    NodePtr zero_node = Node::Create();
    zero_node->attrs.op = Op::Get("zeros");
    zero_node->attrs.name = "zero_grad";
    zero_node->attrs.op->attr_parser(&(zero_node->attrs));
    return NodeEntry{zero_node, 0, 0};
  } else {
    NodePtr sum_node = Node::Create();
    sum_node->attrs.op = Op::Get("elemwise_sum");
    sum_node->inputs = std::move(v);
    sum_node->attrs.name = "grad_sum";
    sum_node->attrs.dict["num_args"] = std::to_string(sum_node->inputs.size());
    sum_node->attrs.op->attr_parser(&(sum_node->attrs));
    return NodeEntry{sum_node, 0, 0};
  }
}

bool CheckGradAllZero(const std::vector<NodeEntry>& grads,
                      const std::vector<const Op*>& zero_ops) {
  if (!grads.size() || !zero_ops.size()) return false;
  for (const auto& g : grads) {
    bool found = false;
    for (const auto& op : zero_ops) {
      if (g.node->op() == op) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

// helper entry
struct GradEntry {
  #ifdef _MSC_VER
  NodeEntry sum = NodeEntry{nullptr, 0, 0};
  #else
    NodeEntry sum{nullptr, 0, 0};
  #endif
  std::vector<NodeEntry> grads;
  bool need_attr_hint{true};
};

Graph Gradient(Graph src) 
{
  using nnvm::FGradient;
  using MirrorFun = std::function<int (const Node& node)>;
  using AttrHintFun = std::function<NodeEntry (const NodeEntry& src, const NodeEntry &like)>;

  CHECK_NE(src.attrs.count("grad_ys"), 0U)
      << "Gradient require grad_ys to be presented.";
  CHECK_NE(src.attrs.count("grad_ys_out_grad"), 0U)
      << "Gradient require grad_ys_out_grad to be presented.";
  CHECK_NE(src.attrs.count("grad_xs"), 0U)
      << "Gradient require grad_xs to be presented.";
  
  //
  const std::vector<NodeEntry>& ys =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys");
  const std::vector<NodeEntry>& ys_out_grad =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys_out_grad");
  const std::vector<NodeEntry>& xs =
      src.GetAttr<std::vector<NodeEntry> >("grad_xs");
  using AggFun = std::function<NodeEntry (std::vector<NodeEntry>&& inputs)>;
  AggFun agg_fun = DefaultAggregateGradient;
  //  使用自定义的梯度的聚集函数
  if (src.attrs.count("grad_aggregate_fun") != 0) 
  {
       agg_fun = src.GetAttr<AggFun>("grad_aggregate_fun");
  }
  MirrorFun mirror_fun = nullptr;
  //  ??????????????????
  if (src.attrs.count("grad_mirror_fun") != 0)
  {
    mirror_fun = src.GetAttr<MirrorFun>("grad_mirror_fun");
  }
  //  ?????????????????
  AttrHintFun attr_hint_fun = nullptr;
  if (src.attrs.count("attr_hint_fun") != 0) {
    attr_hint_fun = src.GetAttr<AttrHintFun>("attr_hint_fun");
  }
  //  ??????????????????
  std::vector<const Op*> zero_ops;
  if (src.attrs.count("zero_ops") != 0) 
  {
    zero_ops = src.GetAttr<std::vector<const Op*> >("zero_ops");
  }

  const Op* copy_op = (src.attrs.count("copy_op") != 0) ?
      Op::Get(src.GetAttr<std::string>("copy_op")) :
      nullptr;

  // topo sort
  std::vector<NodePtr> topo_order;
  //  表示一个前项节点对应的梯度的实体有哪些。
  std::unordered_map<Node*, std::vector<GradEntry> > output_grads;
  //  节点和对应的梯度
  //  这里面又是采用深度优先遍历的方式。
  //  从ys  也就是损失结果反向深度遍历
  DFSVisit(ys, [&](const NodePtr& node) 
  {
      //  匿名的遍历函数，如果这个节点属于梯度的输出的节点,还没有便利到。
      if (output_grads.count(node.get()) == 0) 
      {
        // 重新修改这个节点的结果梯度的大小
         LOG(INFO)<<"在DFSVIST里面"<<node->attrs.name;
         output_grads[node.get()].resize(node->num_outputs());
      }
      // 将结果加入到排序里面
      topo_order.push_back(node);
    });
//  最终得到逆向拓扑排序的结果。topo_order里面。
//  并且得到的，我们在这里面输出一下试一下哈。
for(int i=0;i<topo_order.size();i++)
{
   // using NodePtr = std::shared_ptr<Node>;
   //  得到这个节点。
   auto nodeptr=topo_order[i];
   LOG(INFO)<<"q求梯度时候反向遍历结果"<<nodeptr->attrs.name;
}
      //  我们可以看出按照先后的14个进行了排序
      //  排序的结果是节点。
      /* 
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果X
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果w0
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果b0
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果w1
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果b1
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果w2
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果b2
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果label
      [21:27:09] src/pass/gradient.cc:147: q求梯度时候反向遍历结果 
      */
      //   14  节点的个数

  LOG(INFO)<<"topo_order.size()"<<topo_order.size();
  CHECK_EQ(ys.size(), ys_out_grad.size());
  //   1
  LOG(INFO)<<"在pass  梯里面ys.size()"<<ys.size();
  //   1
  LOG(INFO)<<"在pass  梯里面ys_out_grad.size()"<<ys_out_grad.size();
  //   1
  //   1
  //  首先将输出节点进行对应
  for (size_t i = 0; i < ys.size(); ++i) 
  {
    
    NodeEntry ograd = ys_out_grad[i];
    //  输入节点对应的梯度的节点。
    output_grads[ys[i].node.get()][ys[i].index].grads = { ograd };
  }

  //  Check that all xs are reachable from ys
  //  7
  LOG(INFO)<<"在pass  梯里面xs.size()"<<xs.size();
  // 7
  // 如果需要反向的实体的节点不在反向便利结果当中，当然出错了。
  for (size_t i = 0; i < xs.size(); ++i)
  {
    //  发现有些梯度在输入需要计算梯度的前向节点可以找到。
    CHECK(output_grads.find(xs[i].node.get()) != output_grads.end())
        << "Cannot differentiate with respect to the " << i+1 << "-th variable "
        << "because it is unreachable from the outputs.";
  }

  // construct mirror reduece memory strategy if needed
  std::unordered_map<Node*, NodePtr> mirror_map;
  if (mirror_fun != nullptr) {
    for (const NodePtr& n : topo_order) 
    {
      if (mirror_fun(*n)) {
        NodePtr new_node = Node::Create();
        *new_node = *n;
        new_node->attrs.name += "_mirror";
        for (auto& e : new_node->inputs) {
          e.node = mirror_map.at(e.node.get());
        }
        for (auto& n : new_node->control_deps) {
          n = mirror_map.at(n.get());
        }
        mirror_map[n.get()] = std::move(new_node);
      } else {
        mirror_map[n.get()] = n;
      }
    }
  }

  // traverse backward
  static auto& grad_fun_map = Op::GetAttr<FGradient>("FGradient");
  static auto& finfer_shape = Op::GetAttr<FInferShape>("FInferShape");
  std::vector<NodeEntry>    out_agg_grads;
  //   遍历拓扑排序
  //   反向迭代器（rbegin,rend）
  //   遍历每一个节点。需要反向的节点。从后向里面  
  for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) 
  {
    //LOG(INFO)<<"在pass  梯里面++rit"<<rit;
    // 获取到这个节点
    const NodePtr& ptr = *rit;
    //
    if (ptr->is_variable()) continue;
    out_agg_grads.clear();
    // 获取到这个反向节点的梯度实体向量
    auto& out_grad_vec = output_grads.at(ptr.get());
    //LOG(INFO)<<"auto& out_grad_vec = output_grads.at(ptr.get());  out_grad_vec  在pass  out_grad_vec.size()  "<<out_grad_vec<<"   "<<out_grad_vec.size();
    //  对于这个节点的每一个梯度值
    //  1 1 1 1 1 1 1   6 个  1
    LOG(INFO)<<" out_grad_vec.size()"<<out_grad_vec.size();
    for(uint32_t i = 0; i < out_grad_vec.size(); ++i) 
    {
      GradEntry& e = out_grad_vec[i];
      e.sum = agg_fun(std::move(e.grads));
      if (e.need_attr_hint && attr_hint_fun != nullptr)
      {
        e.sum = attr_hint_fun(e.sum, NodeEntry{ptr, 0, i});
      }
      out_agg_grads.push_back(e.sum);
    }
    if ((*rit)->inputs.size() != 0) 
    {
      int  b=mirror_map.size() == 0 ? 1 : 0;
      LOG(INFO)<<"b=mirror_map.size() == 0 ? 1 : 0     "<<b;
      NodePtr fwd_node = (mirror_map.size() == 0 ? ptr : mirror_map.at(ptr.get()));
      //  得到这个节点的所有的输入的梯度的实体。
      //  不一定是节点。
      std::vector<NodeEntry> input_grads;
      //
      if (grad_fun_map.count(ptr->op())) 
      {
        //  通过节点找到需要计算的梯度的值。
        //  根据输出的梯度以及计算的OP 得到输入的梯度的实体
        input_grads = grad_fun_map[ptr->op()](fwd_node, out_agg_grads);
        //LOG(INFO)<<"对应的input_grads大小为"<<input_grads.size();
        //LOG(INFO)<<"(*rit)->inputs.size()"<<(*rit)->inputs.size();
        //这个节点本身的输入和需要计算的输入的梯度数目是不是相同
        CHECK_EQ((*rit)->inputs.size(), input_grads.size())
            << "Gradient function not returning enough gradient";   
      } 
      else if (CheckGradAllZero(out_agg_grads, zero_ops)) 
      {
        for (size_t i = 0; i < fwd_node->num_inputs(); ++i) {
          std::ostringstream os;
          if (1 == fwd_node->num_inputs()) {
            os << fwd_node->attrs.name << "_backward";
          } else {
            os << fwd_node->attrs.name << "_in" << i << "_backward";
          }
          auto p = Node::Create();
          p->attrs.op = zero_ops[0];
          p->attrs.name = os.str();
          p->inputs.push_back(fwd_node->inputs[i]);
          p->control_deps.emplace_back(fwd_node);
          if (p->op()->attr_parser != nullptr) {
            p->op()->attr_parser(&(p->attrs));
          }
          input_grads.emplace_back(nnvm::NodeEntry{p, 0, 0});
        }
      } 
      else 
      {
        // 说明这个节点是不可差分的
        LOG(FATAL) << "Operator " << fwd_node->op()->name << " is non-differentiable "
                   << "because it didn't register FGradient attribute.";
      }
      //  得到了这个节点的所有的输入的梯度的实体
      auto git = input_grads.begin(); //  第一个梯度的实体
      //   遍历这个节点的所有的输入实体
      for (auto it = (*rit)->inputs.begin(); it != (*rit)->inputs.end(); ++it, ++git) 
      {
        //  一个实体属于哪一个节点，以及在节点里面的排序。
        //  std::unordered_map<nnvm::Node *, std::vector<nnvm::pass::<unnamed>::GradEntry>> output_grads
        auto& ge = output_grads[it->node.get()][it->index];

        // if any of the backward op can do shape inference, the hint is not necessary.
        if (finfer_shape.count(git->node->op())) 
        {
          ge.need_attr_hint = false;
        }
        ge.grads.emplace_back(std::move(*git));
      }
    }
  }
  // take out the xs' grads

  Graph ret;
  ret.outputs.resize(xs.size());
  NodeEntryMap<std::pair<size_t, size_t> >  unique_grads;
  size_t counter = 0;
  //  对于每一个需要计算梯度的实体 
  for (const NodeEntry& e : xs)
  {
    LOG(INFO)<<"e.node->attrs.name"<<e.node->attrs.name;
    LOG(INFO)<<"e.index"<<e.index;
    //  属于这个节点的第几个梯度值。
    GradEntry& entry = output_grads[e.node.get()][e.index];
    // aggregate sum if there haven't been
    if (entry.sum.node.get() == nullptr) 
    {
      LOG(INFO)<<"entry.sum.node.get() == nullptr";
      entry.sum = agg_fun(std::move(entry.grads));
      if (entry.need_attr_hint && attr_hint_fun != nullptr) {
        entry.sum = attr_hint_fun(entry.sum, e);
      }
    }
    if (copy_op != nullptr) 
    {
      auto kv = unique_grads.find(entry.sum);
      if (kv == unique_grads.end())
       {
        unique_grads.emplace(std::move(entry.sum), std::make_pair(1, counter));
      }
      else 
      {
        NodePtr copy_node = Node::Create();
        std::ostringstream os;
        os << entry.sum.node->attrs.name << "_" << kv->second.first << "_copy";
        kv->second.first++;
        copy_node->attrs.op = copy_op;
        copy_node->attrs.name = os.str();
        copy_node->inputs.emplace_back(entry.sum);
        if (copy_node->attrs.op->attr_parser != nullptr) 
        {
            copy_node->attrs.op->attr_parser(&(copy_node->attrs));
        }
        unique_grads.emplace(NodeEntry{std::move(copy_node), 0, 0}, std::make_pair(1, counter));
      }
    } 
    else 
    {
        ret.outputs[counter] = entry.sum;
    }
    //  对于有效的输出的实体计数
    ++counter;
  }
  //LOG(INFO)<<"auto& out_grad_vec = output_grads.at(ptr.get());  out_grad_vec  在pass  out_grad_vec.size()  "<<out_grad_vec<<"   "<<out_grad_vec.size();
  if (copy_op != nullptr) 
  {
    for (const auto& kv : unique_grads) 
    {
      ret.outputs[kv.second.second] = kv.first;
    }
  }
  return ret;
}

// register pass
NNVM_REGISTER_PASS(Gradient)
.describe("Return a gradient graph of src.attrs[\"ys\"] wrt src.attrs[\"xs\"]")
.set_body(Gradient)
.set_change_graph(true)
.depend_graph_attr("grad_ys")
.depend_graph_attr("grad_xs")
.depend_graph_attr("grad_ys_out_grad");

}  // namespace
}  // namespace pass
}  // namespace nnvm
