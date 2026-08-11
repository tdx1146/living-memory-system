"""怀疑融入（体验层 D，设计 v1.1 §6）—— 专注方向版。

怀疑 = 专注的质检员（dandan 拍板 2）：怀疑修正"已关注方向"内记忆的
信任权重（置信度），不改变 precision 的方向（专注）。所有机制对照
P1 原设计做"专注化审查"，两处修订（gap C 类降级 + 配额 relevance-gated），
其余按 P1 §2.1-2.5 落地。

模块：
  - confidence_field: 置信度场纯函数（更新/重算/强制降权/来源信任）
  - reconsolidation:  惊讶度双角色-角色2（去稳定化，labile 标记）
  - recall_scheduler: 召回唤起 salience（贝叶斯三项 + 低置信配额门控）
  - doubt_ingest:     /feed 结构化怀疑摄入（fail-open）
  - gap_registry:     信息缺口登记（A/B 类进怀疑灯，C 类仅诊断）

红线：surprise/free_energy/per_dim 语义、attractor infer/learn、
precision 三层结构、G1 归一化、doubt-system 四脚本 —— 全部零触碰。
"""
