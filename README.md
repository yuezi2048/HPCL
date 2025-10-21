# 代码编写

```bash
INFO AAA(
  (mlp): Linear(in_features=128, out_features=128, bias=True)
  (image_embedding): Embedding(7050, 4096)
  (image_trs): Linear(in_features=4096, out_features=64, bias=True)
  (text_embedding): Embedding(7050, 384)
  (text_trs): Linear(in_features=384, out_features=64, bias=True)
  (MLP_user): Linear(in_features=128, out_features=64, bias=True)
  (v_gcn): GCN(
    (MLP): Sequential(
      (0): Linear(in_features=4096, out_features=256, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=256, out_features=64, bias=True)
    )
    (shared_encoder): Linear(in_features=64, out_features=64, bias=True)
    (conv_embed_layers): ModuleList(
      (0-1): 2 x Base_gcn(64, 64)
    )
  )
  (v_gcn_n1): GCN(
    (MLP): Sequential(
      (0): Linear(in_features=4096, out_features=256, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=256, out_features=64, bias=True)
    )
    (shared_encoder): Linear(in_features=64, out_features=64, bias=True)
    (conv_embed_layers): ModuleList(
      (0-1): 2 x Base_gcn(64, 64)
    )
  )
  (v_gcn_n2): GCN(
    (MLP): Sequential(
      (0): Linear(in_features=4096, out_features=256, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=256, out_features=64, bias=True)
    )
    (shared_encoder): Linear(in_features=64, out_features=64, bias=True)
    (conv_embed_layers): ModuleList(
      (0-1): 2 x Base_gcn(64, 64)
    )
  )
  (t_gcn): GCN(
    (MLP): Sequential(
      (0): Linear(in_features=384, out_features=256, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=256, out_features=64, bias=True)
    )
    (shared_encoder): Linear(in_features=64, out_features=64, bias=True)
    (conv_embed_layers): ModuleList(
      (0-1): 2 x Base_gcn(64, 64)
    )
  )
  (t_gcn_n1): GCN(
    (MLP): Sequential(
      (0): Linear(in_features=384, out_features=256, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=256, out_features=64, bias=True)
    )
    (shared_encoder): Linear(in_features=64, out_features=64, bias=True)
    (conv_embed_layers): ModuleList(
      (0-1): 2 x Base_gcn(64, 64)
    )
  )
  (t_gcn_n2): GCN(
    (MLP): Sequential(
      (0): Linear(in_features=384, out_features=256, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=256, out_features=64, bias=True)
    )
    (shared_encoder): Linear(in_features=64, out_features=64, bias=True)
    (conv_embed_layers): ModuleList(
      (0-1): 2 x Base_gcn(64, 64)
    )
  )
  (id_gcn): GCN(
    (MLP): Sequential(
      (0): Linear(in_features=64, out_features=256, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=256, out_features=64, bias=True)
    )
    (shared_encoder): Linear(in_features=64, out_features=64, bias=True)
    (conv_embed_layers): ModuleList(
      (0-1): 2 x Base_gcn(64, 64)
    )
  )
  (shared_encoder): Linear(in_features=64, out_features=64, bias=True)
  (image_encoder_s): Linear(in_features=4096, out_features=64, bias=True)
  (text_encoder_s): Linear(in_features=384, out_features=64, bias=True)
  (query_common): Linear(in_features=64, out_features=1, bias=False)
  (softmax): Softmax(dim=-1)
)
```



核心故事主线：CMDL

目前小论文思路（只有解耦 + 用户注入过程）：

大论文可以融合因果正则'' + 模块E'

- MENTER：虽然更好地表示了模态融合过程，但是应当考虑到每个模态的特定模块。
- 特征提取_解耦共享和特定模态：CMDL'
  - CMDL——自注意力模块
  - 弱模态增强：M3CSR-2024？弱模态增强模块——POWEREc
- 行为信号注入自适应图扩散模块：DiffMM'
  -  **DiffMM-2024** 注入解耦的用户信号
- BPR损失函数改进
  - the BPR loss can only supervise the final user rating learning, but cannot optimize each modality-specific rating learning（CHOSION POWERec）



1. **C (UCPDA) 优先：** 原始多模态特征首先进入 UCPDA 模块。在这里，它们被解耦为模态共享和模态特定部分，并根据用户对不同模态的偏好强度进行动态组合。同时，通过多层次语义对齐和因果正则化，确保这些特征与推荐任务目标对齐，并消除虚假关联。输出的是**用户个性化的、解耦且对齐的多模态物品表示**。
2. **B (BGGD) 增强：** 接着，BGGD 模块介入，利用用户行为信号来**引导扩散过程**，对图上的信息进行**去噪和强化**。这个过程可以作用于 UCPDA 提供的精细化特征，从而使图信号的增强更具语义和偏好感知能力。
3. **A (MENTOR') 核心：** 最后，MENTOR' 作为核心 GCN 模型，接收来自 UCPDA 的**个性化、解耦且对齐的多模态特征**以及 ID 嵌入。GCN 在 BGGD 处理后的**行为引导去噪图结构**上进行信息传播，并进行最终的融合和表示学习。




好的，我来为你详细整理一下这个创新框架的完整多模态推荐过程，并明确每个阶段中“缝入”的模块、其修改点、参考论文以及修改的理由。

你提出的创新框架可以概括为：**MENTER' + BGAD + DMSP**。

这个框架的核心思想是：在 **MENTER** 这个强大的图神经网络基准模型之上，通过 **DMSP** 模块在**特征层面**进行精细化处理（解耦和个性化偏好），并通过 **BGAD** 模块在**图结构和信息传播层面**进行优化（行为引导去噪），从而实现更准确、鲁棒、个性化且可解释的多模态推荐。

------



### **整体多模态推荐过程概述**

一个典型的基于GNN的多模态推荐系统流程通常包括以下几个核心阶段：

1. **原始数据输入与初步嵌入：** 获取用户ID、物品ID以及物品的原始多模态数据（如图像、文本、音频）。
2. **图结构构建：** 基于用户-物品交互数据构建协同图，可能还包括物品-物品相似图。
3. **特征融合与传播：** 将多模态特征与ID嵌入结合，并通过GNN在图上传播信息，学习更丰富的用户和物品表示。
4. **自监督学习（SSL）：** 通常会引入SSL任务来增强表示学习，例如通过对比学习进行模态对齐或去噪。
5. **预测与排序：** 利用学习到的用户和物品表示，计算用户对物品的偏好分数，并进行排序推荐。

------



### **MENTER' + BGAD + DMSP 框架的详细分解与模块缝合**

以下是你的创新框架在上述通用流程中的具体实现和修改：

------



#### **阶段 1: 原始数据输入与初步嵌入**

- **原始模块：** 这是所有多模态推荐模型的基础步骤，不涉及直接修改。
- **缝入模块：** 无。
- **修改方面：** 无。
- **参考论文：**
  - **VBPR (2016)**: 早期将视觉特征与ID嵌入结合的代表 [Initial research report].
  - **MENTER-2025 / FREEDOM-2023 / LATTICE-2021**: 这些模型都依赖于预先提取的多模态特征和ID嵌入作为输入 [Initial research report].
- **具体作用：**
  - 获取物品的原始多模态特征（例如，通过预训练的视觉/文本编码器提取的特征向量）。
  - 为用户和物品生成可学习的ID嵌入。

------



#### **阶段 2: 模态特定偏好学习与解耦 (DMSP)**

- **原始模块：** 在MENTER中，这部分功能由其“多层次跨模态对齐”模块（Lately Fusion增强）承担，但DMSP对其进行了**根本性改进和前置**。
- **缝入模块：** **DMSP (基于因果正则化的解耦模态特定偏好学习)**
- **模块在哪个阶段：** 在原始多模态特征提取之后，GNN图传播之前。它负责对原始多模态特征进行**精细化预处理**。
- **修改的方面与参考论文：**
  1. **模态解耦表示：**
     - **修改内容：** DMSP会明确地将每种模态的特征分解为**模态共享嵌入**（捕获所有模态的共同语义）和**模态特定嵌入**（捕获仅在一种模态中存在的独特信息）。MENTER原有的融合可能直接对齐和融合模态，但DMSP在融合前就进行了这种结构化分解。
     - **参考论文：** **CMDL** (模态解耦表示，互信息正则化) [Initial research report]；**FMMRec** (公平导向的模态解耦，分离敏感和非敏感信息)。
     - **为何需要：** 解决多模态特征的“纠缠数值向量”问题 1，使模型能够区分共同和独特的模态贡献，为后续的个性化融合打下基础，提升可解释性。
  2. **用户特定动态组合：**
     - **修改内容：** 在解耦的基础上，DMSP会引入一个**用户特定的注意力或门控机制**。该机制根据每个用户的偏好强度，动态地权衡不同模态的共享和特定嵌入的重要性，并进行组合，生成**个性化的多模态物品表示**。
     - **参考论文：** **M3CSR-2024** (测量用户模态特定强度，通过成对损失解耦用户多模态兴趣) 2。
     - **为何需要：** 解决用户对不同模态偏好强度存在“差异”的问题 2，实现更精细的个性化推荐。同时，通过这种可学习的组合，解决了QARM中提到的“表示不可学习性”问题，使多模态特征能够通过梯度更新 4。
  3. **因果正则化：**
     - **修改内容：** 在解耦和组合过程中，DMSP会应用**因果正则化**，以确保模态特定组件和共享组件之间的统计独立性，并防止学习到虚假关联。
     - **参考论文：** **FMMRec** (因果启发式公平表示学习，通过对抗学习确保表示独立于敏感属性)；**Preference Learning for AI Alignment: a Causal Perspective (2025)** (将偏好学习框架置于因果范式中，解决因果误识别、偏好异质性等问题)。
     - **为何需要：** 确保解耦的有效性，提高可解释性，并减轻因虚假关联导致的偏差，从而提升模型的公平性。
- **输出：** 针对每个用户和物品的、经过解耦和个性化组合的、高质量的多模态特征表示。这些特征将作为GNN的输入。

------



#### **阶段 3: 图结构构建与行为引导扩散 (BGAD)**

- **原始模块：** MENTER (以及FREEDOM, LATTICE) 都有图结构构建和GCN传播模块。BGAD将**增强和优化**这个图传播过程。

- **缝入模块：** **BGAD (行为引导的自适应图扩散)**

- **模块在哪个阶段：** 在GNN进行信息传播之前或过程中。它负责对图结构和信息传播进行**去噪和行为引导**。

- **修改的方面与参考论文：**

  1. **图结构构建：** 沿用MENTER的图构建方式（用户-物品二分图，物品-物品同构图）。

  2. **行为信号注入与自适应扩散：**

     - **修改内容：** 在MENTER的GCN进行信息传播之前或过程中，引入一个**扩散过程**。这个扩散过程不再是通用的去噪，而是**由用户历史行为信号明确引导**的。它会根据行为信号的密度和质量，自适应地调整扩散步骤，优先处理与行为相关的多模态特征。

     - **参考论文：** **DiffMM-2024** (多模态图扩散模块，GNN与扩散结合) [Initial research report]；**GDMCF-2025** (图基扩散模型，多层次噪声破坏机制，用户活跃引导扩散过程) 5；

       **LD4MRec (2023)** (轻量级扩散模型，使用协同信号和个性化模态偏好信号作为引导)。

     - **为何需要：** 解决多模态数据中的噪声和“信息漂移”问题，确保扩散过程专注于真正反映用户偏好的特征，而非统计上突出但行为上不重要的特征。这使得模型从一般的物品内容相似性转向**与偏好相关**的内容相似性。

  3. **图剪枝/细化：**

     - **修改内容：** 在扩散过程中，根据行为相关性来**剪枝或细化多模态图中的边**，防止不相关或嘈杂的多模态信息通过图结构传播。

     - **参考论文：** **FREEDOM-2023** (去噪用户-物品交互图) 1；

       **GDMCF-2025** (选择性关注最有意义的边和活跃用户) 5。

     - **为何需要：** 解决复杂多模态图中的“噪声异构性”和“关系爆炸”问题 5，降低计算负担，同时保持拓扑完整性和信号质量，尤其有助于冷启动物品。

- **输出：** 经过行为引导去噪和细化后的、更鲁棒和行为对齐的用户和物品图表示。

------



#### **阶段 4: GNN-based 协同过滤与表示学习 (MENTER'核心)**

- **原始模块：** MENTER的核心GCN编码器（继承自FREEDOM和LATTICE）。
- **缝入模块：** MENTER'的GCN编码器将直接利用BGAD的输出。
- **模块在哪个阶段：** 这是模型的核心学习阶段，在DMSP和BGAD处理之后。
- **修改的方面与参考论文：**
  - **输入特征：** GCN的输入将是DMSP模块输出的**个性化、解耦的多模态特征**，以及原始ID嵌入。
  - **图结构：** GCN将在**BGAD模块处理后的、行为引导的去噪图结构**上进行信息传播和聚合。这意味着GCN在聚合邻居信息时，会优先考虑经过行为验证的、更相关的多模态信息。
  - **参考论文：** **MENTER-2025 / FREEDOM-2023 / LATTICE-2021** (GCN在双图上进行信息传播) [Initial research report]。
- **具体作用：** 通过GCN在优化后的图结构上进行多跳信息传播，捕获用户和物品之间的高阶协同信号，并学习到融合了多模态信息的用户和物品的最终表示。

------



#### **阶段 5: 融合与自监督学习 (MENTER'核心增强)**

- **原始模块：** MENTER的“多层次跨模态对齐”模块（Lately Fusion增强）和“混合自监督学习”模块。
- **缝入模块：** MENTER'的这些模块将处理来自DMSP和BGAD的增强特征。
- **模块在哪个阶段：** 在GNN表示学习之后，预测之前。
- **修改的方面与参考论文：**
  - **融合机制：** MENTER原有的“多层次跨模态对齐”机制将作用于**DMSP解耦并个性化组合后的特征**，以及**BGAD处理后的图表示**。这使得融合过程更加精细和有效，确保最终表示的语义一致性。
  - **自监督学习：** MENTER原有的“混合自监督学习”（特征dropout + 图扰动）将继续用于增强表示的鲁棒性，但现在它作用于更纯净、更具语义的特征。DMSP中的因果正则化也可以视为一种高级的自监督任务，确保解耦的有效性。
  - **参考论文：** **MENTER-2025** (多层次跨模态对齐，混合SSL) [Initial research report]。
- **具体作用：** 进一步细化和对齐用户和物品的表示，确保它们在共享潜在空间中具有语义一致性，并提升模型的泛化能力。

------



#### **阶段 6: 预测与排序**

- **原始模块：** 所有推荐模型的最终阶段。
- **缝入模块：** 无。
- **修改方面：** 无直接修改，使用最终学习到的用户和物品表示计算预测分数。
- **参考论文：** 任何推荐系统论文。
- **具体作用：** 根据学习到的用户和物品最终表示，计算用户对物品的偏好分数（例如，通过内积），并对物品进行排序，生成推荐列表。

------



### **代码实现思路总结**

你可以将这个框架理解为：

1. **DMSP层：** 在原始多模态特征输入后，先通过DMSP模块对物品的模态特征进行**解耦**和**用户个性化组合**，并施加**因果正则化**。这会输出一组更“干净”、更“个性化”的物品多模态嵌入。
2. **图构建与BGAD层：**
   - 构建MENTER所需的图结构（用户-物品二分图，物品-物品同构图）。
   - 在GCN进行图传播的**内部**或**之前**，引入BGAD模块。BGAD会利用用户行为信号来**引导扩散过程**，对图上的信息进行**去噪和强化**，确保只有与用户偏好强相关的多模态信息才有效传播。
3. **MENTER核心GCN与融合层：**
   - MENTER的GCN将接收DMSP输出的个性化多模态嵌入和ID嵌入。
   - GCN在BGAD处理后的**优化图结构**上进行信息传播。
   - MENTER原有的多层次融合和混合SSL机制将作用于这些经过DMSP和BGAD处理后的**高质量、行为对齐的表示**。
4. **预测层：** 最后，利用MENTER输出的最终用户和物品表示进行预测。

这个详细的分解应该能帮助你清晰地理解每个模块在整个框架中的位置、作用以及与相关论文的对应关系，从而指导你的代码实现。祝你编码顺利！