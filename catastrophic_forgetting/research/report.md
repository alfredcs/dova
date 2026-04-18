# Catastrophic Forgetting Mitigation for 70B+ LLM Fine-Tuning

## Comprehensive Research Report

---

## 1. Background & Problem Statement

### 1.1 What Is Catastrophic Forgetting?

Catastrophic forgetting (CF) is the phenomenon where a neural network, upon learning new information, rapidly overwrites previously acquired knowledge. First identified by McCloskey & Cohen (1989) and French (1999), the problem has become especially critical in the era of large language models (LLMs).

When fine-tuning a pre-trained LLM on domain-specific data, the model's parameters shift to accommodate the new distribution, degrading performance on the original general capabilities. Empirical studies show this degradation can be severe and rapid -- for example, Reynolds (2025) demonstrated a 64.5 percentage point drop in NLI accuracy within the first 1,000 training steps when fine-tuning on mathematical reasoning data (arXiv:2512.13706).

### 1.2 Why 70B+ Models Are Especially Vulnerable

Several factors make catastrophic forgetting particularly severe at the 70B+ parameter scale:

1. **Massive parameter interdependence**: With ~70 billion parameters organized across 80+ transformer layers, weight updates propagate through deeply interconnected attention and feed-forward networks. A small perturbation in early layers cascades into significant representational drift in later layers.

2. **Empirical scaling of forgetting**: Luo et al. (2023) found that as model scale increases from 1B to 7B parameters, the severity of forgetting actually *intensifies*, likely because larger models achieve higher initial performance, leaving more room for degradation (arXiv:2308.08747). Imanov (2026) extended this analysis to 109B-400B models, identifying three primary forgetting mechanisms: gradient interference in attention weights, representational drift in intermediate layers, and loss landscape flattening (HF papers:2601.18699).

3. **Attention head disruption**: Approximately 15-23% of attention heads undergo severe disruption during fine-tuning, with lower layers showing greater susceptibility (Imanov, 2026).

4. **Scaling laws for forgetting**: Kalajdzievski (2024) established precise scaling laws showing forgetting increases as a shifted power law in both the number of parameters fine-tuned and the number of update steps: $F \propto (n - n_0)^{\alpha} \cdot (s - s_0)^{\beta}$ where $n$ is parameters fine-tuned, $s$ is update steps, and $n_0, s_0, \alpha, \beta$ are fitted constants (arXiv:2401.05605).

5. **Memory constraints**: Full fine-tuning of a 70B model requires ~280GB in fp32 (or ~140GB in fp16) just for parameters, plus optimizer states (Adam requires 2x parameter memory). This limits the practical approaches available.

### 1.3 Theoretical Framework

The forgetting problem can be understood through the lens of **loss landscape geometry**. Li et al. (2024) demonstrated a direct link between the flatness of the model loss landscape and the extent of CF -- sharper minima in the loss landscape correlate with more forgetting (arXiv:2406.04836). Fine-tuning drives the model toward a new minimum that may be far from the pre-training minimum in parameter space.

Formally, if $\theta^*$ denotes the pre-trained parameters and $\hat{\theta}$ the fine-tuned parameters, forgetting occurs when:

$$\mathcal{L}_{\text{pretrain}}(\hat{\theta}) \gg \mathcal{L}_{\text{pretrain}}(\theta^*)$$

The **NTK overlap matrix** (Doan et al., 2020) provides a theoretical measure: CF increases as two tasks increasingly align in the Neural Tangent Kernel space (arXiv:2010.04003).

---

## 2. Elastic Weight Consolidation (EWC)

### 2.1 Core Concept

Elastic Weight Consolidation (EWC), introduced by Kirkpatrick et al. (2017) in the seminal paper "Overcoming catastrophic forgetting in neural networks" (arXiv:1612.00796), draws inspiration from synaptic consolidation in neuroscience. The key insight is that not all parameters are equally important for a given task -- some weights are critical for previously learned knowledge, while others can be freely modified.

EWC identifies important parameters using the **Fisher Information Matrix (FIM)** and penalizes changes to those parameters during fine-tuning on new tasks.

### 2.2 Mathematical Formulation

Given a model with parameters $\theta$, after learning task $A$ with optimal parameters $\theta_A^*$, EWC modifies the loss function for learning task $B$ as:

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_{i} F_i \left(\theta_i - \theta_{A,i}^*\right)^2$$

Where:
- $\mathcal{L}_B(\theta)$ is the loss on the new task $B$
- $\lambda$ is the consolidation strength hyperparameter (controls the plasticity-stability trade-off)
- $F_i$ is the $i$-th diagonal element of the Fisher Information Matrix
- $\theta_{A,i}^*$ is the $i$-th parameter value after training on task $A$
- The sum runs over all parameters $i$

### 2.3 Fisher Information Matrix

The Fisher Information Matrix is defined as:

$$F = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \nabla_\theta \log p_\theta(x) \cdot \nabla_\theta \log p_\theta(x)^T \right]$$

For a classification/generation task with labels $y$:

$$F = \mathbb{E}_{x \sim p_{\text{data}}} \mathbb{E}_{y \sim p_\theta(y|x)} \left[ \nabla_\theta \log p_\theta(y|x) \cdot \nabla_\theta \log p_\theta(y|x)^T \right]$$

The full FIM is a $|\theta| \times |\theta|$ matrix. For a 70B model, this would require storing $\sim 70\text{B}^2$ entries -- clearly infeasible.

### 2.4 Diagonal Approximation

In practice, only the **diagonal** of the FIM is computed:

$$\hat{F}_i = \frac{1}{N} \sum_{n=1}^{N} \left( \frac{\partial \log p_\theta(y_n | x_n)}{\partial \theta_i} \right)^2$$

This reduces storage from $O(|\theta|^2)$ to $O(|\theta|)$. For a 70B model, this requires ~70B floats = ~280GB in fp32 (or ~140GB in fp16), which is substantial but feasible.

**Important implementation detail**: Van de Ven (2025) showed that the exact way the Fisher Information is computed significantly impacts EWC performance, and many reported results could be improved by changing the computation method (HF papers:2502.11756). Key variants include:
- **Empirical Fisher**: Uses true labels $y_n$ from the dataset
- **True Fisher**: Samples $y$ from the model's own distribution $p_\theta(y|x)$
- **Batch vs. single-sample**: Computing over batches provides smoother estimates

Soen & Sun (2024) provided rigorous analysis of trade-offs between diagonal FIM estimators, showing variance depends on non-linearity across parameter groups (arXiv:2402.05379).

**Kronecker-factored approximation (K-FAC)**: Chekalina et al. (2025) proposed Generalized Fisher-Weighted SVD (GFWSVD) using Kronecker-factored approximation of the observed Fisher information, accounting for both diagonal and off-diagonal elements. This outperforms diagonal-only approximations by ~5% on MMLU at 20x compression (arXiv:2505.17974).

**"Squisher" approximation**: Li et al. (2025) showed that Adam's squared gradient accumulator can approximate the Fisher diagonal "for free" -- recycling the already-computed moving average $v_t$ from Adam as a Fisher approximation, eliminating the separate Fisher computation pass entirely (arXiv:2507.18807).

### 2.5 Online EWC

For sequential learning of multiple tasks $A_1, A_2, \ldots, A_T$, standard EWC requires storing a separate Fisher matrix and parameter snapshot for each task, leading to linear memory growth. **Online EWC** (Schwarz et al., 2018) solves this by maintaining a running sum:

$$\tilde{F}_i^{(t)} = \gamma \tilde{F}_i^{(t-1)} + F_i^{(t)}$$

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_t(\theta) + \frac{\lambda}{2} \sum_i \tilde{F}_i^{(t-1)} (\theta_i - \theta_{t-1,i}^*)^2$$

Where $\gamma \in [0,1]$ is a decay factor that down-weights older tasks.

### 2.6 Scalability to 70B+ Models

**Memory overhead**: The diagonal FIM requires one extra scalar per parameter. For a 70B model in fp16, this is ~140GB additional memory. In fp32, it is ~280GB.

**Compute overhead**: Computing the FIM requires a forward-backward pass over a representative dataset (typically a few hundred to a few thousand samples). For a 70B model, this takes several GPU-hours.

**Practical mitigation**:
- Use fp16/bf16 for FIM storage to halve memory
- Compute FIM only for a subset of layers (e.g., attention weights only, since 15-23% of attention heads are most vulnerable)
- Use the "Squisher" approach to recycle Adam's $v_t$ accumulator as a free Fisher approximation
- Apply Kronecker-factored approximation for better accuracy with manageable overhead

### 2.7 EWC Algorithm / Pseudocode

```python
def ewc_training(model, old_params, fisher_diag, new_dataloader,
                 lambda_ewc=1000, epochs=3, lr=1e-5):
    """
    Elastic Weight Consolidation for LLM fine-tuning.

    Args:
        model: Pre-trained LLM (e.g., 70B parameter model)
        old_params: Dict of parameter snapshots {name: tensor} from pre-training
        fisher_diag: Dict of diagonal Fisher {name: tensor} computed on pre-training data
        new_dataloader: DataLoader for the new domain-specific task
        lambda_ewc: Consolidation strength (higher = more preservation)
        epochs: Number of training epochs
        lr: Learning rate
    """
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in new_dataloader:
            # 1. Compute task loss
            outputs = model(**batch)
            task_loss = outputs.loss

            # 2. Compute EWC penalty
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in fisher_diag:
                    ewc_loss += (fisher_diag[name] *
                                 (param - old_params[name]).pow(2)).sum()

            # 3. Combined loss
            total_loss = task_loss + (lambda_ewc / 2) * ewc_loss

            # 4. Update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


def compute_fisher_diagonal(model, dataloader, num_samples=1000):
    """
    Compute diagonal of the Fisher Information Matrix.

    For 70B models, use gradient checkpointing and process in chunks.
    Alternative: reuse Adam's v_t accumulator ("Squisher" method).
    """
    fisher = {name: torch.zeros_like(param)
              for name, param in model.named_parameters()}

    model.eval()
    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break

        outputs = model(**batch)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

        # Sample from model's own distribution (true Fisher)
        # Or use true labels (empirical Fisher -- simpler, often sufficient)
        labels = batch["input_ids"]  # empirical Fisher
        nll = F.nll_loss(log_probs.view(-1, log_probs.size(-1)),
                         labels.view(-1), reduction='sum')

        model.zero_grad()
        nll.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data.pow(2)

        count += batch["input_ids"].size(0)

    # Average
    for name in fisher:
        fisher[name] /= count

    return fisher
```

### 2.8 Hyperparameter Guidance for 70B+

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| $\lambda$ (consolidation strength) | 100 -- 10,000 | Start at 1000; increase if forgetting persists |
| Fisher samples | 500 -- 2,000 | More samples = more stable FIM estimate |
| Learning rate | 1e-6 -- 5e-5 | Lower than standard fine-tuning |
| Fisher computation | Empirical Fisher | Simpler, sufficient for most cases |

---

## 3. Adapter Isolation (LoRA/QLoRA)

### 3.1 Core Concept

Low-Rank Adaptation (LoRA), introduced by Hu et al. (2021) (arXiv:2106.09685), takes a fundamentally different approach to preventing forgetting: **freeze the entire pre-trained model** and only train small, additive low-rank matrices. Since the base model weights $W_0$ remain unchanged, the original knowledge is perfectly preserved when the adapters are removed.

### 3.2 Mathematical Formulation

For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the weight update as:

$$W = W_0 + \Delta W = W_0 + BA$$

Where:
- $B \in \mathbb{R}^{d \times r}$ (projection up)
- $A \in \mathbb{R}^{r \times k}$ (projection down)
- $r \ll \min(d, k)$ is the rank

The forward pass becomes:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

Where $\alpha$ is a scaling factor. The key insight from Kalajdzievski (2023) is that the scaling should be $\alpha / \sqrt{r}$ (rsLoRA) rather than $\alpha / r$ for rank-stabilized training (HF papers:2312.03732).

**Initialization**: $A$ is initialized from $\mathcal{N}(0, \sigma^2)$ and $B$ is initialized to zero, so $\Delta W = 0$ at the start of training, ensuring the model begins from the pre-trained state.

### 3.3 How Frozen Weights Prevent Forgetting

The key advantage of LoRA for forgetting prevention:

1. **Base weights are untouched**: $W_0$ retains all pre-trained knowledge exactly
2. **Additive composition**: New knowledge lives in $BA$, which can be merged or removed
3. **Low-rank constraint**: The update $\Delta W$ is constrained to rank $r$, limiting the expressivity of changes and acting as implicit regularization

However, Kalajdzievski (2024) showed that even LoRA suffers from forgetting, following a scaling law where forgetting increases as a shifted power law in rank and update steps (arXiv:2401.05605). The forgetting is less severe than full fine-tuning but is not zero.

### 3.4 Trainable Parameters

For a 70B model, LoRA dramatically reduces trainable parameters:

| Configuration | Trainable Params | % of Total |
|--------------|-----------------|------------|
| Full fine-tuning | 70B | 100% |
| LoRA r=8, all attention | ~67M | 0.096% |
| LoRA r=16, all attention | ~134M | 0.19% |
| LoRA r=64, all attention | ~537M | 0.77% |
| LoRA r=16, attn + MLP | ~402M | 0.57% |

### 3.5 Task-Specific Adapter Strategies

**Adapter Banks**: Train separate LoRA adapters for each task/domain. At inference, select the appropriate adapter. This provides perfect isolation -- each adapter captures only its task's knowledge, and the base model serves as a shared backbone.

**LoRA routing / MoE-LoRA**: Sun et al. (2025) proposed combining LoRA with Mixture-of-Experts, where multiple LoRA adapters are trained and a router selects which to apply based on the input (HF papers:2502.15828).

**C-LoRA (Continual LoRA)**: Zhang et al. (2025) proposed using a learnable routing matrix to dynamically manage parameter updates across tasks, ensuring efficient reuse of learned subspaces while enforcing orthogonality to minimize interference (HF papers:2502.17920).

**Merge before Forget**: Qiao & Mahdavi (2025) proposed orthogonally initializing and sequentially merging LoRA updates into a single unified LoRA, maintaining constant memory complexity regardless of task count (arXiv:2512.23017).

**CURLoRA**: Fawi (2024) leverages CUR matrix decomposition for LoRA, using inverted probabilities for column/row selection as implicit regularization and initializing the $U$ matrix to zero. CURLoRA outperforms standard LoRA in mitigating forgetting while using fewer trainable parameters (arXiv:2408.14572).

### 3.6 QLoRA: Quantization + Adapters for Memory Efficiency

QLoRA (Dettmers et al., 2023; arXiv:2305.14314) combines 4-bit quantization with LoRA to enable fine-tuning of very large models on limited hardware:

$$h = \text{dequant}(W_0^{\text{4bit}}) \cdot x + \frac{\alpha}{r} B A x$$

Key innovations:
- **4-bit NormalFloat (NF4)**: Information-theoretically optimal quantization for normally distributed weights
- **Double quantization**: Quantizes the quantization constants, saving ~0.37 bits per parameter
- **Paged optimizers**: Uses CPU memory for optimizer state spikes via unified memory

**Memory requirements for 70B model**:

| Method | GPU Memory |
|--------|-----------|
| Full fine-tuning (fp16) | ~280GB+ |
| LoRA (fp16 base) | ~140GB+ |
| QLoRA (4-bit base) | ~35-48GB |

QLoRA makes 70B fine-tuning feasible on a single 48GB GPU (e.g., A100), achieving performance comparable to full 16-bit fine-tuning.

### 3.7 Rank Selection for Knowledge Preservation

The rank $r$ controls the trade-off between expressivity and forgetting:

- **Lower rank (r=4-8)**: Less forgetting, but may underfit new tasks
- **Medium rank (r=16-64)**: Good balance for most domain adaptation tasks
- **Higher rank (r=128-256)**: Better task performance but more forgetting

The CoDyRA approach (Lu et al., 2024) proposes adaptive rank selection per parameter and per task, using sparsity-promoting regularization to minimize ranks and reduce interference (arXiv:2412.01004).

**Layer-wise rank allocation**: Ogawa et al. (2026) showed that not all layers contribute equally -- selectively fine-tuning only the most relevant layers with LoRA can reduce trainable parameters by up to 50% while maintaining performance (arXiv:2602.05988).

### 3.8 LoRA/QLoRA Algorithm / Pseudocode

```python
def lora_finetune_70b(base_model_id, dataset, rank=16, alpha=32,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                       use_qlora=True, lr=2e-4, epochs=3):
    """
    LoRA/QLoRA fine-tuning for 70B+ models.

    Args:
        base_model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-70b")
        dataset: Fine-tuning dataset
        rank: LoRA rank (16-64 recommended for 70B)
        alpha: LoRA scaling factor (typically 2*rank)
        target_modules: Which layers to adapt
        use_qlora: Whether to use 4-bit quantization
        lr: Learning rate (higher than full FT due to fewer params)
        epochs: Training epochs
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    # Step 1: Load model (quantized if QLoRA)
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NormalFloat4
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,       # Double quantization
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

    # Step 2: Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: "trainable params: ~134M || all params: 70B || 0.19%"

    # Step 3: Train
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=lr,
            num_train_epochs=epochs,
            fp16=False, bf16=True,
            optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
        ),
    )
    trainer.train()

    # Step 4: Save adapter (small, ~500MB for r=16)
    model.save_pretrained("./lora_adapter")
    # Base model knowledge is perfectly preserved --
    # just remove the adapter to restore original model


def adapter_bank_inference(base_model, adapter_paths, task_id):
    """
    Load task-specific adapter at inference time.
    Multiple adapters share the same base model in memory.
    """
    from peft import PeftModel

    # Load base model once (shared across all tasks)
    model = PeftModel.from_pretrained(base_model, adapter_paths[task_id])

    # To switch tasks, just swap the adapter:
    # model.load_adapter(adapter_paths[other_task_id])

    return model
```

### 3.9 Hyperparameter Guidance for 70B+

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| Rank $r$ | 16 -- 64 | Higher rank = more capacity but more forgetting |
| $\alpha$ | $2r$ | Standard scaling; use $\alpha/\sqrt{r}$ for rsLoRA |
| Target modules | q, k, v, o projections | Add MLP layers for more capacity |
| Learning rate | 1e-4 -- 3e-4 | Higher than full FT |
| Batch size (effective) | 16 -- 64 | Use gradient accumulation |
| QLoRA quant type | NF4 | Optimal for normally distributed weights |

---

## 4. Replay Buffers / Experience Replay

### 4.1 Core Concept

Experience replay combats forgetting by mixing samples from previous tasks/distributions into the current training data. By periodically revisiting old data, the model maintains its representation of prior knowledge. This is directly analogous to the interleaving effect in human learning.

### 4.2 Reservoir Sampling

When the total dataset is too large to store, **reservoir sampling** maintains a fixed-size buffer that is a uniform random sample of all data seen so far:

```
Algorithm: Reservoir Sampling for Continual Learning

Initialize: buffer B of size k, count n = 0

For each new training example x:
    n += 1
    if |B| < k:
        Add x to B
    else:
        j = random_integer(1, n)
        if j <= k:
            Replace B[j] with x
```

This guarantees that at any point, each example seen so far has equal probability $k/n$ of being in the buffer.

### 4.3 Importance-Weighted Replay

Not all examples are equally valuable for preventing forgetting. Importance-weighted strategies prioritize examples that are most informative:

**Gradient-based selection (GCR)**: Tiwari et al. (2021) proposed maintaining a coreset that closely approximates the gradient of all data seen so far. This achieved 2-4% absolute gains over uniform replay (arXiv:2111.11210).

**Adaptive Experience Replay (AdaER)**: Li et al. (2023) introduced contextually-cued memory recall that selectively replays memories most conflicting with current input, combined with entropy-balanced reservoir sampling to maximize information content (arXiv:2308.03810).

**Loss-based selection**: Prioritize examples with the highest loss or largest loss increase since last training, indicating the model is beginning to forget them.

### 4.4 Generative Replay

Instead of storing actual data (which may be impractical or raise privacy concerns), **generative replay** uses the model itself to generate synthetic examples representative of previous knowledge:

1. Before fine-tuning on new data, use the current model to generate representative outputs
2. During fine-tuning, mix generated examples with new task data
3. The generated examples act as a proxy for the original training distribution

For LLMs, this is particularly natural:

```
Algorithm: Generative Replay for LLM Fine-Tuning

1. Before fine-tuning:
   - Sample diverse prompts P = {p_1, ..., p_m} covering general capabilities
   - Generate responses: R_i = model.generate(p_i) for each prompt
   - Store (P, R) as the replay dataset

2. During fine-tuning:
   For each training step:
     - Sample batch_new from new domain data
     - Sample batch_replay from (P, R) with probability p_replay
     - Combined batch = mix(batch_new, batch_replay, ratio=mix_ratio)
     - Train on combined batch
```

**Self-Distillation Fine-Tuning (SDFT)**: Yang et al. (2024) formalized this as self-distillation, where the model generates a dataset matching its original distribution, which is then used to bridge the distribution gap during fine-tuning. This effectively mitigates CF while achieving comparable performance on downstream tasks (HF papers:2402.13669).

### 4.5 Mixed Training (Data Interleaving)

Reynolds (2025) demonstrated that simply interleaving original-domain and new-domain examples during training can completely eliminate catastrophic forgetting:

- **1:1 ratio**: 12.0% math accuracy (matching task-only training) while preserving 86.2% NLI accuracy
- **Even 15:1 ratio** (93.8% new data, 6.2% old data): Still provides effective regularization against forgetting

This is the simplest and most effective replay strategy, requiring only a small fraction of general-domain data (arXiv:2512.13706).

### 4.6 Memory Budget Analysis for 70B+ Training

| Buffer Strategy | Storage Required | Overhead |
|----------------|-----------------|----------|
| Raw token buffer (10K examples, 2048 tokens) | ~80MB (int32 token IDs) | Negligible |
| Raw token buffer (100K examples) | ~800MB | Negligible |
| Gradient coreset (10K examples + metadata) | ~160MB | Minimal |
| Generative replay (10K prompt-response pairs) | ~200MB | Requires generation pass |

The storage overhead of replay buffers is negligible compared to the model itself (~35-140GB depending on precision). The main cost is the compute overhead of training on replay examples.

### 4.7 Experience Replay Algorithm / Pseudocode

```python
class ReplayBuffer:
    """Fixed-size replay buffer with reservoir sampling."""

    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        self.count = 0

    def add(self, example):
        self.count += 1
        if len(self.buffer) < self.max_size:
            self.buffer.append(example)
        else:
            # Reservoir sampling
            j = random.randint(0, self.count - 1)
            if j < self.max_size:
                self.buffer[j] = example

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


class ImportanceWeightedBuffer(ReplayBuffer):
    """Buffer that prioritizes high-loss examples."""

    def __init__(self, max_size=10000):
        super().__init__(max_size)
        self.losses = []

    def add_with_loss(self, example, loss):
        self.count += 1
        if len(self.buffer) < self.max_size:
            self.buffer.append(example)
            self.losses.append(loss)
        else:
            # Replace lowest-loss example
            min_idx = np.argmin(self.losses)
            if loss > self.losses[min_idx]:
                self.buffer[min_idx] = example
                self.losses[min_idx] = loss

    def sample_weighted(self, batch_size):
        """Sample with probability proportional to loss."""
        probs = np.array(self.losses) / sum(self.losses)
        indices = np.random.choice(len(self.buffer), size=batch_size,
                                    p=probs, replace=False)
        return [self.buffer[i] for i in indices]


def replay_finetuning(model, new_data, replay_buffer,
                       mix_ratio=0.1, lr=2e-5, epochs=3):
    """
    Fine-tuning with experience replay.

    Args:
        model: Pre-trained LLM
        new_data: New domain-specific training data
        replay_buffer: Buffer of general-domain examples
        mix_ratio: Fraction of each batch from replay (0.05-0.2 recommended)
        lr: Learning rate
        epochs: Training epochs
    """
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for new_batch in new_data:
            batch_size = len(new_batch)
            replay_size = max(1, int(batch_size * mix_ratio / (1 - mix_ratio)))

            # Sample replay examples
            replay_batch = replay_buffer.sample(replay_size)

            # Combine batches
            combined_batch = concatenate(new_batch, replay_batch)
            shuffle(combined_batch)

            # Standard training step
            outputs = model(**combined_batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def generative_replay_setup(model, tokenizer, num_prompts=5000):
    """
    Generate replay data using the model itself before fine-tuning.

    This captures the model's current knowledge distribution
    without needing access to the original training data.
    """
    prompts = [
        # Diverse prompts covering general capabilities
        "Explain the concept of",
        "What are the main differences between",
        "Summarize the following topic:",
        "Write a short paragraph about",
        "Solve the following problem:",
        # ... generate diverse prompts programmatically
    ]

    replay_data = []
    for prompt in prompts[:num_prompts]:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256,
                temperature=0.7, top_p=0.9,
                do_sample=True
            )
        response = tokenizer.decode(outputs[0])
        replay_data.append({"text": response})

    return replay_data
```

### 4.8 Hyperparameter Guidance for 70B+

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| Buffer size | 5,000 -- 50,000 examples | Larger = better; storage is cheap |
| Mix ratio (replay fraction) | 0.05 -- 0.20 | Even 5% old data helps significantly |
| Sampling strategy | Uniform or loss-weighted | Uniform is simple and effective |
| Generative replay prompts | 5,000 -- 20,000 | Cover diverse capabilities |
| Generation temperature | 0.7 -- 1.0 | Higher = more diverse replay data |

---

## 5. Hybrid Approaches

### 5.1 EWC + LoRA

Combining EWC with LoRA provides dual protection:

1. **LoRA freezes base weights**: Prevents direct modification of pre-trained knowledge
2. **EWC regularizes adapter weights**: Prevents the adapter itself from learning representations that interfere with base model capabilities

This is especially useful when the adapter needs to be large (high rank) for complex domains:

```python
def ewc_lora_training(model, lora_config, new_data,
                       fisher_diag_adapter, old_adapter_params,
                       lambda_ewc=500, lr=2e-4):
    """
    EWC applied to LoRA adapter parameters only.

    After training adapter on task A, compute Fisher for adapter params.
    When training on task B, penalize drift of adapter params from task A.
    """
    peft_model = get_peft_model(model, lora_config)
    optimizer = AdamW(peft_model.parameters(), lr=lr)

    for batch in new_data:
        task_loss = peft_model(**batch).loss

        # EWC penalty on adapter parameters only
        ewc_loss = 0.0
        for name, param in peft_model.named_parameters():
            if "lora_" in name and name in fisher_diag_adapter:
                ewc_loss += (fisher_diag_adapter[name] *
                             (param - old_adapter_params[name]).pow(2)).sum()

        total_loss = task_loss + (lambda_ewc / 2) * ewc_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Advantage**: The Fisher only needs to be computed over the adapter parameters (~134M for r=16), which is ~500x cheaper than computing it over all 70B parameters.

### 5.2 Replay + Adapter Isolation

The most practical hybrid for 70B+ models:

1. Freeze base model with QLoRA (4-bit quantization)
2. Train LoRA adapters on new domain data
3. Mix in replay data (5-10% of each batch) from original distribution
4. Optionally apply EWC to the adapter parameters

This provides three layers of protection:
- **Frozen base**: Preserves core knowledge
- **Low-rank constraint**: Limits capacity for destructive updates
- **Replay regularization**: Maintains alignment with original distribution

### 5.3 Bayesian PEFT

Chen & Garner (2024) demonstrated that Bayesian learning techniques applied to PEFT (specifically LoRA) can prevent catastrophic forgetting. Using Laplace approximations (diagonal and Kronecker-factored) to regularize LoRA, they showed forgetting can be overcome without degrading fine-tuning performance (arXiv:2402.12220). The Kronecker-factored approximation provided better knowledge preservation than diagonal methods.

### 5.4 Half Fine-Tuning (HFT)

Hui et al. (2024) introduced a simple but effective approach: freeze half the parameters and train the other half, alternating which half is frozen. This acts as implicit regularization, preserving knowledge in the frozen parameters while updating the active ones. HFT achieves ~30% reduction in training time while alleviating forgetting (HF papers:2404.18466).

### 5.5 Dynamic Orthogonal Continual (DOC) Fine-Tuning

Zhang et al. (2025) proposed tracking the drift of "functional directions" during fine-tuning and dynamically updating them. By constraining new-task gradients to be orthogonal to historical functional directions, DOC minimizes interference between new and old tasks (arXiv:2509.23893).

### 5.6 Sparse Memory Fine-Tuning

Lin et al. (2025) leveraged memory layer models where only sparsely activated memory slots are updated. By updating only slots highly activated by new data relative to pre-training usage, interference is minimized. This achieved only an 11% drop in NaturalQuestions F1 vs. 89% for full fine-tuning and 71% for LoRA (HF papers:2510.15103).

### 5.7 Sharpness-Aware Minimization (SAM) for LLMs

Li et al. (2024) showed that flattening the loss landscape via SAM complements existing anti-forgetting strategies. SAM seeks parameters in flatter minima that are more robust to perturbation, directly reducing forgetting. It can be combined with any of the above methods (arXiv:2406.04836).

---

## 6. Comparative Analysis Table

| Method | Memory Overhead (70B) | Compute Overhead | Forgetting Mitigation | New Task Performance | Implementation Complexity | Practical for 70B? |
|--------|----------------------|-----------------|----------------------|---------------------|--------------------------|-------------------|
| **Full Fine-Tuning** | Baseline (~280GB fp16) | Baseline | None | Best | Low | Requires multi-GPU |
| **EWC** | +140GB (FIM storage, fp16) | +20-30% (FIM compute + penalty) | Moderate | Slightly reduced | Medium | Challenging (memory) |
| **LoRA (r=16)** | +~0.5GB (adapters) | -60% (fewer params to update) | Good (frozen base) | Good | Low | Yes (with model parallelism) |
| **QLoRA (4-bit + r=16)** | ~35-48GB total | -70% vs full FT | Good (frozen base) | Good | Low | **Yes (single GPU)** |
| **Replay Buffer** | +0.1-1GB (buffer) | +5-20% (replay steps) | Good | Maintained | Low | Yes |
| **Generative Replay** | +0.2GB (generated data) | +generation cost | Good | Maintained | Medium | Yes |
| **Mixed Training** | Requires old data access | +5-10% | Excellent | Equivalent | Very Low | Yes |
| **EWC + LoRA** | +~0.5GB | +10% | Very Good | Good | Medium | Yes |
| **QLoRA + Replay** | ~35-48GB + buffer | +5-15% | Very Good | Good | Low-Medium | **Yes (recommended)** |
| **Sparse Memory FT** | Model-specific | Moderate | Excellent (11% drop) | Good | High | Research stage |
| **DOC Fine-Tuning** | +gradient history | +15-25% | Excellent | Good | High | Research stage |

### Recommendations Ranked by Practicality for 70B+ Models

1. **QLoRA + Mixed Training / Replay** -- Best practical option. Single-GPU feasible, excellent forgetting mitigation.
2. **LoRA with adapter banks** -- Multiple task-specific adapters sharing one base model. Zero forgetting by design.
3. **QLoRA + EWC on adapters** -- When adapter-level forgetting across sequential tasks is a concern.
4. **Full LoRA (fp16) + Replay** -- When QLoRA quality loss is unacceptable.
5. **EWC alone** -- Only if you have sufficient memory (multi-node) and need full parameter fine-tuning.

---

## 7. Implementation Recommendations

### 7.1 Recommended Training Recipe for 70B+ Models

```
Phase 1: Preparation
  - Load base model with QLoRA (NF4, double quantization)
  - Configure LoRA: r=32, alpha=64, target=q,k,v,o projections
  - Prepare replay buffer:
    Option A: Sample 10K-50K examples from general-domain data
    Option B: Generate 10K examples via generative replay

Phase 2: Training
  - Effective batch size: 32 (batch_size=1, grad_accum=32)
  - Learning rate: 2e-4 with cosine schedule, warmup=3%
  - Mix ratio: 10% replay, 90% new domain data
  - Train for 1-3 epochs (monitor forgetting metrics!)
  - Use gradient checkpointing to reduce memory
  - Optionally: apply SAM for flatter minima

Phase 3: Evaluation
  - Measure new task performance (domain-specific benchmarks)
  - Measure forgetting (general benchmarks: MMLU, HellaSwag, ARC, etc.)
  - Compare base model vs. fine-tuned model on general benchmarks
  - Compute forgetting ratio: F = 1 - (score_finetuned / score_base)
```

### 7.2 Specific Hyperparameter Ranges

| Hyperparameter | Range | Best Starting Point |
|---------------|-------|-------------------|
| LoRA rank | 16 -- 64 | 32 |
| LoRA alpha | 2r | 64 |
| LoRA dropout | 0.0 -- 0.1 | 0.05 |
| Learning rate | 5e-5 -- 3e-4 | 2e-4 |
| Batch size (effective) | 16 -- 64 | 32 |
| Replay mix ratio | 0.05 -- 0.20 | 0.10 |
| EWC lambda (if used) | 100 -- 10,000 | 1,000 |
| EWC lambda for adapters | 10 -- 1,000 | 100 |
| Fisher samples | 500 -- 2,000 | 1,000 |
| Warmup ratio | 0.01 -- 0.05 | 0.03 |
| Epochs | 1 -- 5 | 3 |
| Max sequence length | 1024 -- 4096 | 2048 |
| Weight decay | 0.0 -- 0.1 | 0.01 |

### 7.3 Evaluation Metrics for Measuring Forgetting vs. Acquisition

**Forgetting metrics**:
- **Backward Transfer (BWT)**: $\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} (a_{T,i} - a_{i,i})$ where $a_{t,i}$ is accuracy on task $i$ after learning task $t$
- **Forgetting Ratio**: $F = 1 - \frac{\text{score}_{\text{finetuned}}}{\text{score}_{\text{base}}}$ on general benchmarks
- **Perplexity drift**: Change in perplexity on a held-out general corpus

**Acquisition metrics**:
- **Forward Transfer**: Performance improvement on target domain
- **Task-specific accuracy/F1/BLEU** on domain benchmarks

**Combined metrics**:
- **H-Mean**: Harmonic mean of general preservation and task performance
- **Area Under the Forgetting Curve**: Track general performance over training steps

Ashley et al. (2021) strongly recommend measuring forgetting with both **retention** and **relearning** metrics concurrently, as conclusions can change dramatically depending on the metric (arXiv:2102.07686).

### 7.4 Monitoring During Training

```python
# Evaluate every N steps on both new task AND general benchmarks
eval_steps = 100

# Key signals to watch:
# 1. General benchmark scores dropping -> increase replay ratio or lambda
# 2. New task not improving -> decrease lambda, increase rank
# 3. Both declining -> learning rate too high, reduce it
# 4. Perplexity on general corpus spiking -> early sign of forgetting
```

---

## 8. References

### 8.1 Foundational Papers

| Paper | arXiv ID | Key Contribution |
|-------|----------|-----------------|
| Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (2017) | [1612.00796](https://arxiv.org/abs/1612.00796) | Original EWC method |
| Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021) | [2106.09685](https://arxiv.org/abs/2106.09685) | Original LoRA method |
| Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) | [2305.14314](https://arxiv.org/abs/2305.14314) | QLoRA 4-bit fine-tuning |
| Rolnick et al. "Experience Replay for Continual Learning" (2019) | [1811.11682](https://arxiv.org/abs/1811.11682) | Experience replay for CL |

### 8.2 Catastrophic Forgetting in LLMs

| Paper | arXiv ID / Link | Key Contribution |
|-------|-----------------|-----------------|
| Luo et al. "An Empirical Study of CF in LLMs During Continual Fine-tuning" (2023) | [2308.08747](https://arxiv.org/abs/2308.08747) | Forgetting scales with model size |
| Kalajdzievski "Scaling Laws for Forgetting When Fine-Tuning LLMs" (2024) | [2401.05605](https://hf.co/papers/2401.05605) | Power-law scaling of forgetting |
| Li et al. "Revisiting Catastrophic Forgetting in LLM Tuning" (2024) | [2406.04836](https://arxiv.org/abs/2406.04836) | Loss landscape flatness link |
| Imanov "Mechanistic Analysis of CF in LLMs" (2026) | [HF:2601.18699](https://hf.co/papers/2601.18699) | Gradient interference, attention head disruption |
| Reynolds "Mitigating CF through Mixed Training" (2025) | [2512.13706](https://arxiv.org/abs/2512.13706) | Mixed training eliminates forgetting |
| Ding & Wang "Improved SFT to Mitigate CF" (2025) | [2506.09428](https://arxiv.org/abs/2506.09428) | Synthetic data for replay |
| Song et al. "Hierarchical Layer-Wise and Element-Wise Regularization" (2025) | [HF:2501.13669](https://hf.co/papers/2501.13669) | Element-wise importance |
| Li & Lee "Examining Forgetting in Continual Pre-training" (2024) | [HF:2401.03129](https://hf.co/papers/2401.03129) | Forgetting during continual pre-training |

### 8.3 EWC and Regularization Methods

| Paper | arXiv ID / Link | Key Contribution |
|-------|-----------------|-----------------|
| Thorne & Vlachos "EWC for Better Bias Inoculation" (2020) | [2004.14366](https://arxiv.org/abs/2004.14366) | EWC for NLP tasks |
| Ahadzi et al. "Continuous Learning for ASR: EWC and SI" (2025) | [2505.20216](https://arxiv.org/abs/2505.20216) | EWC + Synaptic Intelligence |
| Liu et al. "Rotate your Networks: Better Weight Consolidation" (2018) | [1802.02950](https://arxiv.org/abs/1802.02950) | Rotation for diagonal FIM |
| Batra & Clark "EVCL: Elastic Variational CL with Weight Consolidation" (2024) | [2406.15972](https://arxiv.org/abs/2406.15972) | VCL + EWC hybrid |
| Van de Ven "On the Computation of Fisher Information in CL" (2025) | [HF:2502.11756](https://hf.co/papers/2502.11756) | Fisher computation matters |
| Soen & Sun "Trade-Offs of Diagonal Fisher Estimators" (2024) | [2402.05379](https://arxiv.org/abs/2402.05379) | Variance analysis of FIM estimators |
| Chekalina et al. "Generalized Fisher-Weighted SVD" (2025) | [2505.17974](https://arxiv.org/abs/2505.17974) | Kronecker-factored Fisher for LLM compression |
| Li et al. "Fishers for Free? Squisher" (2025) | [2507.18807](https://arxiv.org/abs/2507.18807) | Adam v_t as free Fisher approximation |
| Doan et al. "Theoretical Analysis via NTK Overlap" (2020) | [2010.04003](https://arxiv.org/abs/2010.04003) | NTK-based forgetting theory |

### 8.4 LoRA and Adapter Methods

| Paper | arXiv ID / Link | Key Contribution |
|-------|-----------------|-----------------|
| Fawi "CURLoRA: Stable LLM Continual Fine-Tuning" (2024) | [2408.14572](https://arxiv.org/abs/2408.14572) | CUR decomposition for LoRA |
| Chen & Garner "Bayesian PEFT for Overcoming CF" (2024) | [2402.12220](https://arxiv.org/abs/2402.12220) | Laplace approximation on LoRA |
| Lu et al. "CoDyRA: Dynamic Rank-Selective LoRA" (2024) | [2412.01004](https://arxiv.org/abs/2412.01004) | Adaptive rank for CL |
| Qiao & Mahdavi "Merge before Forget" (2025) | [2512.23017](https://arxiv.org/abs/2512.23017) | Orthogonal LoRA merging |
| Kalajdzievski "Rank Stabilization Scaling Factor (rsLoRA)" (2023) | [HF:2312.03732](https://hf.co/papers/2312.03732) | Proper alpha/sqrt(r) scaling |
| Zhang et al. "C-LoRA: Continual Low-Rank Adaptation" (2025) | [HF:2502.17920](https://hf.co/papers/2502.17920) | Routing matrix for CL |
| Ogawa et al. "Layer-wise LoRA fine-tuning" (2026) | [2602.05988](https://arxiv.org/abs/2602.05988) | Selective layer adaptation |
| Lermen et al. "LoRA Fine-tuning Undoes Safety Training in 70B" (2023) | [2310.20624](https://arxiv.org/abs/2310.20624) | QLoRA on 70B feasibility |
| Hui et al. "HFT: Half Fine-Tuning" (2024) | [HF:2404.18466](https://hf.co/papers/2404.18466) | Freeze-half strategy |

### 8.5 Replay and Continual Learning Methods

| Paper | arXiv ID / Link | Key Contribution |
|-------|-----------------|-----------------|
| Tiwari et al. "GCR: Gradient Coreset Based Replay" (2021) | [2111.11210](https://arxiv.org/abs/2111.11210) | Gradient-based buffer selection |
| Li et al. "AdaER: Adaptive Experience Replay" (2023) | [2308.03810](https://arxiv.org/abs/2308.03810) | Entropy-balanced reservoir sampling |
| Wang et al. "Experience Replay Addresses Loss of Plasticity" (2025) | [2503.20018](https://arxiv.org/abs/2503.20018) | Replay + Transformers |
| Sarfraz et al. "SYNERgy: Synaptic Consolidation + Replay" (2022) | [2206.04016](https://arxiv.org/abs/2206.04016) | Dual-memory replay + synaptic consolidation |
| Zhang et al. "DOC: Dynamic Orthogonal Continual Fine-tuning" (2025) | [2509.23893](https://arxiv.org/abs/2509.23893) | Orthogonal gradient projection |
| Feng et al. "KIF: Knowledge Identification and Fusion" (2024) | [2408.05200](https://arxiv.org/abs/2408.05200) | Skill-unit based CL for LLMs |
| Yang et al. "Self-Distillation Fine-Tuning (SDFT)" (2024) | [HF:2402.13669](https://hf.co/papers/2402.13669) | Self-distillation bridges distribution gap |
| Lin et al. "Continual Learning via Sparse Memory Finetuning" (2025) | [HF:2510.15103](https://hf.co/papers/2510.15103) | Sparse memory for 11% forgetting |

### 8.6 GitHub Repositories

| Repository | URL | Description |
|-----------|-----|-------------|
| ContinualAI/avalanche | [github.com/ContinualAI/avalanche](https://github.com/ContinualAI/avalanche) | End-to-end continual learning library (2040 stars) |
| chrhenning/hypercl | [github.com/chrhenning/hypercl](https://github.com/chrhenning/hypercl) | Continual learning with hypernetworks (170 stars) |
| stokesj/EWC | [github.com/stokesj/EWC](https://github.com/stokesj/EWC) | TensorFlow EWC implementation (75 stars) |
| Yuxing-Wang-THU/Elastic-Weights-Consolidation | [github.com/Yuxing-Wang-THU/Elastic-Weights-Consolidation](https://github.com/Yuxing-Wang-THU/Elastic-Weights-Consolidation) | PyTorch EWC implementation (39 stars) |
| meloxxxxxx/DOC | [github.com/meloxxxxxx/DOC](https://github.com/meloxxxxxx/DOC) | Dynamic Orthogonal Continual fine-tuning |
| jeff024/codyra | [github.com/jeff024/codyra](https://github.com/jeff024/codyra) | Continual Dynamic Rank-Selective LoRA |
| c2d-usp/Layer-wise-LoRA-with-CKA | [github.com/c2d-usp/Layer-wise-LoRA-with-CKA](https://github.com/c2d-usp/Layer-wise-LoRA-with-CKA) | Layer-wise LoRA selection |
| microsoft/LoRA | [github.com/microsoft/LoRA](https://github.com/microsoft/LoRA) | Original LoRA implementation |
| huggingface/peft | [github.com/huggingface/peft](https://github.com/huggingface/peft) | HuggingFace PEFT library (LoRA, QLoRA, etc.) |

### 8.7 HuggingFace Resources

| Resource | Link | Description |
|----------|------|-------------|
| PEFT library | [huggingface.co/docs/peft](https://huggingface.co/docs/peft) | Official PEFT documentation |
| BitsAndBytes | [huggingface.co/docs/bitsandbytes](https://huggingface.co/docs/bitsandbytes) | 4-bit quantization for QLoRA |
| LoRA Land paper | [hf.co/papers/2405.00732](https://hf.co/papers/2405.00732) | 310 fine-tuned LLMs benchmark |

---

*Report generated on 2026-03-18. This report synthesizes findings from 50+ papers, multiple GitHub repositories, and HuggingFace resources to provide a comprehensive guide for mitigating catastrophic forgetting when fine-tuning 70B+ parameter language models.*
