import json
import os
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from peft.utils.save_and_load import set_peft_model_state_dict


class DecisionTransformer(nn.Module):
    """
    Qwen3 0.6B model for verifying memory events.
    
    This model processes agent tokens and a next agent token to determine
    which agent should be executed next in the workflow.
    
    Args:
        max_seq_len (int): Maximum sequence length for positional encoding.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        trust_remote_code: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict | None = "auto",
        valid_strings: list[str] = ["YES", "NO"],
        team_idx: int = 0,
        query: str = "",
    ):
        super().__init__()
        # Load pretrained Qwen3 0.6B causal LM as the verifier backbone.
        # Some transformers versions want `dtype`, others want `torch_dtype`.
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device_map=device_map,
        )
        self.hidden_dim = getattr(base_model.config, "hidden_size", None)

        # Attach a LoRA adapter so only adapter weights are updated during training.
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.transformer = get_peft_model(base_model, lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.valid_strings = valid_strings
        # Use the first token id of each string (assumes YES/NO are single-token for the tokenizer).
        self.valid_tokens = []
        for s in valid_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if not ids:
                continue
            self.valid_tokens.append(int(ids[0]))

        backbone_param = next(base_model.parameters())
        backbone_device = backbone_param.device
        backbone_dtype = backbone_param.dtype

        self.memory_bank_proj = nn.Linear(self.hidden_dim, self.hidden_dim).to(device=backbone_device, dtype=backbone_dtype)
        self.verification_proj = nn.Linear(self.hidden_dim, self.hidden_dim).to(device=backbone_device, dtype=backbone_dtype)
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim).to(device=backbone_device, dtype=backbone_dtype)
        self.team_idx = team_idx
        self.query = self.embed_text(query, num_tokens=1).detach()
        
    
    def get_memory_bank_embedding(self, memory_bank):
        return self.memory_bank_proj(memory_bank)
    
    def get_verification_embedding(self, verification):
        return self.verification_proj(verification)

    def get_query_embedding(self, query):
        return self.query_proj(query)


    def forward(self, x, query=None, memory_bank=None):
        x = self.get_verification_embedding(x)
        # Add batch dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (1, seq_len, hidden_dim)
        
        if memory_bank is not None:
            if memory_bank.dim() == 1:
                memory_bank = memory_bank.unsqueeze(0)
            memory_bank = self.get_memory_bank_embedding(memory_bank)
            # Repeat the memory bank embedding for each batch
            memory_bank = memory_bank.repeat(x.shape[0], 1, 1)

            # Prepend the memory bank embedding to the input
            x = torch.cat([memory_bank, x], dim=1)

        if query is None:
            query = self.query

        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = self.get_query_embedding(query)
        # Repeat the query embedding for each batch
        query = query.repeat(x.shape[0], 1, 1)

        # Prepend the query embedding to the input
        x = torch.cat([query, x], dim=1)
        
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape
        
        attention_mask = torch.ones((batch_size, seq_len), device=x.device)
        
        outputs = self.transformer(
            inputs_embeds=x, 
            attention_mask=attention_mask
        )

        # import ipdb; ipdb.set_trace()
        # Take logits at the final time step: shape (batch_size, vocab_size)
        logits = outputs.logits[:, -1, :]
        
        # Restrict to the subset of valid tokens (e.g., YES / NO)
        valid_logits = self.get_valid_token_logits(logits)
        # Optionally squeeze batch dimension when batch_size == 1, for convenience
        if batch_size == 1:
            valid_logits = valid_logits.squeeze(0)  # (num_valid_tokens,)
        return valid_logits

    
    def get_valid_token_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Slice the full-vocabulary logits down to just the entries for the
        valid tokens (e.g., YES / NO).

        Args:
            logits: Tensor of shape (vocab_size,) or (batch_size, vocab_size).

        Returns:
            Tensor of shape (num_valid_tokens,) or (batch_size, num_valid_tokens),
            preserving any leading batch dimension.
        """
        index = torch.tensor(self.valid_tokens, device=logits.device)
        return logits.index_select(-1, index)


    def get_decision(self, logits: torch.Tensor) -> int:
        """
        Get the decision from the logits.
        """
        return int(torch.argmax(logits, dim=-1).item())

    def sample_decision(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        return_log_prob: bool = False,
    ) -> int | tuple[int, torch.Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")

        if logits.dim() != 1:
            raise ValueError("sample_decision expects 1D logits over actions")

        probs = torch.softmax(logits / float(temperature), dim=-1)
        if epsilon:
            probs = probs * (1.0 - float(epsilon)) + float(epsilon) / probs.numel()

        dist = torch.distributions.Categorical(probs=probs)
        action = int(dist.sample().item())
        if return_log_prob:
            return action, dist.log_prob(torch.tensor(action, device=logits.device))
        return action


    def run_decision_prediction_grad(self, gradient_input: dict) -> tuple[float, dict[str, float]]:
        # Extract task query and other inputs
        query_text = gradient_input['query_text']
        base_advantage = gradient_input['advantage']
        decisions = gradient_input['decisions']
        trace_length = gradient_input['trace_length']
        memory_bank_embeddings = gradient_input['memory_bank_embeddings']
        used_indices = gradient_input['memory_usage']
        decision_embeddings = gradient_input['decision_embeddings']
        decision_logits = gradient_input['decision_logits']

        USAGE_BONUS = 0.5 
        STORAGE_COST = 0.0
        SPARSITY_PENALTY = 0.1  # Penalize high P(accept/YES) to avoid degenerate "accept everything"
        CLIP_EPS = 0.2  # PPO/GRPO standard clipping range

        used_set: set[int] = set()
        if isinstance(used_indices, list) and len(used_indices) == len(decisions):
            used_set = {i for i, c in enumerate(used_indices) if isinstance(c, (int, float)) and c > 0}
        elif isinstance(used_indices, (list, set)):
            used_set = {i for i in used_indices if isinstance(i, int)}

        # Prepare embeddings for the task
        param = next(self.parameters())
        param_device = param.device
        param_dtype = param.dtype
        query_embedding = self.embed_text(query_text, num_tokens=1).detach().to(device=param_device, dtype=param_dtype)

        total_loss = 0.0
        step_advantages: list[float] = []
        step_advantages_nonzero: list[float] = []
        usage_bonus_count = 0
        # storage_cost_count = 0
        total_steps = max(0, trace_length - 1)
        step_losses: list[float] = []
        sparsity_losses: list[float] = []
        policy_losses: list[float] = []

        # Process each step in the trace
        for step_idx in range(trace_length - 1):  # -1 because the last step is decision maker
            # --- 1. Calculate Stepwise Advantage ---
            step_advantage = base_advantage
            # Apply usage bonus
            if base_advantage > 0 and step_idx in used_set:
                step_advantage += USAGE_BONUS
                usage_bonus_count += 1
            # if decision == 0:
                # step_advantage -= STORAGE_COST
                # storage_cost_count += 1

            step_advantages.append(float(step_advantage))

            # Skip if step advantage is too small
            if abs(step_advantage) < 1e-8:
                continue
            
            step_advantages_nonzero.append(float(step_advantage))

            # --- 2. Prepare Embeddings ---
            memory_bank_embedding = memory_bank_embeddings[step_idx]
            if memory_bank_embedding is not None:
                memory_bank_embedding = memory_bank_embedding.to(device=param_device, dtype=param_dtype)

            decision_embedding = decision_embeddings[step_idx]
            if decision_embedding is not None:
                decision_embedding = decision_embedding.to(device=param_device, dtype=param_dtype)
            decision = decisions[step_idx]

            # --- 3. Get Predicted Logits and Probabilities ---
            pred_logits = self.forward(decision_embedding, query=query_embedding, memory_bank=memory_bank_embedding)
            if pred_logits.dim() == 2:
                pred_logits = pred_logits.squeeze(0)
            pred_logprob_dist = torch.nn.functional.log_softmax(pred_logits, dim=-1)
            pred_logprob = pred_logprob_dist[decision]
            # pred_accept_prob = pred_logprob_dist[0].exp()

            # old_logprob_dist = torch.log_softmax(decision_logits[step_idx], dim=0)
            # old_logprob = old_logprob_dist[decision].detach()
            # old_accept_prob = old_logprob[0].exp()

            # --- 4. GRPO / PPO Clipped Loss ---
            # Ratio = exp(new_logprob - old_logprob)
            # ratio = torch.exp(pred_logprob - old_logprob)

            # surr1 = ratio * step_advantage
            # surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * step_advantage
            
            # # PPO minimizes the negative of the objective
            # policy_loss = -torch.min(surr1, surr2)

            policy_loss = -pred_logprob * step_advantage

            # Sparsity penalty
            accept_prob = pred_logprob_dist[0].exp()
            sparsity_loss = SPARSITY_PENALTY * accept_prob

            step_loss = policy_loss + sparsity_loss
            
            # step_loss = SPARSITY_PENALTY * accept_prob - pred_logprob[decision] * step_advantage

            # # Calculate similarity scores for agent selection
            # # agent_similarity_scores = F.cosine_similarity(encoded_tokens[0:num_roles], encoded_nap.unsqueeze(0), dim=1)
            # # use inner product + standardization (mean=0, std=1)
            # agent_similarity_scores = (encoded_tokens[0:num_roles] * encoded_nap.unsqueeze(0)).sum(dim=1)
            # mean = agent_similarity_scores.mean()
            # std = agent_similarity_scores.std()
            # agent_similarity_scores = (agent_similarity_scores - mean) / (std + 1e-8)

            # # Get logprobs for the agent selection
            # agent_logprobs = F.log_softmax(self.cos_scaling * agent_similarity_scores, dim=0)
            
            # # Get the selected agent from trace
            # selected_agent_idx = agent_selections[step_idx]
            
            # # Calculate agent selection loss (negative because we want to maximize reward)
            # step_loss = -agent_logprobs[selected_agent_idx] * advantage

            step_losses.append(float(step_loss.item()))
            sparsity_losses.append(float(sparsity_loss.item()))
            policy_losses.append(float(policy_loss.item()))

            step_loss.backward()
            total_loss += float(step_loss.item())

        metrics = {
            "reward/base_advantage": float(base_advantage),
            "reward/usage_bonus_per_step": float(usage_bonus_count) / float(total_steps) if total_steps else 0.0,
            # "reward/storage_cost_per_step": float(storage_cost_count) / float(total_steps) if total_steps else 0.0,
            "reward/combined_step_advantage_mean": float(sum(step_advantages) / len(step_advantages)) if step_advantages else 0.0,
            "reward/combined_step_advantage_mean_nonzero": float(sum(step_advantages_nonzero) / len(step_advantages_nonzero)) if step_advantages_nonzero else 0.0,
            "reward/nonzero_step_fraction": float(len(step_advantages_nonzero)) / float(total_steps) if total_steps else 0.0,
            "step_losses": step_losses,
            "sparsity_losses": sparsity_losses,
            "policy_losses": policy_losses,
        }
        return total_loss, metrics


    def get_base_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_tokens: int = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Extract hidden LLM embeddings from the *base* Qwen model (before the LM head),
        with LoRA adapters applied.

        Args:
            input_ids: LongTensor of shape (seq_len,) or (batch_size, seq_len).
            attention_mask: Optional LongTensor of same shape as input_ids.
            output_all_layers: If True, return the full tuple of hidden states
                from all transformer layers. If False, return only the final
                hidden state (last layer).

        Returns:
            - If output_all_layers=False: Tensor of shape (batch_size, seq_len, hidden_dim)
              for the last layer.
            - If output_all_layers=True: Tuple[Tensor] where each element is
              (batch_size, seq_len, hidden_dim) for one layer.
        """
        # Ensure batch dimension.
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        # PeftModel exposes the underlying backbone as get_base_model().
        base_model = self.transformer.get_base_model()
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

        # Optionally reduce/expand sequence length to exactly num_tokens positions.
        if num_tokens is not None:
            batch_size, seq_len, hidden_dim = hidden_states.shape

            if num_tokens == 1:
                # Global average over the sequence -> (batch_size, 1, hidden_dim)
                hidden_states = hidden_states.mean(dim=1, keepdim=True)
            elif num_tokens > seq_len:
                # Pad by repeating the *last* embedding until we reach num_tokens.
                pad_len = num_tokens - seq_len
                last = hidden_states[:, -1:, :].expand(batch_size, pad_len, hidden_dim)
                hidden_states = torch.cat([hidden_states, last], dim=1)  # (batch_size, num_tokens, hidden_dim)
            elif num_tokens < seq_len:
                # Chunked averaging: partition the sequence into `num_tokens` contiguous
                # segments (sizes differ by at most 1) and average within each segment.
                base = seq_len // num_tokens
                rem = seq_len % num_tokens
                chunks = []
                start = 0
                for i in range(num_tokens):
                    seg_len = base + (1 if i < rem else 0)
                    end = start + seg_len
                    # Average over this segment, keep a singleton time dimension.
                    seg_mean = hidden_states[:, start:end, :].mean(dim=1, keepdim=True)
                    chunks.append(seg_mean)
                    start = end
                hidden_states = torch.cat(chunks, dim=1)  # (batch_size, num_tokens, hidden_dim)

        return hidden_states

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_text(self, text: str, max_length: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text to (input_ids, attention_mask) on the model device."""
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        return input_ids, attention_mask

    def embed_text(self, text: str, num_tokens: int = 1, max_length: int = 512) -> torch.Tensor:
        """Convenience wrapper: tokenize + return pooled base embeddings (B, num_tokens, D)."""
        input_ids, attention_mask = self.encode_text(text, max_length=max_length)
        return self.get_base_embeddings(input_ids, attention_mask=attention_mask, num_tokens=num_tokens)


    def eval(self):
        self.memory_bank_proj.eval()
        self.verification_proj.eval()
        self.query_proj.eval()
        self.transformer.eval()

        
    def train(self):
        self.memory_bank_proj.train()
        self.verification_proj.train()
        self.query_proj.train()
        self.transformer.train()

    
    def add_to_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.optimizer.add_param_group({"params": self.memory_bank_proj.parameters()})
        self.optimizer.add_param_group({"params": self.verification_proj.parameters()})
        self.optimizer.add_param_group({"params": self.query_proj.parameters()})
        # Only add trainable (LoRA) parameters from the PEFT-wrapped transformer.
        # In a PeftModel, base model weights are frozen (requires_grad=False) and
        # LoRA adapter weights have requires_grad=True.
        lora_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer.add_param_group({"params": lora_params})

    def save(self, path: str):
        """
        Save only the lightweight learnable components:
        - memory_bank_proj / verification_proj (standard weights)
        - LoRA adapter weights for the Qwen backbone (no full base model)
        """
        lora_state = get_peft_model_state_dict(self.transformer)
        model_state = {
            "memory_bank_proj": self.memory_bank_proj.state_dict(),
            "verification_proj": self.verification_proj.state_dict(),
            "query_proj": self.query_proj.state_dict(),
            "lora_adapter": lora_state,
        }
        torch.save(model_state, path)

    def load(self, path: str):
        """
        Load the projection layers and LoRA adapter weights.

        The base Qwen model is loaded in __init__; here we only restore the
        adapter weights on top of it.
        """
        model_state = torch.load(path, map_location="cpu")
        self.memory_bank_proj.load_state_dict(model_state["memory_bank_proj"], strict=True)
        self.verification_proj.load_state_dict(model_state["verification_proj"], strict=True)
        self.query_proj.load_state_dict(model_state["query_proj"], strict=True)
        lora_state = model_state.get("lora_adapter", {})
        if lora_state:
            msg = set_peft_model_state_dict(self.transformer, lora_state, adapter_name="default")
            # print(msg)
            assert msg.unexpected_keys == [], f"Unexpected keys: {msg.unexpected_keys}"
        print("model loaded from: ", path)


    # Save and load trace states for RL training
    def save_trace_state(
        self,
        step_num: int,
        team_idx: int | None = None,
        trace_dir: str | None = None,
        memory_bank: torch.Tensor | None = None,
        verification_embeddings: torch.Tensor | None = None,
        valid_logits: torch.Tensor | None = None,
        decision: int | None = None,
        decision_str: str | None = None,
    ):
        """Save a trace state snapshot (for debugging / offline training)."""
        if team_idx is None:
            team_idx = self.team_idx

        # Normalize tensors to CPU for saving
        mb = memory_bank.detach().cpu() if memory_bank is not None else None
        ve = verification_embeddings.detach().cpu() if verification_embeddings is not None else None
        vl = valid_logits.detach().cpu() if valid_logits is not None else None

        if trace_dir is None:
            trace_dir = "."
        os.makedirs(trace_dir, exist_ok=True)

        # Wall-clock timestamp (float seconds since epoch, with millisecond+ resolution)
        ts = time.time()
        ts_ms = int(ts * 1000)

        # Include timestamp in filename so callers can order traces lexicographically.
        trace_path = os.path.join(
            trace_dir,
            f"trace_state_team{team_idx}_step{step_num}-ts{ts_ms}.pt",
        )

        torch.save(
            {
                "step_num": step_num,
                "team_idx": team_idx,
                "memory_bank": mb,
                "verification_embeddings": ve,
                "valid_logits": vl,
                "valid_strings": self.valid_strings,
                "decision": decision,
                "decision_str": decision_str,
                "timestamp": ts,
            },
            trace_path,
        )

    def load_trace_state(self, trace_path: str):
        return torch.load(trace_path)


    def get_embedding_info(self):
        """
        Print information about the transformer's embedding dimensions.
        
        Returns:
            dict: A dictionary containing embedding dimension information
        """
        info = {
            "embedding_dim": getattr(self.transformer.config, "hidden_size", None),
            "model_hidden_dim": self.hidden_dim,
            "vocab_size": getattr(self.transformer.config, "vocab_size", None),
            "max_position_embeddings": getattr(self.transformer.config, "max_position_embeddings", None),
            "num_layers": getattr(self.transformer.config, "num_hidden_layers", None),
            "num_heads": getattr(self.transformer.config, "num_attention_heads", None),
        }
        
        print(f"Transformer Embedding Dimension: {info['embedding_dim']}")
        print(f"Model Hidden Dimension: {info['model_hidden_dim']}")
        print(f"Max Position Embeddings: {info['max_position_embeddings']}")
        print(f"Number of Layers: {info['num_layers']}")
        print(f"Number of Attention Heads: {info['num_heads']}")
        
        return info 


if __name__ == "__main__":
    dt = DecisionTransformer()
    dt.eval()

    # Example: build a tiny "memory bank" of prior kept summaries (1 token each).
    with torch.no_grad():
        mem1 = dt.embed_text("Looked up Mercedes Sosa discography; identified 2000–2009 studio albums.", num_tokens=1)
        mem2 = dt.embed_text("Verified final numeric answer matches Wikipedia table.", num_tokens=1)
        memory_bank = torch.cat([mem1, mem2], dim=1)  # (1, 2, D)

        # Example: embed agent input/output/summary, concatenate into decision input.
        agent_input = "Please list the studio albums and years from Wikipedia."
        agent_output = "Found Cantora (2009) and other albums; count appears to be 3."
        summary = "Counted 3 qualifying studio albums from 2000–2009."

        agent_input_emb = dt.embed_text(agent_input, num_tokens=1)   # (1, 1, D)
        agent_output_emb = dt.embed_text(agent_output, num_tokens=1) # (1, 1, D)
        summary_emb = dt.embed_text(summary, num_tokens=1)           # (1, 1, D)

        x = torch.cat([agent_input_emb, agent_output_emb, summary_emb], dim=1)  # (1, 3, D)

        valid_logits = dt(x, memory_bank=memory_bank)  # (num_valid_tokens,)
        print("valid_logits shape:", tuple(valid_logits.shape))
        print("valid_logits:", valid_logits)

        decision_idx = dt.get_decision(valid_logits)
        decision_str = dt.valid_strings[decision_idx]
        print("decision:", decision_str)

        # Save a trace state example for inspection/debugging.
        dt.save_trace_state(step_num=0, memory_bank=memory_bank, verification_embeddings=x, valid_logits=valid_logits)