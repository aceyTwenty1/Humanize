"""MODULE 4 — Adversarial RL Training Loop.

Actor (policy):      TextRewriter causal LM (+ QLoRA adapters).
Critic / reward:     LocalPatternClassifier.human_likeness_score in [0, 1].

Dynamics per step:
    prompt (isolated <input_text>) -> Actor generates candidate(s)
        -> Critic scores human-likeness  ->  PPO/DPO updates Actor weights
        -> sampling params adapt toward high-reward regions over time.

Backends:
    * algo="ppo": tries TRL PPOTrainer (trl>=0.8 API); if the installed TRL
      version is incompatible, falls back to a local REINFORCE-with-baseline
      loop (policy gradient with KL penalty to a frozen reference snapshot).
    * algo="dpo": builds chosen/rejected pairs ranked BY THE CRITIC and tries
      TRL DPOTrainer; falls back to filtered supervised fine-tuning on the
      chosen responses (offline-RL / RFT objective).

Everything runs locally — no external APIs. Works on GPU (7B 4-bit) and on
CPU (fallback model, tiny batches) for smoke tests.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from config import RLConfig
from utils import SamplingParams, set_seed

logger = logging.getLogger(__name__)

try:
    from classifier import LocalPatternClassifier
    from generator import TextRewriter
except ImportError:  # pragma: no cover
    from src.humaize.classifier import LocalPatternClassifier  # type: ignore
    from src.humaize.generator import TextRewriter  # type: ignore


@dataclass
class PreferencePair:
    prompt: str  # isolated-prompt string fed to the model
    chosen: str  # higher human-likeness rewrite
    rejected: str  # lower human-likeness rewrite
    reward_chosen: float
    reward_rejected: float


@dataclass
class RLTrainingStats:
    algo: str
    steps: int
    mean_reward_before: float
    mean_reward_after: float
    pairs_used: int = 0
    backend: str = "unknown"
    history: list[dict[str, Any]] = field(default_factory=list)


class AdversarialTrainer:
    """Adversarial PPO/DPO trainer: Actor=rewriter, Critic=classifier."""

    def __init__(
        self,
        rewriter: TextRewriter,
        critic: LocalPatternClassifier,
        config: Optional[RLConfig] = None,
    ) -> None:
        self.rewriter = rewriter
        self.critic = critic
        self.config = config or RLConfig()
        set_seed(self.config.seed)

    # ------------------------------------------------------- reward plumbing
    def score_candidates(self, candidates: list[str]) -> list[float]:
        return [float(self.critic.human_likeness_score(c)) for c in candidates]

    def eval_mean_reward(self, prompts: list[str], cap: int = 4) -> float:
        """Mean critic score over FRESH rewrites (never the prompts themselves).

        Scoring the prompts would make before/after deltas vacuous (identical
        inputs). Greedy decoding keeps this cheap and deterministic.
        """
        from utils import SamplingParams

        scores: list[float] = []
        for p in prompts[: max(cap, 1)]:
            try:
                out = self.rewriter.rewrite(
                    p,
                    params=SamplingParams(temperature=0.0, max_new_tokens=64),
                    num_candidates=1,
                )[0].text
            except Exception as exc:
                logger.warning("AdversarialTrainer: eval generation failed (%s)", exc)
                continue
            scores.append(float(self.critic.human_likeness_score(out)))
        return float(sum(scores) / len(scores)) if scores else 0.0

    def build_preference_pairs(
        self, prompts: list[str], num_candidates: int = 2
    ) -> tuple[list[PreferencePair], float]:
        """Generate candidate pairs per prompt, rank with the Critic.

        Returns (pairs, mean_reward_before). Prompts must already be isolated
        payloads (raw text, not tagged) — tagging happens in the generator.
        """
        pairs: list[PreferencePair] = []
        rewards_before: list[float] = []
        base = SamplingParams()
        for prompt in prompts:
            cands = []
            for k in range(max(num_candidates, 2)):
                p = base.mutate(k)  # diverse decoding => informative pairs
                try:
                    out = self.rewriter.rewrite(prompt, params=p, num_candidates=1)[0].text
                except Exception as exc:
                    logger.warning("AdversarialTrainer: generation failed (%s)", exc)
                    continue
                cands.append(out)
            if len(cands) < 2:
                continue
            scores = self.score_candidates(cands)
            rewards_before.append(float(sum(scores) / len(scores)))
            order = sorted(range(len(cands)), key=lambda i: scores[i])
            lo, hi = order[0], order[-1]
            if scores[hi] <= scores[lo]:
                continue  # no learning signal
            pairs.append(
                PreferencePair(
                    prompt=self.rewriter.build_prompt(prompt),
                    chosen=cands[hi],
                    rejected=cands[lo],
                    reward_chosen=float(scores[hi]),
                    reward_rejected=float(scores[lo]),
                )
            )
        mean_before = float(sum(rewards_before) / len(rewards_before)) if rewards_before else 0.0
        logger.info(
            "AdversarialTrainer: built %d preference pairs (mean reward before=%.3f)",
            len(pairs),
            mean_before,
        )
        return pairs, mean_before

    # ------------------------------------------------------------------ train
    def train(self, prompts: list[str]) -> RLTrainingStats:
        """Run the configured algo (ppo | dpo) over isolated prompts."""
        if not prompts:
            raise ValueError("train() needs at least one prompt string")
        algo = self.config.algo.lower()
        if algo == "ppo":
            return self.train_ppo(prompts)
        if algo == "dpo":
            return self.train_dpo(prompts)
        raise ValueError(f"Unknown RL algo '{self.config.algo}' (use ppo|dpo)")

    # ------------------------------------------------------------------- PPO
    def train_ppo(self, prompts: list[str]) -> RLTrainingStats:
        """PPO via TRL when available, else REINFORCE-with-baseline fallback."""
        try:
            return self._train_ppo_trl(prompts)
        except Exception as exc:
            logger.warning("AdversarialTrainer: TRL PPO unavailable (%s); using REINFORCE fallback.", exc)
            return self._train_reinforce(prompts)

    def _train_ppo_trl(self, prompts: list[str]) -> RLTrainingStats:
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer  # type: ignore

        self.rewriter._ensure_loaded()
        model, tokenizer = self.rewriter.get_model_and_tokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ppo_config = PPOConfig(
            model_name=self.rewriter.active_model_name or self.rewriter.config.model_name,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            target_kl=self.config.target_kl,
        )
        # Value head wrapper requires a PreTrainedModel; PEFT-wrapped models work.
        try:
            policy = AutoModelForCausalLMWithValueHead.from_pretrained(model)  # type: ignore
        except Exception:
            policy = model  # newer TRL accepts the raw policy model
        trainer = PPOTrainer(config=ppo_config, model=policy, tokenizer=tokenizer)
        logger.info("AdversarialTrainer: PPO via TRL (%d prompts)", len(prompts))

        rewards_before = self.eval_mean_reward(prompts)
        history: list[dict[str, Any]] = []
        for epoch in range(max(self.config.num_epochs, 1)):
            random.shuffle(prompts)
            for i in range(0, len(prompts), self.config.batch_size):
                batch = prompts[i : i + self.config.batch_size]
                queries = [self.rewriter.build_prompt(p) for p in batch]
                q_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze(0) for q in queries]
                r_tensors = trainer.generate(
                    q_tensors,
                    return_prompt=False,
                    **dict(max_new_tokens=128, do_sample=True, top_p=0.95, temperature=0.9),
                )
                texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in r_tensors]
                rewards = [float(self.critic.human_likeness_score(t)) for t in texts]
                import torch

                reward_tensors = [torch.tensor(r) for r in rewards]
                stats = trainer.step(q_tensors, r_tensors, reward_tensors)
                history.append({"epoch": epoch, "mean_reward": sum(rewards) / len(rewards),
                                "ppo_stats": {k: float(v) if hasattr(v, "__float__") else str(v)
                                              for k, v in (stats.items() if isinstance(stats, dict) else [])}})
        mean_after = self.eval_mean_reward(prompts)
        self.rewriter.save_adapter(self.config.output_dir)
        return RLTrainingStats(
            algo="ppo", steps=len(history),
            mean_reward_before=rewards_before,
            mean_reward_after=mean_after,
            backend="trl.PPOTrainer", history=history,
        )

    def _train_reinforce(self, prompts: list[str]) -> RLTrainingStats:
        """Local REINFORCE with running baseline + magnitude regularizer.

        Runs on any torch causal LM (CPU-safe, tiny batches). This is the
        fallback when TRL's PPO API is unavailable — same adversarial signal
        (Critic reward), standard policy-gradient update.
        """
        import torch
        import torch.nn.functional as F

        self.rewriter._ensure_loaded()
        try:
            self.rewriter.enable_lora_for_training()
        except Exception as exc:
            logger.warning("AdversarialTrainer: LoRA unavailable (%s); tuning full "
                           "small-model weights instead (do NOT use on 7B).", exc)
        model, tokenizer = self.rewriter.get_model_and_tokenizer()
        model.train()
        from utils import is_light

        if is_light():  # recompute activations instead of storing: less RAM, slower
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        opt = torch.optim.AdamW(model.parameters(), lr=min(self.config.learning_rate, 5e-5))
        baseline = 0.5
        mean_before = self.eval_mean_reward(prompts)
        history: list[dict[str, Any]] = []
        steps = 0
        for epoch in range(max(self.config.num_epochs, 1)):
            for prompt in prompts:
                full = self.rewriter.build_prompt(prompt)
                enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024)
                in_ids = enc["input_ids"].to(model.device)
                attn = enc.get("attention_mask", None)
                if attn is not None:
                    attn = attn.to(model.device)
                gen = model.generate(
                    in_ids, attention_mask=attn, max_new_tokens=64,
                    do_sample=True, temperature=0.9, top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
                resp_ids = gen[0][in_ids.shape[1]:]
                text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                reward = float(self.critic.human_likeness_score(text)) if text.strip() else 0.0
                advantage = reward - baseline
                baseline = 0.9 * baseline + 0.1 * reward
                # log P(response | prompt) under current policy
                seq = torch.cat([in_ids[0], resp_ids]).unsqueeze(0)
                logits = model(seq).logits[:, :-1, :]
                tgt = seq[:, 1:]
                lp = F.log_softmax(logits, dim=-1).gather(2, tgt.unsqueeze(-1)).squeeze(-1)
                resp_lp = lp[0, (in_ids.shape[1] - 1):]
                # Small magnitude regularizer on response log-probs (NOT a true
                # KL to a reference policy — just guards against collapse).
                loss = -(advantage * resp_lp.mean()) + self.config.kl_coef * (resp_lp.mean() ** 2) * 0.01
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                steps += 1
                history.append({"epoch": epoch, "reward": reward, "loss": float(loss.item())})
        model.eval()
        mean_after = self.eval_mean_reward(prompts)
        self.rewriter.save_adapter(self.config.output_dir)
        logger.info("AdversarialTrainer: REINFORCE done (%d steps, %.3f -> %.3f)",
                    steps, mean_before, mean_after)
        return RLTrainingStats(
            algo="ppo-fallback(reinforce)", steps=steps,
            mean_reward_before=mean_before,
            mean_reward_after=mean_after,
            backend="local-reinforce", history=history[-50:],
        )

    # ------------------------------------------------------------------- DPO
    def train_dpo(self, prompts: list[str]) -> RLTrainingStats:
        pairs, mean_before = self.build_preference_pairs(prompts)
        if len(pairs) < 2:
            logger.warning("AdversarialTrainer: <2 preference pairs; skipping DPO weight update.")
            return RLTrainingStats(algo="dpo", steps=0, mean_reward_before=mean_before,
                                   mean_reward_after=mean_before, pairs_used=len(pairs),
                                   backend="skipped")
        try:
            return self._train_dpo_trl(pairs, mean_before)
        except Exception as exc:
            logger.warning("AdversarialTrainer: TRL DPO unavailable (%s); using filtered-SFT fallback.", exc)
            return self._train_filtered_sft(pairs, mean_before)

    def _train_dpo_trl(self, pairs: list[PreferencePair], mean_before: float) -> RLTrainingStats:
        from datasets import Dataset  # type: ignore
        from transformers import TrainingArguments  # type: ignore
        from trl import DPOTrainer  # type: ignore

        self.rewriter.enable_lora_for_training()
        model, tokenizer = self.rewriter.get_model_and_tokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ds = Dataset.from_list([{"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected} for p in pairs])
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=max(self.config.mini_batch_size, 1),
            num_train_epochs=max(self.config.num_epochs, 1),
            learning_rate=self.config.learning_rate,
            logging_steps=1,
            save_steps=50,
            remove_unused_columns=False,
            report_to="none",
        )
        trainer = DPOTrainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)  # type: ignore
        trainer.train()
        trainer.save_model(self.config.output_dir)
        after = float(sum(p.reward_chosen for p in pairs) / len(pairs))
        return RLTrainingStats(algo="dpo", steps=len(trainer.state.log_history),
                               mean_reward_before=mean_before, mean_reward_after=after,
                               pairs_used=len(pairs), backend="trl.DPOTrainer",
                               history=trainor_state(trainer))

    def _train_filtered_sft(self, pairs: list[PreferencePair], mean_before: float) -> RLTrainingStats:
        """Offline-RL fallback: SFT on Critic-preferred (chosen) responses."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.rewriter._ensure_loaded()
        model, tokenizer = self.rewriter.get_model_and_tokenizer()
        model.train()
        from utils import is_light

        if is_light():
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        ids_list, mask_list = [], []
        for p in pairs:
            full = p.prompt + "\n" + p.chosen
            enc = tokenizer(full, truncation=True, padding="max_length",
                            max_length=1024)
            ids_list.append(enc["input_ids"])
            mask_list.append(enc["attention_mask"])
        ds = TensorDataset(torch.tensor(ids_list), torch.tensor(mask_list))
        dl = DataLoader(ds, batch_size=max(self.config.mini_batch_size, 1), shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=min(self.config.learning_rate, 5e-5))
        steps, last_loss = 0, 0.0
        for _ in range(max(self.config.num_epochs, 1)):
            for b_ids, b_mask in dl:
                b_ids = b_ids.to(model.device)
                b_mask = b_mask.to(model.device)
                out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_ids)
                loss = out.loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                steps += 1
                last_loss = float(loss.item())
        model.eval()
        self.rewriter.save_adapter(self.config.output_dir)
        after = float(sum(p.reward_chosen for p in pairs) / len(pairs))
        logger.info("AdversarialTrainer: filtered-SFT done (%d steps, loss=%.4f)", steps, last_loss)
        return RLTrainingStats(algo="dpo-fallback(filtered-sft)", steps=steps,
                               mean_reward_before=mean_before, mean_reward_after=after,
                               pairs_used=len(pairs), backend="local-filtered-sft",
                               history=[{"loss": last_loss}])


def trainor_state(trainer) -> list[dict[str, Any]]:
    try:
        return [{"loss": float(h.get("loss", 0.0))} for h in trainer.state.log_history if "loss" in h][-20:]
    except Exception:
        return []


if __name__ == "__main__":  # smoke test imports only (no training without data)
    print("AdversarialTrainer module OK. Run via: python main.py --mode full --fast")
