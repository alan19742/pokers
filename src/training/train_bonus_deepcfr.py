# src/training/train_bonus_deepcfr.py
"""
Deep CFR training for Texas Hold'em Bonus (BonusState environment).

Adapted from the NLHE Deep CFR reference. Key differences:
  * Single-player vs dealer  -> no opponent agent, dealer is a chance node.
  * 4 discrete actions       -> no bet-size head in the network.
  * External-sampling MCCFR  -> dealer hand and future board sampled once per
                                rollout (via BonusState.from_seed), player
                                actions enumerated at every decision node.
"""

import os
import time
import random
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pokers as pkrs


# ---------------------------------------------------------------------------
# Action / encoding constants
# ---------------------------------------------------------------------------
NUM_ACTIONS = 4  # Fold, Play, Check, Bet

ACTION_INDEX_TO_ENUM = {
    0: pkrs.BonusActionEnum.Fold,
    1: pkrs.BonusActionEnum.Play,
    2: pkrs.BonusActionEnum.Check,
    3: pkrs.BonusActionEnum.Bet,
}


ACTION_ENUM_TO_INDEX = {v: k for k, v in ACTION_INDEX_TO_ENUM.items()}


def action_enum_to_index(action):
    """Convert a BonusActionEnum to its integer index."""
    return ACTION_ENUM_TO_INDEX[action]

# State encoding:
#   stage one-hot         : 5
#   player hand (2 cards) : 2 * (4 + 13) = 34
#   public cards (<=5)    : 5 * (1 + 4 + 13) = 90
#   bet/stake features    : 6
#   legal action mask     : 4
STATE_DIM = 5 + 34 + 90 + 6 + 4


def encode_card(card, dealt=True, with_flag=False):
    """Encode a single card into a fixed-size vector."""
    suit = np.zeros(4, dtype=np.float32)
    rank = np.zeros(13, dtype=np.float32)
    if dealt and card is not None:
        suit[int(card.suit)] = 1.0
        rank[int(card.rank)] = 1.0
    if with_flag:
        return np.concatenate([[1.0 if dealt else 0.0], suit, rank])
    return np.concatenate([suit, rank])


def encode_bonus_state(state, initial_stake):
    """Encode a BonusState into a fixed-size float32 vector."""
    # Stage one-hot
    stage_oh = np.zeros(5, dtype=np.float32)
    stage_oh[int(state.stage)] = 1.0

    # Player hand
    p1, p2 = state.player_hand
    hand = np.concatenate([encode_card(p1), encode_card(p2)])

    # Public cards (pad to 5)
    public_parts = []
    for i in range(5):
        if i < len(state.public_cards):
            public_parts.append(encode_card(state.public_cards[i], dealt=True, with_flag=True))
        else:
            public_parts.append(np.zeros(18, dtype=np.float32))
    public = np.concatenate(public_parts)

    # Bet / stake features (normalized)
    denom = max(initial_stake, 1.0)
    bets = np.array([
        state.ante / denom,
        state.flop_bet / denom,
        state.turn_bet / denom,
        state.river_bet / denom,
        state.bonus_bet / denom,
        state.stake / denom,
    ], dtype=np.float32)

    # Legal action mask
    legal = set(state.legal_actions)
    mask = np.array(
        [ACTION_INDEX_TO_ENUM[i] in legal for i in range(NUM_ACTIONS)],
        dtype=np.float32,
    )

    return np.concatenate([stage_oh, hand, public, bets, mask])


def get_legal_action_indices(state):
    legal = state.legal_actions
    return [action_enum_to_index(a) for a in legal]


# ---------------------------------------------------------------------------
# Reservoir buffer (uniform sample over the stream)
# ---------------------------------------------------------------------------
class ReservoirBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0

    def add(self, item):
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, batch_size):
        size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class AdvantageNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, num_actions=NUM_ACTIONS, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, num_actions)

    def forward(self, x):
        return self.head(self.body(x))


class StrategyNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, num_actions=NUM_ACTIONS, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, num_actions)

    def forward(self, x):
        # Returns logits; softmax is applied externally with masking
        return self.head(self.body(x))


# ---------------------------------------------------------------------------
# Random baseline agent
# ---------------------------------------------------------------------------
class RandomBonusAgent:
    """Uniform random over legal actions, used as evaluation baseline."""

    def choose_action(self, state):
        legal = state.legal_actions
        if not legal:
            return None
        return random.choice(legal)


# ---------------------------------------------------------------------------
# Deep CFR agent
# ---------------------------------------------------------------------------
class BonusDeepCFRAgent:
    def __init__(self,
                 device="cpu",
                 advantage_capacity=400_000,
                 strategy_capacity=400_000,
                 lr=1e-4,
                 hidden=256):
        self.device = device
        self.num_actions = NUM_ACTIONS
        self.iteration_count = 0

        self.advantage_net = AdvantageNet(hidden=hidden).to(device)
        self.strategy_net = StrategyNet(hidden=hidden).to(device)

        self.advantage_optimizer = optim.Adam(self.advantage_net.parameters(), lr=lr)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=lr)

        self.advantage_memory = ReservoirBuffer(advantage_capacity)
        self.strategy_memory = ReservoirBuffer(strategy_capacity)

    # ---- Action selection (used during evaluation) ------------------------
    def choose_action(self, state, initial_stake=1000.0, deterministic=False):
        legal_idx = get_legal_action_indices(state)
        if not legal_idx:
            return None

        encoded = encode_bonus_state(state, initial_stake)
        x = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.strategy_net(x)[0].cpu().numpy()

        # Softmax with legal mask
        masked = np.full(self.num_actions, -1e9, dtype=np.float32)
        for i in legal_idx:
            masked[i] = logits[i]
        exp = np.exp(masked - masked.max())
        probs = exp / exp.sum()

        if deterministic:
            chosen_idx = int(np.argmax(probs))
        else:
            chosen_idx = int(np.random.choice(self.num_actions, p=probs))
        return ACTION_INDEX_TO_ENUM[chosen_idx]

    # ---- CFR traversal ----------------------------------------------------
    def cfr_traverse(self, state, iteration, initial_stake, depth=0, max_depth=64):
        """External-sampling MCCFR traversal.

        Dealer hand and full board are predetermined by the seed used to create
        `state`, so we treat chance as already sampled. At each player decision
        node we recursively evaluate every legal action and accumulate regrets.
        """
        if depth > max_depth:
            return 0.0

        if state.final_state:
            return float(state.reward)

        legal_idx = get_legal_action_indices(state)
        if not legal_idx:
            return float(state.reward)

        encoded = encode_bonus_state(state, initial_stake)
        x = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            advantages = self.advantage_net(x)[0].cpu().numpy()

        # Regret-matching strategy over legal actions
        masked = np.zeros(self.num_actions, dtype=np.float32)
        for i in legal_idx:
            masked[i] = max(advantages[i], 0.0)

        if masked.sum() > 0:
            strategy = masked / masked.sum()
        else:
            strategy = np.zeros(self.num_actions, dtype=np.float32)
            for i in legal_idx:
                strategy[i] = 1.0 / len(legal_idx)

        # Evaluate every legal action
        action_values = np.zeros(self.num_actions, dtype=np.float32)
        for i in legal_idx:
            try:
                new_state = state.apply_action(ACTION_INDEX_TO_ENUM[i])
                if new_state.status != pkrs.BonusStatus.Ok:
                    action_values[i] = 0.0
                    continue
                action_values[i] = self.cfr_traverse(
                    new_state, iteration, initial_stake, depth + 1, max_depth
                )
            except Exception:
                action_values[i] = 0.0

        # Expected value under current strategy
        ev = sum(strategy[i] * action_values[i] for i in legal_idx)

        # Store regrets in advantage memory (linear-CFR weighting)
        max_abs = max(abs(action_values.max()), abs(action_values.min()), 1.0)
        scale = float(np.sqrt(max(iteration, 1)))
        for i in legal_idx:
            regret = (action_values[i] - ev) / max_abs
            regret = float(np.clip(regret, -10.0, 10.0)) * scale
            self.advantage_memory.add((encoded, i, regret))

        # Store strategy snapshot (linear weight = iteration)
        full_strategy = np.zeros(self.num_actions, dtype=np.float32)
        for i in legal_idx:
            full_strategy[i] = strategy[i]
        self.strategy_memory.add((encoded, full_strategy, iteration))

        return ev

    # ---- Network training -------------------------------------------------
    def train_advantage_network(self, batch_size=2048, epochs=4):
        if len(self.advantage_memory) < batch_size:
            return 0.0
        self.advantage_net.train()
        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            batch = self.advantage_memory.sample(batch_size)
            states = np.stack([b[0] for b in batch])
            actions = np.array([b[1] for b in batch], dtype=np.int64)
            regrets = np.array([b[2] for b in batch], dtype=np.float32)

            x = torch.from_numpy(states).to(self.device)
            a = torch.from_numpy(actions).to(self.device)
            y = torch.from_numpy(regrets).to(self.device)

            preds = self.advantage_net(x)
            pred_a = preds.gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(pred_a, y)

            self.advantage_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), 5.0)
            self.advantage_optimizer.step()

            total_loss += loss.item()
            steps += 1
        self.advantage_net.eval()
        return total_loss / max(steps, 1)

    def train_strategy_network(self, batch_size=2048, epochs=4):
        if len(self.strategy_memory) < batch_size:
            return 0.0
        self.strategy_net.train()
        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            batch = self.strategy_memory.sample(batch_size)
            states = np.stack([b[0] for b in batch])
            targets = np.stack([b[1] for b in batch])
            weights = np.array([b[2] for b in batch], dtype=np.float32)

            x = torch.from_numpy(states).to(self.device)
            t = torch.from_numpy(targets).to(self.device)
            w = torch.from_numpy(weights).to(self.device)
            w = w / (w.mean() + 1e-8)

            logits = self.strategy_net(x)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Cross-entropy weighted by iteration count
            ce = -(t * log_probs).sum(dim=-1)
            loss = (w * ce).mean()

            self.strategy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), 5.0)
            self.strategy_optimizer.step()

            total_loss += loss.item()
            steps += 1
        self.strategy_net.eval()
        return total_loss / max(steps, 1)

    # ---- Persistence ------------------------------------------------------
    def save_model(self, path):
        torch.save({
            "iteration": self.iteration_count,
            "advantage_net": self.advantage_net.state_dict(),
            "strategy_net": self.strategy_net.state_dict(),
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.advantage_net.load_state_dict(ckpt["advantage_net"])
        self.strategy_net.load_state_dict(ckpt["strategy_net"])
        self.iteration_count = ckpt.get("iteration", 0)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def run_episode(agent, ante, bonus_bet, stake, seed, deterministic=True):
    """Play one Bonus hand to completion and return final reward."""
    state = pkrs.BonusState.from_seed(
        ante=ante, bonus_bet=bonus_bet, stake=stake, seed=seed
    )
    while not state.final_state:
        if isinstance(agent, RandomBonusAgent):
            action = agent.choose_action(state)
        else:
            action = agent.choose_action(state, initial_stake=stake,
                                         deterministic=deterministic)
        if action is None:
            break
        new_state = state.apply_action(action)
        if new_state.status != pkrs.BonusStatus.Ok:
            break
        state = new_state
    return float(state.reward) if state.final_state else 0.0


def evaluate(agent, num_games=2000, ante=10.0, bonus_bet=1.0,
             stake=1000.0, seed_offset=10_000, deterministic=True):
    rewards = []
    for g in range(num_games):
        rewards.append(run_episode(
            agent, ante, bonus_bet, stake, seed_offset + g, deterministic
        ))
    if not rewards:
        return 0.0, 0.0, 0.0
    arr = np.asarray(rewards, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    win_rate = float((arr > 0).mean())
    return mean, std, win_rate


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_deep_cfr_bonus(num_iterations=1000,
                         traversals_per_iteration=200,
                         ante=10.0,
                         bonus_bet=1.0,
                         stake=1000.0,
                         save_dir="models_bonus",
                         log_dir="logs/deepcfr_bonus",
                         eval_games=1000,
                         strategy_train_every=10,
                         checkpoint_frequency=100,
                         lr=1e-4,
                         verbose=False):
    """Train a Deep CFR agent on the Texas Hold'em Bonus environment."""
    from torch.utils.tensorboard import SummaryWriter

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    agent = BonusDeepCFRAgent(device=device, lr=lr)
    losses = []
    profits = []

    # Initial evaluation
    print("Initial evaluation (random init policy)...")
    mean, std, wr = evaluate(agent, num_games=eval_games, ante=ante,
                             bonus_bet=bonus_bet, stake=stake)
    profits.append(mean)
    print(f"  mean={mean:.3f}  std={std:.3f}  win_rate={wr:.3%}")
    writer.add_scalar("Performance/MeanReward", mean, 0)
    writer.add_scalar("Performance/WinRate", wr, 0)

    rand_mean, rand_std, rand_wr = evaluate(
        RandomBonusAgent(), num_games=eval_games,
        ante=ante, bonus_bet=bonus_bet, stake=stake,
    )
    print(f"Random baseline: mean={rand_mean:.3f}  std={rand_std:.3f}  win_rate={rand_wr:.3%}")
    writer.add_scalar("Performance/RandomBaseline", rand_mean, 0)

    for iteration in range(1, num_iterations + 1):
        agent.iteration_count = iteration
        start = time.time()
        if verbose:
            print(f"Iteration {iteration}/{num_iterations}")

        # Collect data via CFR traversals
        for _ in range(traversals_per_iteration):
            seed = random.randint(0, 1_000_000)
            state = pkrs.BonusState.from_seed(
                ante=ante, bonus_bet=bonus_bet, stake=stake, seed=seed
            )
            agent.cfr_traverse(state, iteration, stake)

        traverse_time = time.time() - start
        writer.add_scalar("Time/Traversal", traverse_time, iteration)

        # Train advantage network
        adv_loss = agent.train_advantage_network()
        losses.append(adv_loss)
        writer.add_scalar("Loss/Advantage", adv_loss, iteration)
        writer.add_scalar("Memory/Advantage", len(agent.advantage_memory), iteration)

        # Periodically train strategy net + evaluate
        if iteration % strategy_train_every == 0 or iteration == num_iterations:
            strat_loss = agent.train_strategy_network()
            writer.add_scalar("Loss/Strategy", strat_loss, iteration)
            writer.add_scalar("Memory/Strategy", len(agent.strategy_memory), iteration)

            mean, std, wr = evaluate(agent, num_games=eval_games, ante=ante,
                                     bonus_bet=bonus_bet, stake=stake)
            profits.append(mean)
            writer.add_scalar("Performance/MeanReward", mean, iteration)
            writer.add_scalar("Performance/WinRate", wr, iteration)

            print(f"[iter {iteration:5d}] adv_loss={adv_loss:.4f}  "
                  f"strat_loss={strat_loss:.4f}  "
                  f"mean_reward={mean:+.3f}  win_rate={wr:.3%}  "
                  f"adv_mem={len(agent.advantage_memory)}")

        # Checkpoint
        if iteration % checkpoint_frequency == 0:
            ckpt_path = os.path.join(save_dir, f"bonus_deepcfr_iter_{iteration}.pt")
            torch.save({
                "iteration": iteration,
                "advantage_net": agent.advantage_net.state_dict(),
                "strategy_net": agent.strategy_net.state_dict(),
                "losses": losses,
                "profits": profits,
            }, ckpt_path)
            if verbose:
                print(f"  checkpoint saved: {ckpt_path}")

        elapsed = time.time() - start
        writer.add_scalar("Time/Iteration", elapsed, iteration)
        writer.flush()

    # Final evaluation
    print("Final evaluation...")
    mean, std, wr = evaluate(agent, num_games=max(eval_games * 2, 5000),
                             ante=ante, bonus_bet=bonus_bet, stake=stake)
    print(f"FINAL  mean={mean:+.4f}  std={std:.4f}  win_rate={wr:.3%}")
    writer.add_scalar("Performance/FinalMeanReward", mean, 0)
    writer.add_scalar("Performance/FinalWinRate", wr, 0)
    writer.close()

    final_path = os.path.join(save_dir, "bonus_deepcfr_final.pt")
    agent.save_model(final_path)
    print(f"Final model saved: {final_path}")

    return agent, losses, profits


def continue_training_bonus(checkpoint_path,
                            additional_iterations=1000,
                            traversals_per_iteration=200,
                            ante=10.0,
                            bonus_bet=1.0,
                            stake=1000.0,
                            save_dir="models_bonus",
                            log_dir="logs/deepcfr_bonus_continued",
                            eval_games=1000,
                            verbose=False):
    """Resume Deep CFR training from a saved checkpoint."""
    from torch.utils.tensorboard import SummaryWriter

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = BonusDeepCFRAgent(device=device)
    agent.load_model(checkpoint_path)
    print(f"Loaded checkpoint from iteration {agent.iteration_count}: {checkpoint_path}")

    start_iter = agent.iteration_count + 1
    losses, profits = [], []

    mean, _, wr = evaluate(agent, num_games=eval_games, ante=ante,
                           bonus_bet=bonus_bet, stake=stake)
    print(f"Initial mean_reward={mean:+.3f}  win_rate={wr:.3%}")
    writer.add_scalar("Performance/MeanReward", mean, start_iter - 1)

    for iteration in range(start_iter, start_iter + additional_iterations):
        agent.iteration_count = iteration
        start = time.time()

        for _ in range(traversals_per_iteration):
            seed = random.randint(0, 1_000_000)
            state = pkrs.BonusState.from_seed(
                ante=ante, bonus_bet=bonus_bet, stake=stake, seed=seed
            )
            agent.cfr_traverse(state, iteration, stake)

        adv_loss = agent.train_advantage_network()
        losses.append(adv_loss)
        writer.add_scalar("Loss/Advantage", adv_loss, iteration)

        if iteration % 10 == 0:
            strat_loss = agent.train_strategy_network()
            writer.add_scalar("Loss/Strategy", strat_loss, iteration)
            mean, _, wr = evaluate(agent, num_games=eval_games, ante=ante,
                                   bonus_bet=bonus_bet, stake=stake)
            profits.append(mean)
            writer.add_scalar("Performance/MeanReward", mean, iteration)
            writer.add_scalar("Performance/WinRate", wr, iteration)
            print(f"[iter {iteration:5d}] adv={adv_loss:.4f}  strat={strat_loss:.4f}  "
                  f"mean={mean:+.3f}  wr={wr:.3%}")

        if iteration % 100 == 0:
            ckpt_path = os.path.join(save_dir, f"bonus_deepcfr_iter_{iteration}.pt")
            agent.save_model(ckpt_path)

        writer.add_scalar("Time/Iteration", time.time() - start, iteration)
        writer.flush()

    writer.close()
    return agent, losses, profits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Deep CFR for Texas Hold'em Bonus.")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--traversals", type=int, default=200)
    p.add_argument("--ante", type=float, default=10.0)
    p.add_argument("--bonus-bet", type=float, default=1.0)
    p.add_argument("--stake", type=float, default=1000.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-games", type=int, default=1000)
    p.add_argument("--save-dir", type=str, default="models_bonus")
    p.add_argument("--log-dir", type=str, default="logs/deepcfr_bonus")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Resume from this checkpoint path.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.checkpoint:
        continue_training_bonus(
            checkpoint_path=args.checkpoint,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            ante=args.ante,
            bonus_bet=args.bonus_bet,
            stake=args.stake,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            eval_games=args.eval_games,
            verbose=args.verbose,
        )
    else:
        train_deep_cfr_bonus(
            num_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            ante=args.ante,
            bonus_bet=args.bonus_bet,
            stake=args.stake,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            eval_games=args.eval_games,
            lr=args.lr,
            verbose=args.verbose,
        )
