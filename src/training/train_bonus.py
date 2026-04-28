# src/training/train_bonus.py
"""Deep CFR-style training for the Texas Hold'em Bonus environment (pkrs.BonusState).

Differences vs the multi-player NLHE training script:
    * Single agent vs deterministic dealer (no opponent agents, no current_player).
    * Reward is read directly from state.reward (no players_state index).
    * Action space is 4 discrete actions: Fold(0) / Play(1) / Check(2) / Bet(3).
      Bet sizes are fixed by the environment, so no bet-size head is required.
    * The CFR traversal only branches at the player's decision nodes; the dealer
      is part of the deterministic state transition handled inside apply_action.
"""

import argparse
import os
import random
import time

import numpy as np
import pokers as pkrs
import torch

from src.core.deep_cfr import DeepCFRAgent
from src.core.model import encode_state, set_verbose
from src.utils.logging import log_game_error
from src.utils.settings import STRICT_CHECKING, set_strict_checking


# --------------------------------------------------------------------------- #
# Action mapping (Solution A: integer-indexed, no enum dict keys)             #
# --------------------------------------------------------------------------- #
# `pkrs.BonusActionEnum` is not hashable in Python (PyO3 0.18 does not auto-
# generate __hash__ for pyclass enums). To avoid touching bonus.rs, we use a
# parallel list and map enum <-> index by name string. The order of ACTION_LIST
# defines the canonical action index used by the network (num_actions=4).
ACTION_LIST = [
    pkrs.BonusActionEnum.Fold,   # 0
    pkrs.BonusActionEnum.Play,   # 1
    pkrs.BonusActionEnum.Check,  # 2
    pkrs.BonusActionEnum.Bet,    # 3
]
ACTION_NAMES = ["Fold", "Play", "Check", "Bet"]
_NAME_TO_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}


def _action_name(action):
    """Extract the variant name from a BonusActionEnum value.

    Works regardless of whether `str(action)` returns "BonusActionEnum.Fold"
    or just "Fold" across pyo3 versions.
    """
    s = str(action)
    return s.rsplit(".", 1)[-1]


def action_to_idx(action):
    """Map a pkrs.BonusActionEnum to its canonical integer index."""
    return _NAME_TO_INDEX[_action_name(action)]


def index_to_bonus_action(action_idx):
    """Map an integer index back to a pkrs.BonusActionEnum."""
    return ACTION_LIST[action_idx]


def legal_action_indices(state):
    """Return the list of legal action indices for the current BonusState."""
    return [action_to_idx(a) for a in state.legal_actions]


# --------------------------------------------------------------------------- #
# Environment factory                                                         #
# --------------------------------------------------------------------------- #
def make_bonus_state(seed, ante=10.0, bonus_bet=1.0, stake=1000.0):
    """Initialize a fresh BonusState. This is the canonical env init point."""
    return pkrs.BonusState.from_seed(
        ante=ante,
        bonus_bet=bonus_bet,
        stake=stake,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def evaluate_against_dealer(agent, num_games=500,
                            ante=10.0, bonus_bet=1.0, stake=1000.0):
    """Evaluate the trained agent against the deterministic dealer.

    Returns the average reward per completed hand.
    """
    total_profit = 0.0
    completed_games = 0

    for game in range(num_games):
        try:
            state = make_bonus_state(
                seed=game,
                ante=ante,
                bonus_bet=bonus_bet,
                stake=stake,
            )

            while not state.final_state:
                action = agent.choose_action(state)

                new_state = state.apply_action(action)
                if new_state.status != pkrs.BonusStatus.Ok:
                    log_file = log_game_error(
                        state, action,
                        f"BonusState status not OK ({new_state.status})",
                    )
                    if STRICT_CHECKING:
                        raise ValueError(
                            f"BonusState status not OK ({new_state.status}). "
                            f"Details logged to {log_file}"
                        )
                    print(
                        f"WARNING: BonusState status not OK ({new_state.status}) "
                        f"in game {game}. Details logged to {log_file}"
                    )
                    break

                state = new_state

            if state.final_state:
                total_profit += state.reward
                completed_games += 1

        except Exception as exc:
            if STRICT_CHECKING:
                raise
            print(f"Error in game {game}: {exc}")

    if completed_games == 0:
        print("WARNING: No games completed during evaluation!")
        return 0.0
    return total_profit / completed_games


# --------------------------------------------------------------------------- #
# CFR-style traversal (single-agent variant)                                  #
# --------------------------------------------------------------------------- #
def _bonus_cfr_traverse(agent, state, iteration, depth=0, verbose=False):
    """Single-agent CFR-style traversal over a BonusState tree.

    Because the dealer is deterministic, every transition is either
        (a) a player decision node (we expand all legal actions), or
        (b) a terminal node (we read state.reward).
    No opponent sampling is required.
    """
    max_depth = 64  # Bonus has at most 4 player decisions, so this is generous.
    if depth > max_depth:
        if verbose:
            print(f"WARNING: Max recursion depth reached ({max_depth}).")
        return 0.0

    if state.final_state:
        return state.reward

    legal = legal_action_indices(state)
    if not legal:
        # Stages with no legal actions (e.g. Showdown before settlement) just
        # propagate the reward of the terminal state.
        return state.reward

    # --------------------------------------------------------------- #
    # Forward pass: predict advantages for current state              #
    # --------------------------------------------------------------- #
    encoded_state = encode_state(state)
    state_tensor = torch.FloatTensor(encoded_state).to(agent.device)

    with torch.no_grad():
        advantages = agent.advantage_net(state_tensor.unsqueeze(0))
        # Some agent variants return (advantages, bet_size); we ignore bet size
        # because BonusState bet sizes are fixed.
        if isinstance(advantages, tuple):
            advantages = advantages[0]
        advantages = advantages[0].cpu().numpy()

    advantages_masked = np.zeros(agent.num_actions)
    for action_idx in legal:
        advantages_masked[action_idx] = max(advantages[action_idx], 0.0)

    if advantages_masked.sum() > 0:
        strategy = advantages_masked / advantages_masked.sum()
    else:
        strategy = np.zeros(agent.num_actions)
        for action_idx in legal:
            strategy[action_idx] = 1.0 / len(legal)

    # --------------------------------------------------------------- #
    # Recurse into each legal action                                  #
    # --------------------------------------------------------------- #
    action_values = np.zeros(agent.num_actions)
    for action_idx in legal:
        try:
            bonus_action = index_to_bonus_action(action_idx)
            new_state = state.apply_action(bonus_action)

            if new_state.status != pkrs.BonusStatus.Ok:
                log_file = log_game_error(
                    state, bonus_action,
                    f"BonusState status not OK ({new_state.status})",
                )
                if STRICT_CHECKING:
                    raise ValueError(
                        f"BonusState status not OK ({new_state.status}) "
                        f"during CFR traversal. Details logged to {log_file}"
                    )
                if verbose:
                    print(
                        f"WARNING: Invalid action {action_idx} at depth {depth}. "
                        f"Status: {new_state.status}. Details: {log_file}"
                    )
                continue

            action_values[action_idx] = _bonus_cfr_traverse(
                agent, new_state, iteration, depth + 1, verbose
            )
        except Exception as exc:
            if verbose:
                print(f"ERROR in traversal for action {action_idx}: {exc}")
            action_values[action_idx] = 0.0
            if STRICT_CHECKING:
                raise

    # --------------------------------------------------------------- #
    # Compute regrets and store samples                               #
    # --------------------------------------------------------------- #
    ev = sum(strategy[a] * action_values[a] for a in legal)
    max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)

    # The Bonus environment has no opponent, so opponent_features is zero-padded
    # for compatibility with the existing memory schema.
    opponent_features = np.zeros(20)

    for action_idx in legal:
        regret = action_values[action_idx] - ev
        normalized_regret = regret / max_abs_val
        clipped_regret = np.clip(normalized_regret, -10.0, 10.0)
        scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0
        weighted_regret = clipped_regret * scale_factor
        priority = abs(weighted_regret) + 0.01

        agent.advantage_memory.add(
            (
                encoded_state,
                opponent_features,
                action_idx,
                0.0,                # bet_size_multiplier (unused for Bonus)
                weighted_regret,
            ),
            priority,
        )

    strategy_full = np.zeros(agent.num_actions)
    for action_idx in legal:
        strategy_full[action_idx] = strategy[action_idx]

    agent.strategy_memory.append(
        (
            encoded_state,
            opponent_features,
            strategy_full,
            0.0,                    # bet_size_multiplier (unused for Bonus)
            iteration,
        )
    )

    return ev


# --------------------------------------------------------------------------- #
# Training loops                                                              #
# --------------------------------------------------------------------------- #
def train_deep_cfr_bonus(
    num_iterations=1000,
    traversals_per_iteration=200,
    ante=10.0,
    bonus_bet=1.0,
    stake=1000.0,
    save_dir="models_bonus",
    log_dir="logs/deepcfr_bonus",
    verbose=False,
):
    """Train a Deep CFR agent on the Texas Hold'em Bonus environment."""
    from torch.utils.tensorboard import SummaryWriter

    set_verbose(verbose)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # The agent is expected to be configured with num_actions=4. For BonusState
    # there is no notion of player_id or num_players; we pass placeholders.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DeepCFRAgent(player_id=0, num_players=1, device=device)

    losses = []
    profits = []

    print("Initial evaluation...")
    initial_profit = evaluate_against_dealer(
        agent,
        num_games=500,
        ante=ante,
        bonus_bet=bonus_bet,
        stake=stake,
    )
    profits.append(initial_profit)
    print(f"Initial average profit per hand: {initial_profit:.4f}")
    writer.add_scalar("Performance/Profit", initial_profit, 0)

    checkpoint_frequency = 100

    for iteration in range(1, num_iterations + 1):
        agent.iteration_count = iteration
        start_time = time.time()
        print(f"Iteration {iteration}/{num_iterations}")

        # --------------- traversals --------------- #
        print("  Collecting data...")
        for _ in range(traversals_per_iteration):
            state = make_bonus_state(
                seed=random.randint(0, 1_000_000),
                ante=ante,
                bonus_bet=bonus_bet,
                stake=stake,
            )
            _bonus_cfr_traverse(agent, state, iteration, verbose=verbose)

        traversal_time = time.time() - start_time
        writer.add_scalar("Time/Traversal", traversal_time, iteration)

        # --------------- advantage net --------------- #
        print("  Training advantage network...")
        adv_loss = agent.train_advantage_network()
        losses.append(adv_loss)
        print(f"  Advantage network loss: {adv_loss:.6f}")
        writer.add_scalar("Loss/Advantage", adv_loss, iteration)
        writer.add_scalar("Memory/Advantage", len(agent.advantage_memory), iteration)

        # --------------- strategy net + eval --------------- #
        if iteration % 10 == 0 or iteration == num_iterations:
            print("  Training strategy network...")
            strat_loss = agent.train_strategy_network()
            print(f"  Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar("Loss/Strategy", strat_loss, iteration)

            print("  Evaluating agent...")
            avg_profit = evaluate_against_dealer(
                agent,
                num_games=500,
                ante=ante,
                bonus_bet=bonus_bet,
                stake=stake,
            )
            profits.append(avg_profit)
            print(f"  Average profit per hand: {avg_profit:.4f}")
            writer.add_scalar("Performance/Profit", avg_profit, iteration)

        # --------------- checkpointing --------------- #
        if iteration % checkpoint_frequency == 0:
            checkpoint_path = f"{save_dir}/bonus_checkpoint_iter_{iteration}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "advantage_net": agent.advantage_net.state_dict(),
                    "strategy_net": agent.strategy_net.state_dict(),
                    "losses": losses,
                    "profits": profits,
                    "ante": ante,
                    "bonus_bet": bonus_bet,
                    "stake": stake,
                },
                checkpoint_path,
            )
            print(f"  Checkpoint saved to {checkpoint_path}")

        elapsed = time.time() - start_time
        writer.add_scalar("Time/Iteration", elapsed, iteration)
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print(f"  Advantage memory size: {len(agent.advantage_memory)}")
        print(f"  Strategy memory size: {len(agent.strategy_memory)}")
        writer.add_scalar("Memory/Strategy", len(agent.strategy_memory), iteration)
        writer.flush()
        print()

    print("Final evaluation...")
    final_profit = evaluate_against_dealer(
        agent,
        num_games=2000,
        ante=ante,
        bonus_bet=bonus_bet,
        stake=stake,
    )
    print(f"Final performance: average profit per hand: {final_profit:.4f}")
    writer.add_scalar("Performance/FinalProfit", final_profit, 0)
    writer.close()

    return agent, losses, profits


def continue_training_bonus(
    checkpoint_path,
    additional_iterations=1000,
    traversals_per_iteration=200,
    save_dir="models_bonus",
    log_dir="logs/deepcfr_bonus_continued",
    verbose=False,
):
    """Resume Deep CFR training on BonusState from a saved checkpoint."""
    from torch.utils.tensorboard import SummaryWriter

    set_verbose(verbose)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    ante = checkpoint.get("ante", 10.0)
    bonus_bet = checkpoint.get("bonus_bet", 1.0)
    stake = checkpoint.get("stake", 1000.0)

    agent = DeepCFRAgent(player_id=0, num_players=1, device=device)
    agent.advantage_net.load_state_dict(checkpoint["advantage_net"])
    agent.strategy_net.load_state_dict(checkpoint["strategy_net"])

    start_iteration = checkpoint["iteration"] + 1
    agent.iteration_count = start_iteration - 1

    losses = checkpoint.get("losses", [])
    profits = checkpoint.get("profits", [])

    print(f"Loaded model from iteration {start_iteration - 1}")
    print(f"Continuing training for {additional_iterations} more iterations")

    print("Initial evaluation of loaded model...")
    initial_profit = evaluate_against_dealer(
        agent,
        num_games=500,
        ante=ante,
        bonus_bet=bonus_bet,
        stake=stake,
    )
    if not profits:
        profits.append(initial_profit)
    print(f"Initial average profit per hand: {initial_profit:.4f}")
    writer.add_scalar("Performance/Profit", initial_profit, start_iteration - 1)

    checkpoint_frequency = 100

    for iteration in range(
        start_iteration, start_iteration + additional_iterations
    ):
        agent.iteration_count = iteration
        start_time = time.time()
        print(
            f"Iteration {iteration}/"
            f"{start_iteration + additional_iterations - 1}"
        )

        print("  Collecting data...")
        for _ in range(traversals_per_iteration):
            state = make_bonus_state(
                seed=random.randint(0, 1_000_000),
                ante=ante,
                bonus_bet=bonus_bet,
                stake=stake,
            )
            _bonus_cfr_traverse(agent, state, iteration, verbose=verbose)

        traversal_time = time.time() - start_time
        writer.add_scalar("Time/Traversal", traversal_time, iteration)

        print("  Training advantage network...")
        adv_loss = agent.train_advantage_network()
        losses.append(adv_loss)
        print(f"  Advantage network loss: {adv_loss:.6f}")
        writer.add_scalar("Loss/Advantage", adv_loss, iteration)
        writer.add_scalar("Memory/Advantage", len(agent.advantage_memory), iteration)

        if (
            iteration % 10 == 0
            or iteration == start_iteration + additional_iterations - 1
        ):
            print("  Training strategy network...")
            strat_loss = agent.train_strategy_network()
            print(f"  Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar("Loss/Strategy", strat_loss, iteration)

            print("  Evaluating agent...")
            avg_profit = evaluate_against_dealer(
                agent,
                num_games=500,
                ante=ante,
                bonus_bet=bonus_bet,
                stake=stake,
            )
            profits.append(avg_profit)
            print(f"  Average profit per hand: {avg_profit:.4f}")
            writer.add_scalar("Performance/Profit", avg_profit, iteration)

            model_path = f"{save_dir}/deep_cfr_bonus_iter_{iteration}.pt"
            agent.save_model(model_path)
            print(f"  Model saved to {model_path}")

        if iteration % checkpoint_frequency == 0:
            ckpt_path = f"{save_dir}/bonus_checkpoint_iter_{iteration}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "advantage_net": agent.advantage_net.state_dict(),
                    "strategy_net": agent.strategy_net.state_dict(),
                    "losses": losses,
                    "profits": profits,
                    "ante": ante,
                    "bonus_bet": bonus_bet,
                    "stake": stake,
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved to {ckpt_path}")

        elapsed = time.time() - start_time
        writer.add_scalar("Time/Iteration", elapsed, iteration)
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        writer.flush()
        print()

    writer.close()
    return agent, losses, profits


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Deep CFR training for Texas Hold'em Bonus (BonusState)."
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--traversals", type=int, default=200)
    parser.add_argument("--ante", type=float, default=10.0)
    parser.add_argument("--bonus-bet", type=float, default=1.0)
    parser.add_argument("--stake", type=float, default=1000.0)
    parser.add_argument("--save-dir", type=str, default="models_bonus")
    parser.add_argument("--log-dir", type=str, default="logs/deepcfr_bonus")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail fast on illegal env transitions.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_strict_checking(args.strict)

    if args.resume:
        continue_training_bonus(
            checkpoint_path=args.resume,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
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
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
