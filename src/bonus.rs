// bonus.rs
//
// Texas Hold'em Bonus Poker environment.
//
// Casino-style 1v1 (player vs. dealer) variant using the same 52-card
// evaluator as the multi-player No-Limit Hold'em environment. Stages mirror
// the standard `Stage` enum (Preflop / Flop / Turn / River / Showdown) but
// the available actions and bet sizes are constrained:
//
//   Preflop : Fold | Play   (Play commits a Flop Bet of 2 * ante)
//   Flop    : Check | Bet   (Bet commits a Turn Bet of 1 * ante)
//   Turn    : Check | Bet   (Bet commits a River Bet of 1 * ante)
//   River   : -- (auto-advances to Showdown, no betting on the river)
//
// Settlement (vs. dealer):
//   - Dealer wins  -> player loses ante + all play bets
//   - Tie          -> all main bets push (no change to reward)
//   - Player wins  -> Flop / Turn / River bets pay 1:1
//                     Ante pays 1:1 only if player's final hand is a
//                     Straight or better; otherwise the ante pushes.
//
// Bonus side bet (optional, decided strictly from the player's two hole
// cards, regardless of dealer or board):
//
//     Hand                       Payout
//     A-A                        30 : 1
//     A-K suited                 25 : 1
//     A-Q / A-J suited           20 : 1
//     K-K / Q-Q / J-J             8 : 1
//     A-K offsuit                 5 : 1
//     A-Q / A-J offsuit           4 : 1
//     Any other pair              3 : 1
//     Anything else            loses

#![allow(unused)]

use itertools::Itertools;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use rand::{seq::SliceRandom, SeedableRng};
use strum_macros::EnumIter;

use crate::game_logic::rank_card_combination;
use crate::state::card::{Card, CardRank};
use crate::state::stage::Stage;

// ---------------------------------------------------------------------------
// Action / status enums
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum BonusActionEnum {
    Fold,  // Preflop only
    Play,  // Preflop only -> commits 2 * ante
    Check, // Flop / Turn
    Bet,   // Flop / Turn -> commits 1 * ante
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BonusStatus {
    Ok,
    IllegalAction,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Debug, Clone)]
pub struct BonusState {
    #[pyo3(get, set)]
    pub stage: Stage,

    #[pyo3(get, set)]
    pub player_hand: (Card, Card),

    #[pyo3(get, set)]
    pub dealer_hand: (Card, Card),

    #[pyo3(get, set)]
    pub public_cards: Vec<Card>,

    #[pyo3(get, set)]
    pub deck: Vec<Card>,

    #[pyo3(get, set)]
    pub ante: f64,

    #[pyo3(get, set)]
    pub bonus_bet: f64,

    #[pyo3(get, set)]
    pub flop_bet: f64,

    #[pyo3(get, set)]
    pub turn_bet: f64,

    #[pyo3(get, set)]
    pub river_bet: f64,

    #[pyo3(get, set)]
    pub stake: f64,

    #[pyo3(get, set)]
    pub reward: f64,

    #[pyo3(get, set)]
    pub legal_actions: Vec<BonusActionEnum>,

    #[pyo3(get, set)]
    pub final_state: bool,

    #[pyo3(get, set)]
    pub status: BonusStatus,

    #[pyo3(get)]
    pub from_action: Option<BonusActionEnum>,

    /// Reveal the dealer hole cards. False until showdown so observation
    /// builders can mask them for the agent if desired.
    #[pyo3(get, set)]
    pub dealer_revealed: bool,
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

pub struct InitBonusError {
    msg: String,
}

impl std::convert::From<InitBonusError> for PyErr {
    fn from(err: InitBonusError) -> PyErr {
        PyOSError::new_err(err.msg)
    }
}

#[pymethods]
impl BonusState {
    /// Create a new BonusState by shuffling a 52-card deck with `seed`.
    #[staticmethod]
    #[pyo3(signature = (ante, bonus_bet, stake, seed))]
    pub fn from_seed(
        ante: f64,
        bonus_bet: f64,
        stake: f64,
        seed: u64,
    ) -> Result<BonusState, InitBonusError> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut deck: Vec<Card> = Card::collect();
        deck.shuffle(&mut rng);
        BonusState::from_deck(ante, bonus_bet, stake, deck)
    }

    /// Create a BonusState directly from a deck (useful for tests with a
    /// rigged deck order). Top of `deck` is dealt first to the player.
    #[staticmethod]
    #[pyo3(signature = (ante, bonus_bet, stake, deck))]
    pub fn from_deck(
        ante: f64,
        bonus_bet: f64,
        stake: f64,
        mut deck: Vec<Card>,
    ) -> Result<BonusState, InitBonusError> {
        if ante <= 0.0 {
            return Err(InitBonusError {
                msg: "ante must be greater than 0".to_owned(),
            });
        }
        if bonus_bet < 0.0 {
            return Err(InitBonusError {
                msg: "bonus_bet must be non-negative".to_owned(),
            });
        }
        if stake < ante + bonus_bet + 4.0 * ante {
            // Need to be able to cover ante + bonus + worst-case play
            // commitments (Flop=2A, Turn=A, River=A => 4A on top of ante).
            return Err(InitBonusError {
                msg: "stake must cover ante + bonus + 4 * ante".to_owned(),
            });
        }
        if deck.len() < 9 {
            return Err(InitBonusError {
                msg: "deck must contain at least 9 cards (2+2 holes + 5 board)".to_owned(),
            });
        }

        let p1 = deck.remove(0);
        let p2 = deck.remove(0);
        let d1 = deck.remove(0);
        let d2 = deck.remove(0);

        let mut state = BonusState {
            stage: Stage::Preflop,
            player_hand: (p1, p2),
            dealer_hand: (d1, d2),
            public_cards: Vec::new(),
            deck,
            ante,
            bonus_bet,
            flop_bet: 0.0,
            turn_bet: 0.0,
            river_bet: 0.0,
            stake: stake - ante - bonus_bet,
            reward: 0.0,
            legal_actions: Vec::new(),
            final_state: false,
            status: BonusStatus::Ok,
            from_action: None,
            dealer_revealed: false,
        };
        state.legal_actions = compute_legal_actions(&state);
        Ok(state)
    }

    /// Apply a player action and return the resulting (new) BonusState.
    /// The receiver is left untouched (immutable transitions).
    pub fn apply_action(&self, action: BonusActionEnum) -> BonusState {
        // Already terminal -> echo back unchanged.
        if self.final_state || self.status != BonusStatus::Ok {
            return self.clone();
        }

        let mut s = self.clone();
        s.from_action = Some(action);

        if !self.legal_actions.contains(&action) {
            s.status = BonusStatus::IllegalAction;
            s.final_state = true;
            s.legal_actions = Vec::new();
            return s;
        }

        match (s.stage, action) {
            (Stage::Preflop, BonusActionEnum::Fold) => {
                // Player gives up. Showdown shortcut, settle as fold.
                s.stage = Stage::Showdown;
                s.dealer_revealed = true;
                settle(&mut s);
                s.final_state = true;
            }
            (Stage::Preflop, BonusActionEnum::Play) => {
                s.flop_bet = 2.0 * s.ante;
                s.stake -= s.flop_bet;
                deal_flop(&mut s);
                s.stage = Stage::Flop;
            }
            (Stage::Flop, BonusActionEnum::Bet) => {
                s.turn_bet = s.ante;
                s.stake -= s.turn_bet;
                deal_turn(&mut s);
                s.stage = Stage::Turn;
            }
            (Stage::Flop, BonusActionEnum::Check) => {
                deal_turn(&mut s);
                s.stage = Stage::Turn;
            }
            (Stage::Turn, BonusActionEnum::Bet) => {
                s.river_bet = s.ante;
                s.stake -= s.river_bet;
                deal_river(&mut s);
                s.stage = Stage::Showdown;
                s.dealer_revealed = true;
                settle(&mut s);
                s.final_state = true;
            }
            (Stage::Turn, BonusActionEnum::Check) => {
                deal_river(&mut s);
                s.stage = Stage::Showdown;
                s.dealer_revealed = true;
                settle(&mut s);
                s.final_state = true;
            }
            // legal_actions filtering above guarantees we never reach here.
            _ => unreachable!("legal_actions filter should have rejected this"),
        }

        s.legal_actions = compute_legal_actions(&s);
        s
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

// ---------------------------------------------------------------------------
// Helpers (private to the crate)
// ---------------------------------------------------------------------------

fn deal_flop(s: &mut BonusState) {
    for _ in 0..3 {
        s.public_cards.push(s.deck.remove(0));
    }
}
fn deal_turn(s: &mut BonusState) {
    s.public_cards.push(s.deck.remove(0));
}
fn deal_river(s: &mut BonusState) {
    s.public_cards.push(s.deck.remove(0));
}

pub(crate) fn compute_legal_actions(s: &BonusState) -> Vec<BonusActionEnum> {
    if s.final_state {
        return Vec::new();
    }
    match s.stage {
        Stage::Preflop => vec![BonusActionEnum::Fold, BonusActionEnum::Play],
        Stage::Flop | Stage::Turn => vec![BonusActionEnum::Check, BonusActionEnum::Bet],
        Stage::River | Stage::Showdown => Vec::new(),
    }
}

/// Best 5-card rank value from `hole + board`. Lower is better.
pub(crate) fn best_rank(hole: (Card, Card), board: &[Card]) -> (u64, u64, u64) {
    let mut cards: Vec<Card> = board.to_vec();
    cards.push(hole.0);
    cards.push(hole.1);
    cards
        .into_iter()
        .combinations(5)
        .map(rank_card_combination)
        .min()
        .unwrap()
}

/// First component of the rank tuple is the hand category (1=royal flush,
/// ..., 6=straight, 7=trips, ..., 10=high card). "Straight or better"
/// means category <= 6.
pub(crate) const STRAIGHT_OR_BETTER: u64 = 6;

/// Compute the bonus side-bet payout (signed P&L) for the given hole cards.
/// Returns `multiplier * bonus_bet` on a winning two-card combo, or
/// `-bonus_bet` on a loss. Returns 0.0 if no bonus bet was placed.
pub(crate) fn pay_bonus(hand: (Card, Card), bonus_bet: f64) -> f64 {
    if bonus_bet <= 0.0 {
        return 0.0;
    }

    let (a, b) = hand;
    let suited = a.suit == b.suit;
    let r1 = a.rank;
    let r2 = b.rank;
    let pair = r1 == r2;

    // Treat (r1, r2) order-independent.
    let has = |x: CardRank, y: CardRank| (r1 == x && r2 == y) || (r1 == y && r2 == x);

    let multiplier: f64 = if pair && r1 == CardRank::RA {
        30.0
    } else if suited && has(CardRank::RA, CardRank::RK) {
        25.0
    } else if suited && (has(CardRank::RA, CardRank::RQ) || has(CardRank::RA, CardRank::RJ)) {
        20.0
    } else if pair && (r1 == CardRank::RK || r1 == CardRank::RQ || r1 == CardRank::RJ) {
        8.0
    } else if !suited && has(CardRank::RA, CardRank::RK) {
        5.0
    } else if !suited && (has(CardRank::RA, CardRank::RQ) || has(CardRank::RA, CardRank::RJ)) {
        4.0
    } else if pair {
        3.0
    } else {
        return -bonus_bet;
    };

    multiplier * bonus_bet
}

/// Final settlement. Mutates `reward` only.
pub(crate) fn settle(s: &mut BonusState) {
    // 1. Bonus side bet (independent of dealer outcome).
    s.reward += pay_bonus(s.player_hand, s.bonus_bet);

    // 2. Main game.
    if s.from_action == Some(BonusActionEnum::Fold) {
        s.reward -= s.ante;
        return;
    }

    let p_rank = best_rank(s.player_hand, &s.public_cards);
    let d_rank = best_rank(s.dealer_hand, &s.public_cards);

    if p_rank < d_rank {
        // Player wins: play bets pay 1:1.
        s.reward += s.flop_bet + s.turn_bet + s.river_bet;
        // Ante pays only with straight or better, else push.
        if p_rank.0 <= STRAIGHT_OR_BETTER {
            s.reward += s.ante;
        }
    } else if p_rank > d_rank {
        // Dealer wins: lose ante + all play bets.
        s.reward -= s.ante + s.flop_bet + s.turn_bet + s.river_bet;
    }
    // tie: push (no change).
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::card::CardSuit;

    fn c(suit: CardSuit, rank: CardRank) -> Card {
        Card { suit, rank }
    }

    /// Build a deck whose first cards are the ones we want dealt, padded
    /// with arbitrary other cards. Order dealt: P1, P2, D1, D2,
    /// then later: Flop1, Flop2, Flop3, Turn, River.
    fn rigged_deck(cards: Vec<Card>) -> Vec<Card> {
        let mut all = Card::collect();
        // Remove our chosen cards from the pool, then place them up front.
        let chosen = cards.clone();
        all.retain(|x| !chosen.contains(x));
        let mut out = chosen;
        out.append(&mut all);
        out
    }

    // -------------------- Initialization --------------------

    #[test]
    fn init_legal_actions_are_fold_or_play() {
        let s = BonusState::from_seed(10.0, 0.0, 1000.0, 42).unwrap();
        assert_eq!(s.stage, Stage::Preflop);
        assert!(!s.final_state);
        assert_eq!(s.status, BonusStatus::Ok);
        assert!(s.legal_actions.contains(&BonusActionEnum::Fold));
        assert!(s.legal_actions.contains(&BonusActionEnum::Play));
        assert!(!s.legal_actions.contains(&BonusActionEnum::Bet));
        assert!(!s.legal_actions.contains(&BonusActionEnum::Check));
    }

    #[test]
    fn init_rejects_bad_args() {
        // ante must be > 0
        assert!(BonusState::from_seed(0.0, 0.0, 1000.0, 1).is_err());
        // stake too small
        assert!(BonusState::from_seed(10.0, 0.0, 5.0, 1).is_err());
    }

    #[test]
    fn deck_dealt_in_order() {
        let deck = rigged_deck(vec![
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RA),
            c(CardSuit::Clubs, CardRank::R2),
            c(CardSuit::Diamonds, CardRank::R3),
        ]);
        let s = BonusState::from_deck(10.0, 0.0, 1000.0, deck).unwrap();
        assert_eq!(s.player_hand.0.rank, CardRank::RA);
        assert_eq!(s.player_hand.1.rank, CardRank::RA);
        assert_eq!(s.dealer_hand.0.rank, CardRank::R2);
        assert_eq!(s.dealer_hand.1.rank, CardRank::R3);
    }

    // -------------------- Action transitions --------------------

    #[test]
    fn fold_terminates_and_loses_ante_only() {
        let s = BonusState::from_seed(10.0, 0.0, 1000.0, 7).unwrap();
        let s = s.apply_action(BonusActionEnum::Fold);
        assert!(s.final_state);
        assert_eq!(s.stage, Stage::Showdown);
        assert_eq!(s.reward, -10.0);
        assert!(s.legal_actions.is_empty());
    }

    #[test]
    fn play_advances_to_flop_and_charges_2x_ante() {
        let s = BonusState::from_seed(10.0, 0.0, 1000.0, 7).unwrap();
        let stake_before = s.stake;
        let s = s.apply_action(BonusActionEnum::Play);
        assert_eq!(s.stage, Stage::Flop);
        assert_eq!(s.flop_bet, 20.0);
        assert_eq!(s.public_cards.len(), 3);
        assert_eq!(s.stake, stake_before - 20.0);
        assert!(s.legal_actions.contains(&BonusActionEnum::Check));
        assert!(s.legal_actions.contains(&BonusActionEnum::Bet));
    }

    #[test]
    fn check_check_to_showdown() {
        let s = BonusState::from_seed(10.0, 0.0, 1000.0, 7).unwrap();
        let s = s.apply_action(BonusActionEnum::Play);
        let s = s.apply_action(BonusActionEnum::Check);
        assert_eq!(s.stage, Stage::Turn);
        assert_eq!(s.public_cards.len(), 4);
        let s = s.apply_action(BonusActionEnum::Check);
        assert_eq!(s.stage, Stage::Showdown);
        assert!(s.final_state);
        assert_eq!(s.public_cards.len(), 5);
        assert!(s.dealer_revealed);
    }

    #[test]
    fn bet_bet_charges_extra_ante_each() {
        let s = BonusState::from_seed(10.0, 0.0, 1000.0, 7).unwrap();
        let s = s.apply_action(BonusActionEnum::Play);
        let stake_after_play = s.stake;
        let s = s.apply_action(BonusActionEnum::Bet);
        assert_eq!(s.turn_bet, 10.0);
        assert_eq!(s.stake, stake_after_play - 10.0);
        let stake_after_flop_bet = s.stake;
        let s = s.apply_action(BonusActionEnum::Bet);
        assert_eq!(s.river_bet, 10.0);
        assert_eq!(s.stake, stake_after_flop_bet - 10.0);
        assert!(s.final_state);
        assert_eq!(s.stage, Stage::Showdown);
    }

    #[test]
    fn illegal_action_marks_status_and_terminates() {
        let s = BonusState::from_seed(10.0, 0.0, 1000.0, 7).unwrap();
        // Bet is not legal in Preflop.
        let s2 = s.apply_action(BonusActionEnum::Bet);
        assert_eq!(s2.status, BonusStatus::IllegalAction);
        assert!(s2.final_state);
    }

    #[test]
    fn applying_action_after_terminal_is_noop() {
        let s = BonusState::from_seed(10.0, 0.0, 1000.0, 7).unwrap();
        let folded = s.apply_action(BonusActionEnum::Fold);
        let again = folded.apply_action(BonusActionEnum::Play);
        assert_eq!(again.reward, folded.reward);
        assert!(again.final_state);
    }

    // -------------------- Bonus side bet payout --------------------

    #[test]
    fn bonus_pays_pocket_aces_30x() {
        let hand = (
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RA),
        );
        assert_eq!(pay_bonus(hand, 5.0), 150.0);
    }

    #[test]
    fn bonus_pays_ak_suited_25x() {
        let hand = (
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Spades, CardRank::RK),
        );
        assert_eq!(pay_bonus(hand, 5.0), 125.0);
    }

    #[test]
    fn bonus_pays_aq_aj_suited_20x() {
        let aq = (
            c(CardSuit::Hearts, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RQ),
        );
        let aj = (
            c(CardSuit::Diamonds, CardRank::RJ),
            c(CardSuit::Diamonds, CardRank::RA),
        );
        assert_eq!(pay_bonus(aq, 1.0), 20.0);
        assert_eq!(pay_bonus(aj, 1.0), 20.0);
    }

    #[test]
    fn bonus_pays_kqj_pairs_8x() {
        let kk = (
            c(CardSuit::Spades, CardRank::RK),
            c(CardSuit::Hearts, CardRank::RK),
        );
        let qq = (
            c(CardSuit::Spades, CardRank::RQ),
            c(CardSuit::Hearts, CardRank::RQ),
        );
        let jj = (
            c(CardSuit::Spades, CardRank::RJ),
            c(CardSuit::Hearts, CardRank::RJ),
        );
        assert_eq!(pay_bonus(kk, 1.0), 8.0);
        assert_eq!(pay_bonus(qq, 1.0), 8.0);
        assert_eq!(pay_bonus(jj, 1.0), 8.0);
    }

    #[test]
    fn bonus_pays_ak_offsuit_5x() {
        let hand = (
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RK),
        );
        assert_eq!(pay_bonus(hand, 1.0), 5.0);
    }

    #[test]
    fn bonus_pays_aq_aj_offsuit_4x() {
        let aq = (
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RQ),
        );
        let aj = (
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RJ),
        );
        assert_eq!(pay_bonus(aq, 1.0), 4.0);
        assert_eq!(pay_bonus(aj, 1.0), 4.0);
    }

    #[test]
    fn bonus_pays_other_pairs_3x() {
        let tens = (
            c(CardSuit::Spades, CardRank::RT),
            c(CardSuit::Hearts, CardRank::RT),
        );
        let twos = (
            c(CardSuit::Spades, CardRank::R2),
            c(CardSuit::Hearts, CardRank::R2),
        );
        assert_eq!(pay_bonus(tens, 1.0), 3.0);
        assert_eq!(pay_bonus(twos, 1.0), 3.0);
    }

    #[test]
    fn bonus_loses_on_garbage() {
        let hand = (
            c(CardSuit::Spades, CardRank::R7),
            c(CardSuit::Hearts, CardRank::R2),
        );
        assert_eq!(pay_bonus(hand, 5.0), -5.0);
    }

    #[test]
    fn bonus_zero_when_not_placed() {
        let hand = (
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RA),
        );
        assert_eq!(pay_bonus(hand, 0.0), 0.0);
    }

    // -------------------- Settlement (rigged decks) --------------------

    fn play_check_check(state: BonusState) -> BonusState {
        state
            .apply_action(BonusActionEnum::Play)
            .apply_action(BonusActionEnum::Check)
            .apply_action(BonusActionEnum::Check)
    }

    #[test]
    fn settle_player_wins_with_straight_collects_ante_and_play_bets() {
        // Player: A♠ K♠   Dealer: 2♣ 3♣
        // Board:  Q♠ J♠ T♠ 4♦ 5♦  -> player has Royal-ish (actually
        // royal flush with A K Q J T spades). Definitely > straight.
        let deck = rigged_deck(vec![
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Spades, CardRank::RK),
            c(CardSuit::Clubs, CardRank::R2),
            c(CardSuit::Clubs, CardRank::R3),
            c(CardSuit::Spades, CardRank::RQ),
            c(CardSuit::Spades, CardRank::RJ),
            c(CardSuit::Spades, CardRank::RT),
            c(CardSuit::Diamonds, CardRank::R4),
            c(CardSuit::Diamonds, CardRank::R5),
        ]);
        let s = BonusState::from_deck(10.0, 0.0, 1000.0, deck).unwrap();
        let s = play_check_check(s);
        // Flop bet = 20, no turn/river bets. Wins all + ante (royal flush).
        // Reward = ante (10) + flop_bet (20) = 30.
        assert!(s.final_state);
        assert_eq!(s.reward, 30.0);
    }

    #[test]
    fn settle_player_wins_with_only_pair_pushes_ante() {
        // Player: A♠ A♥ (pocket aces, three of a kind on board possible)
        // Dealer: 2♣ 3♣
        // Board: K♦ 7♥ 9♣ 4♦ 5♥ -> player has pair of Aces (cat 9),
        //   dealer has high card 5-9-K... pair beats high card.
        //   Player wins, but final hand category = 9 (pair) -> ante pushes.
        let deck = rigged_deck(vec![
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RA),
            c(CardSuit::Clubs, CardRank::R2),
            c(CardSuit::Clubs, CardRank::R3),
            c(CardSuit::Diamonds, CardRank::RK),
            c(CardSuit::Hearts, CardRank::R7),
            c(CardSuit::Clubs, CardRank::R9),
            c(CardSuit::Diamonds, CardRank::R4),
            c(CardSuit::Hearts, CardRank::R5),
        ]);
        let s = BonusState::from_deck(10.0, 0.0, 1000.0, deck).unwrap();
        let s = play_check_check(s);
        // Reward = flop_bet (20). Ante pushed.
        assert!(s.final_state);
        assert_eq!(s.reward, 20.0);
    }

    #[test]
    fn settle_dealer_wins_loses_everything() {
        // Player: 2♠ 3♠   Dealer: A♠ A♥
        // Board: 7♣ 8♥ 9♦ 4♣ 5♣ -> player has straight (3-4-5...?)
        //   Actually player has 2-3-4-5 and needs a 6 or A: no 6 here,
        //   so player just makes high card. Dealer has pair of aces,
        //   dealer wins.
        let deck = rigged_deck(vec![
            c(CardSuit::Spades, CardRank::R2),
            c(CardSuit::Spades, CardRank::R3),
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RA),
            c(CardSuit::Clubs, CardRank::R7),
            c(CardSuit::Hearts, CardRank::R8),
            c(CardSuit::Diamonds, CardRank::R9),
            c(CardSuit::Clubs, CardRank::R4),
            c(CardSuit::Clubs, CardRank::R5),
        ]);
        let s = BonusState::from_deck(10.0, 0.0, 1000.0, deck).unwrap();
        let s = play_check_check(s);
        assert!(s.final_state);
        // Lose ante (10) + flop_bet (20) = -30.
        assert_eq!(s.reward, -30.0);
    }

    #[test]
    fn settle_tie_pushes() {
        // Player and dealer both make the exact same straight using the
        // board (board plays). Player: 2♣ 3♥, Dealer: 2♦ 3♠ — both pairs
        // get dominated by board. Use a board straight: 5-6-7-8-9 mixed
        // suits. Both player and dealer best 5 = board straight. Tie.
        let deck = rigged_deck(vec![
            c(CardSuit::Clubs, CardRank::R2),
            c(CardSuit::Hearts, CardRank::R3),
            c(CardSuit::Diamonds, CardRank::R2),
            c(CardSuit::Spades, CardRank::R3),
            c(CardSuit::Hearts, CardRank::R5),
            c(CardSuit::Diamonds, CardRank::R6),
            c(CardSuit::Clubs, CardRank::R7),
            c(CardSuit::Spades, CardRank::R8),
            c(CardSuit::Hearts, CardRank::R9),
        ]);
        let s = BonusState::from_deck(10.0, 0.0, 1000.0, deck).unwrap();
        let s = play_check_check(s);
        assert!(s.final_state);
        // Tie -> push, reward unchanged from 0.
        assert_eq!(s.reward, 0.0);
    }

    #[test]
    fn settle_combines_main_and_bonus_bets() {
        // Same board as the royal flush win test, but with a bonus bet.
        // Player has AKs in spades -> bonus pays 25:1.
        let deck = rigged_deck(vec![
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Spades, CardRank::RK),
            c(CardSuit::Clubs, CardRank::R2),
            c(CardSuit::Clubs, CardRank::R3),
            c(CardSuit::Spades, CardRank::RQ),
            c(CardSuit::Spades, CardRank::RJ),
            c(CardSuit::Spades, CardRank::RT),
            c(CardSuit::Diamonds, CardRank::R4),
            c(CardSuit::Diamonds, CardRank::R5),
        ]);
        let s = BonusState::from_deck(10.0, 5.0, 1000.0, deck).unwrap();
        let s = play_check_check(s);
        assert!(s.final_state);
        // Main game: 30 (as above).
        // Bonus side bet: AK suited -> 25 * 5 = 125.
        // Total reward: 30 + 125 = 155.
        assert_eq!(s.reward, 155.0);
    }

    #[test]
    fn settle_fold_still_pays_bonus_side_bet() {
        // Pocket aces: bonus pays 30:1 regardless of the fold.
        let deck = rigged_deck(vec![
            c(CardSuit::Spades, CardRank::RA),
            c(CardSuit::Hearts, CardRank::RA),
            c(CardSuit::Clubs, CardRank::R2),
            c(CardSuit::Diamonds, CardRank::R3),
        ]);
        let s = BonusState::from_deck(10.0, 5.0, 1000.0, deck).unwrap();
        let s = s.apply_action(BonusActionEnum::Fold);
        assert!(s.final_state);
        // Lose ante (-10). Bonus pays 30 * 5 = 150.
        assert_eq!(s.reward, 140.0);
    }
}
