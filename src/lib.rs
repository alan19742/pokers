// lib.rs
use pyo3::prelude::*;
mod bonus;
mod game_logic;
mod parallel;
mod state;
mod visualization;

/// A Python module implemented in Rust.
#[pymodule]
fn pokers(_py: Python, m: &PyModule) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<state::State>()?;
    m.add_class::<state::PlayerState>()?;
    m.add_class::<state::StateStatus>()?;
    m.add_class::<state::stage::Stage>()?;
    m.add_class::<state::action::ActionEnum>()?;
    m.add_class::<state::action::Action>()?;
    m.add_class::<state::action::ActionRecord>()?;
    m.add_class::<state::card::Card>()?;
    m.add_class::<bonus::BonusState>()?;
    m.add_class::<bonus::BonusActionEnum>()?;
    m.add_class::<bonus::BonusStatus>()?;
    m.add_function(wrap_pyfunction!(visualization::visualize_state, m)?)?;
    m.add_function(wrap_pyfunction!(visualization::visualize_trace, m)?)?;
    m.add_function(wrap_pyfunction!(parallel::parallel_apply_action, m)?)?;
    Ok(())
}
