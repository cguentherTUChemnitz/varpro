use super::LinearSolver;

/// compile time builder for the LevMarProblem.
pub struct LevMarProblemBuilder<Prob, Sol, Par> {
    separable_problem: Prob,
    solver: Sol,
    initial_params: Par,
}
