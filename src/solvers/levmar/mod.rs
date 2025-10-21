use crate::fit::FitResult;
use crate::prelude::*;
use crate::problem::{RhsType, SeparableProblem};
use levenberg_marquardt::LeastSquaresProblem;
/// Type alias for the solver of the [levenberg_marquardt](https://crates.io/crates/levenberg-marquardt) crate.
/// This provides the Levenberg-Marquardt nonlinear least squares optimization algorithm.
// pub use levenberg_marquardt::LevenbergMarquardt as LevMarSolver;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::{ComplexField, Dyn, RealField, Scalar};
#[cfg(feature = "__lapack")]
use nalgebra_lapack::qr::{QrReal, QrScalar};
#[cfg(feature = "__lapack")]
use num_traits::float::TotalOrder;
#[cfg(feature = "__lapack")]
use num_traits::ConstOne;
#[cfg(feature = "__lapack")]
use num_traits::ConstZero;
use num_traits::{Float, FromPrimitive};
#[cfg(any(test, doctest))]
mod test;

// I got 99 problems but a module ain't one...
// Maybe we'll make this module public, but for now I feel this would make
// the API more complicated.
mod levmar_problem;
#[cfg(feature = "__lapack")]
pub use levmar_problem::GeneralQrLinearSolver;
/// linear solver using column pivoted QR decomposition
pub type CpqrLinearSolver<ScalarType> =
    GeneralQrLinearSolver<ScalarType, nalgebra_lapack::ColPivQR<ScalarType, Dyn, Dyn>>;
pub use levmar_problem::LevMarProblem;
#[cfg(feature = "__lapack")]
pub use levmar_problem::LevMarProblemCpQr;
pub use levmar_problem::LevMarProblemSvd;
pub use levmar_problem::LinearSolver;
pub use levmar_problem::SvdLinearSolver;

/// A thin wrapper around the
/// [`LevenbergMarquardt`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html)
/// solver from the `levenberg_marquardt` crate. This wrapper provides multiple
/// linear solver backends for the linear sub-problems and enables additional
/// functionality like statistics calculation.
#[derive(Debug)]
pub struct LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
{
    solver: LevenbergMarquardt<Model::ScalarType>,
}

impl<Model> LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
{
    /// construct a new solver using the given underlying solver. This allows
    /// us to configure the underlying solver with non-default parameters
    pub fn with_solver(solver: LevenbergMarquardt<Model::ScalarType>) -> Self {
        Self { solver }
    }

    #[allow(clippy::result_large_err)]
    /// generic interface to solve an instance of [`SeparableProblem`] using
    /// variable projection. The [`SeparableProblem`] has to be converted into
    /// a [`LevMarProblem`] first. This struct also exposes convenience methods
    /// that accept [`SeparableProblem`] instances directly, which do the conversions
    /// internally.
    pub fn solve_generic<Rhs: RhsType, Solver: LinearSolver<ScalarType = Model::ScalarType>>(
        &self,
        problem: LevMarProblem<Model, Rhs, Solver>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
        LevMarProblem<Model, Rhs, Solver>: LeastSquaresProblem<Model::ScalarType, Dyn, Dyn>,
    {
        let (problem, report) = self.solver.minimize(problem);

        let LevMarProblem {
            separable_problem,
            cached,
        } = problem;

        let linear_coefficients = cached.map(|cached| cached.linear_coefficients_matrix());

        let result = FitResult::new(separable_problem, linear_coefficients, report);
        if result.was_successful() {
            Ok(result)
        } else {
            Err(result)
        }
    }

    #[cfg(feature = "__lapack")]
    #[allow(clippy::result_large_err)]
    /// Solve the given separable problem with VarPro with a linear solver
    /// backend using column-pivoted QR decomposition, which is typically faster
    /// than SVD, while also exhibiting very good numerical stability, even
    /// for ill-conditioned problems.
    ///
    /// **Note**: This method requires the `lapack` feature to be enabled.
    pub fn solve_with_cpqr<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: QrReal
            + QrScalar
            + Scalar
            + ComplexField
            + RealField
            + Float
            + FromPrimitive
            + TotalOrder
            + ConstOne
            + ConstZero,
    {
        let levmar_problem = LevMarProblemCpQr::from(problem);
        self.solve_generic(levmar_problem)
    }

    #[allow(clippy::result_large_err)]
    /// Solve the given separable problem with VarPro with a linear solver
    /// backend using singular value decomposition (SVD), which exhibits
    /// excellent numerical stability, even for ill-conditioned problems.
    pub fn solve_with_svd<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
    {
        let levmar_problem = LevMarProblemSvd::from(problem);
        self.solve_generic(levmar_problem)
    }

    #[allow(clippy::result_large_err)]
    /// Solve the separable problem using the default SVD-based linear solver.
    ///
    /// This is an alias for [`solve_with_svd`](Self::solve_with_svd) and provides
    /// a reasonable general-purpose default that works in all cases without requiring
    /// additional features. For potentially better performance, consider using
    /// [`solve_with_cpqr`](Self::solve_with_cpqr) if the `lapack` feature is available.
    pub fn solve<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
    {
        self.solve_with_svd(problem)
    }

    /// use [`solve`](Self::solve) instead.
    #[allow(clippy::result_large_err)]
    #[deprecated(since = "0.14.0", note = "use the solve method instead")]
    pub fn fit<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Model: SeparableNonlinearModel,
        Model::ScalarType: Scalar + ComplexField + RealField + Float + FromPrimitive,
    {
        self.solve_with_svd(problem)
    }
}

impl<Model> Default for LevMarSolver<Model>
where
    Model: SeparableNonlinearModel,
    Model::ScalarType: RealField + Float,
{
    fn default() -> Self {
        Self::with_solver(Default::default())
    }
}
