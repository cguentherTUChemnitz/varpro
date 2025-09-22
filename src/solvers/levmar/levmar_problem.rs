use crate::{
    model::SeparableNonlinearModel,
    problem::{RhsType, SeparableProblem},
};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{ComplexField, Const, DMatrix, Dyn, Owned, Scalar};
mod svd;

// TODO only on lapack feature
pub mod colpiv_qr;

pub use colpiv_qr::ColPivQrLinearSolver;
pub use svd::SvdSolver;

mod builder;

#[allow(type_alias_bounds)]
/// levmar problem where the linear part is solved via column pivoted QR
/// decomposition.
//TODO expose only on lapack feature
pub type LevMarProblemCpQr<Model: SeparableNonlinearModel, Rhs> =
    LevMarProblem<Model, Rhs, ColPivQrLinearSolver<Model::ScalarType>>;

#[allow(type_alias_bounds)]
pub type LevMarProblemSvd<Model: SeparableNonlinearModel, Rhs> =
    LevMarProblem<Model, Rhs, SvdSolver<Model::ScalarType>>;

#[derive(Debug)]
pub struct LevMarProblem<Model, Rhs, Solver>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    Solver: LinearSolver<ScalarType = Model::ScalarType>,
    Rhs: RhsType,
    Model: SeparableNonlinearModel,
{
    pub(crate) separable_problem: SeparableProblem<Model, Rhs>,
    pub(crate) cached: Option<Solver>,
}

impl<Model, Rhs, Solver> From<SeparableProblem<Model, Rhs>> for LevMarProblem<Model, Rhs, Solver>
where
    Model::ScalarType: Scalar + ComplexField + Copy,
    Solver: LinearSolver<ScalarType = Model::ScalarType>,
    Rhs: RhsType,
    Model: SeparableNonlinearModel,
    Self: LeastSquaresProblem<
        Model::ScalarType,
        Dyn,
        Dyn,
        ParameterStorage = Owned<Model::ScalarType, Dyn, Const<1>>,
    >,
{
    fn from(problem: SeparableProblem<Model, Rhs>) -> Self {
        let mut this = Self {
            separable_problem: problem,
            cached: None,
        };
        this.set_params(&this.separable_problem.model.params());
        this
    }
}

/// helper trait to abstract over the linear solvers (as of now ColPivQr
/// and SVD) used in the LevMarProblem. We don't actually abstract much
/// here, other than giving a method for getting the linear coefficients,
/// becaue the actual implementation of the LeastSquaresProblem
/// trait is done for specializations on the solvers for concrete types.
///
/// This is only used once when the problem is solved internally, which
/// is why we can move out of the solver here and save some needless
/// copies.
pub trait LinearSolver: std::fmt::Debug + sealed::Sealed {
    type ScalarType: Scalar;
    /// get the linear coefficients in matrix form. For single RHS
    /// this is a matrix with just one column.
    fn linear_coefficients_matrix(self) -> DMatrix<Self::ScalarType>;
}

impl<ScalarType: ComplexField> sealed::Sealed for ColPivQrLinearSolver<ScalarType> {}
impl<ScalarType: ComplexField> sealed::Sealed for SvdSolver<ScalarType> {}

pub mod sealed {
    pub trait Sealed {}
}
