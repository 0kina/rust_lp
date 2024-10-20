use ::ndarray::{Array1, Array2};

#[derive(Clone, std::fmt::Debug)]
pub struct LpProblem {
    pub costs: Array1<f64>,
    pub coeffs: Array2<f64>,
    pub rhs_vals: Array1<f64>,
}

impl LpProblem {
    /// コストと制約を表すベクトル・行列から LP 問題を構築する．
    /// 最大化の不等式標準形として入力されることを想定．
    ///
    /// # Arguments
    ///
    /// * `costs`: コストベクトル
    /// * `coeffs`: 制約の係数行列
    /// * `rhs_vals`: 制約の右辺値のベクトル
    ///
    /// # Examples
    ///
    /// ```
    /// use ::ndarray::array;
    /// use rust_lp::lp_problem::LpProblem;
    ///
    /// let costs = array![1., 2., 3.];
    /// let coeffs = array![[10., 20., 30.], [100., 200., 300.]];
    /// let rhs_vals = array![1000., 2000.];
    /// let lp: LpProblem = LpProblem::new(costs, coeffs, rhs_vals).unwrap();
    /// ```
    pub fn new(
        costs: Array1<f64>,
        coeffs: Array2<f64>,
        rhs_vals: Array1<f64>,
    ) -> Result<Self, String> {
        if costs.shape()[0] != coeffs.shape()[1] {
            return Err(format!(
                "コストベクトルの長さ {} と 係数行列の列数 {} が異なります",
                costs.shape()[0],
                coeffs.shape()[1]
            ));
        }
        if coeffs.shape()[0] != rhs_vals.shape()[0] {
            return Err(format!(
                "係数行列の行数 {} と 右辺値ベクトルの長さ {} が異なります",
                coeffs.shape()[0],
                rhs_vals.shape()[0],
            ));
        }
        Ok(LpProblem {
            costs,
            coeffs,
            rhs_vals,
        })
    }
}
