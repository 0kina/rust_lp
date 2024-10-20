use crate::lp_problem::LpProblem;
use ::approx::AbsDiffEq;
use ::ndarray::{array, concatenate, s, Array1, Array2, Axis};

#[derive(std::fmt::Debug, PartialEq)]
struct SimplexTableau {
    basic_variables: Vec<usize>,
    objective_row: Array1<f64>,
    augmented_coeffs: Array2<f64>,
}

impl SimplexTableau {
    const EPS: f64 = 1e-7;
    /// ダミーのタブローを作成する．
    /// このタブローに対して各メソッドを呼んだ場合の動作は未定義とする．
    fn dummy() -> Self {
        SimplexTableau {
            basic_variables: vec![],
            objective_row: array![],
            augmented_coeffs: array![[]],
        }
    }

    fn first_stage(problem: &LpProblem) -> Self {
        let n = problem.costs.shape()[0];
        let m = problem.rhs_vals.shape()[0];
        let inverse_flags = Array1::from(
            problem
                .rhs_vals
                .iter()
                .map(|&v| if v >= 0. { 1. } else { -1. })
                .collect::<Vec<f64>>(),
        );
        let mut tableau = SimplexTableau {
            basic_variables: (n + m..n + 2 * m).collect(),
            objective_row: concatenate![
                Axis(0),
                Array1::zeros(n + m),
                Array1::ones(m) * (-1.),
                Array1::zeros(1)
            ],
            augmented_coeffs: concatenate![
                Axis(1),
                &problem.coeffs * &(inverse_flags.to_shape((m, 1)).unwrap()),
                Array2::eye(m) * &(inverse_flags.to_shape((m, 1)).unwrap()),
                Array2::eye(m),
                (&problem.rhs_vals * &inverse_flags)
                    .to_shape((m, 1))
                    .unwrap()
            ],
        };
        tableau.normalize_objective_row();
        tableau
    }

    fn second_stage(&self, costs: &Array1<f64>) -> Self {
        let m = self.augmented_coeffs.nrows();

        assert_eq!(costs.shape()[0], self.augmented_coeffs.ncols() - 1 - 2 * m);

        // スラック変数と目的関数値の分の m + 1 個を行末に追加する．
        let new_objective_row = concatenate![Axis(0), *costs, Array1::zeros(m + 1)];
        let new_augmented_coeffs = concatenate![
            Axis(1),
            self.augmented_coeffs.slice(s![.., ..-(m as i32 + 1)]),
            self.augmented_coeffs.slice(s![.., -1..])
        ];
        let mut new_tableau = SimplexTableau {
            basic_variables: self.basic_variables.clone(),
            objective_row: new_objective_row,
            augmented_coeffs: new_augmented_coeffs,
        };
        new_tableau.normalize_objective_row();
        new_tableau
    }

    pub fn get_score(&self) -> f64 {
        self.objective_row.last().unwrap() * -1.
    }

    /// 目的関数の基底変数の係数が0になるように調整する．
    fn normalize_objective_row(&mut self) {
        let original_cost = self.objective_row.clone();
        for (row_idx, &basic_idx) in self.basic_variables.iter().enumerate() {
            self.objective_row -= &(original_cost[basic_idx] * &self.augmented_coeffs.row(row_idx));
        }
    }

    /// ピボット列を探してそのインデックスを返す．最適解が求まっている状態だと
    /// ピボット列が存在しないため None を返す．
    /// 巡回を避けるため，最小添字規則を用いる．
    fn find_pivot_column(&self) -> Option<usize> {
        for i in 0..self.objective_row.len() - 1 {
            if self.objective_row[i] > Self::EPS {
                return Some(i);
            }
        }
        None
    }

    /// ピボット行を探してそのインデックスを返す．非有界なインスタンスだと
    /// ピボット行が存在しないため None を返す．
    /// 巡回を避けるため，最小添字規則を用いる．
    fn find_pivot_row(&self, pivot_column: usize) -> Option<usize> {
        let mut pivot_row = 0;
        let mut diff = f64::INFINITY; // TODO: より良い名前を探す
        let m = self.augmented_coeffs.shape()[0];
        let n = self.augmented_coeffs.shape()[1] - 1;
        for i in 0..m {
            let coeff = self.augmented_coeffs[[i, pivot_column]];
            if coeff <= Self::EPS {
                continue;
            }
            let candidate_diff = self.augmented_coeffs[[i, n]] / coeff;
            if candidate_diff < diff {
                pivot_row = i;
                diff = candidate_diff;
            }
        }
        if diff < f64::INFINITY {
            return Some(pivot_row);
        }
        None
    }

    fn simplex_step(&mut self, pivot_row_idx: usize, pivot_column_idx: usize) {
        let pivot_value = self.augmented_coeffs[[pivot_row_idx, pivot_column_idx]];
        // タブローの行に対応する変数を更新
        self.basic_variables[pivot_row_idx] = pivot_column_idx;

        // ピボット行を更新
        {
            let mut pivot_row = self.augmented_coeffs.row_mut(pivot_row_idx);
            pivot_row /= pivot_value;
        }

        // タブローの各行のピボット列が0になるように更新
        let pivot_column = self.augmented_coeffs.column(pivot_column_idx).to_vec();
        for (i, &val) in pivot_column.clone().iter().enumerate() {
            if i == pivot_row_idx {
                continue;
            }
            let (mut row, pivot_row) = self
                .augmented_coeffs
                .multi_slice_mut((s![i, ..], s![pivot_row_idx, ..]));
            row -= &(&pivot_row * val);
        }

        // 目的関数の行を更新
        let val = self.objective_row[pivot_column_idx];
        let pivot_row = self.augmented_coeffs.row(pivot_row_idx);
        self.objective_row -= &(&pivot_row * val);

        // 右辺値が非負になるように係数行列の各要素の符号を調節
        let inverse_flags = Array2::from_shape_vec(
            (self.augmented_coeffs.nrows(), 1),
            self.augmented_coeffs
                .column(self.augmented_coeffs.ncols() - 1)
                .iter()
                .map(|&v| if v >= 0. { 1. } else { -1. })
                .collect::<Vec<f64>>(),
        )
        .unwrap();
        self.augmented_coeffs *= &inverse_flags;
    }

    pub fn solve(&mut self) -> SolverStatus {
        loop {
            let Some(column_idx) = self.find_pivot_column() else {
                return SolverStatus::Optimal;
            };
            let Some(row_idx) = self.find_pivot_row(column_idx) else {
                return SolverStatus::Unbound;
            };
            self.simplex_step(row_idx, column_idx);
        }
    }
}

impl AbsDiffEq for SimplexTableau {
    type Epsilon = f64;
    fn default_epsilon() -> Self::Epsilon {
        1e-7
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.basic_variables == other.basic_variables
            && self
                .objective_row
                .abs_diff_eq(&other.objective_row, epsilon)
            && self
                .augmented_coeffs
                .abs_diff_eq(&other.augmented_coeffs, epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::SimplexTableau;
    use ::ndarray::array;

    /// SimplexTableau::first_stage() によって
    /// 2段階単体法の1段階目に相当するタブローが構築できることのテスト．
    #[test]
    fn test_first_stage() {
        use crate::lp_problem::LpProblem;
        use ::approx::assert_abs_diff_eq;

        let problem = LpProblem::new(
            array![2., 1., 1.],
            array![[1., 2., 0.], [1., 4., 2.]],
            array![12., 20.],
        )
        .unwrap();
        let first_tableau = SimplexTableau::first_stage(&problem);
        assert_abs_diff_eq!(
            first_tableau,
            SimplexTableau {
                basic_variables: vec![5, 6],
                objective_row: array![2., 6., 2., 1., 1., 0., 0., 32.],
                augmented_coeffs: array![
                    [1., 2., 0., 1., 0., 1., 0., 12.],
                    [1., 4., 2., 0., 1., 0., 1., 20.]
                ]
            }
        );
    }
}

#[derive(Clone, Copy, PartialEq, std::fmt::Debug)]
pub enum SolverStatus {
    Unsolved,
    Optimal,
    Infeasible,
    Unbound,
}

pub struct SimplexSolver {
    problem: LpProblem,
    status: SolverStatus,
    tableau: SimplexTableau,
}

impl SimplexSolver {
    pub fn new(problem: &LpProblem) -> Self {
        SimplexSolver {
            problem: (*problem).clone(),
            status: SolverStatus::Unsolved,
            tableau: SimplexTableau::dummy(),
        }
    }

    /// 2段階単体法で求解する．
    pub fn solve(&mut self) -> SolverStatus {
        let mut first_tableau = SimplexTableau::first_stage(&self.problem);
        println!("1段階目のタブロー構築完了");
        println!("{:#?}", first_tableau);
        let first_status = first_tableau.solve();
        match first_status {
            SolverStatus::Optimal => {
                if first_tableau.get_score() < -SimplexTableau::EPS {
                    self.status = SolverStatus::Infeasible;
                    return SolverStatus::Infeasible;
                }
            }
            _ => {
                // 1段階目には必ず最適解が存在するため， panic させる．
                panic!("2段階単体法の1段階目が infeasible または非有界です．")
            }
        }
        println!("1段階目終了．基底変数：{:?}", first_tableau.basic_variables);
        let mut second_tableau = SimplexTableau::second_stage(&first_tableau, &self.problem.costs);
        println!("2段階目のタブロー構築完了");
        println!("{:#?}", second_tableau);
        let second_status = second_tableau.solve();
        self.tableau = second_tableau;
        self.status = second_status;
        println!("2段階目の求解結果：{:?}", second_status);
        if second_status == SolverStatus::Optimal {
            println!("最適値：{}", self.get_score().unwrap());
        }
        second_status
    }

    pub fn get_score(&self) -> Option<f64> {
        match self.status {
            SolverStatus::Optimal => Some(self.tableau.get_score()),
            _ => None,
        }
    }
}
