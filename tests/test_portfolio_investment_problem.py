import pandas as pd

from cifo.custom_problem.portfolio_investment_problem import (
    PortfolioInvestmentProblem,
    pip_encoding_rule,
)
from cifo.problem.solution import LinearSolution


def _problem(**constraints):
    decision_variables = [
        {
            "stock": ["AAA", "BBB"],
            "stock_name": ["A", "B"],
            "price": [10, 20],
            "exp_ret": [5, 10],
            "stdiv": [0.1, 0.2],
        },
        pd.DataFrame({"AAA": [1.0, 1.1, 1.2], "BBB": [2.0, 2.2, 2.4]}),
    ]
    base_constraints = {
        "Risk-free-rate": 0,
        "Max-Investment": 1000,
        "Risk-Tolerance": 0,
    }
    base_constraints.update(constraints)
    return PortfolioInvestmentProblem(decision_variables, base_constraints)


def test_default_optimum_stock_count_uses_all_stocks():
    problem = _problem()

    assert problem._optimum_stocks == 2


def test_portfolio_problem_does_not_mutate_default_encoding_rule():
    original_size = pip_encoding_rule["Size"]
    original_data = list(pip_encoding_rule["Data"])

    _problem()

    assert pip_encoding_rule["Size"] == original_size
    assert pip_encoding_rule["Data"] == original_data


def test_evaluate_solution_handles_zero_weight_portfolio():
    problem = _problem()
    solution = LinearSolution([0, 0], problem.encoding_rule)

    evaluated = problem.evaluate_solution(solution)

    assert evaluated.fitness == 0
