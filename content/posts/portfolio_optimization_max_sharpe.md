---
title: "Portfolio Optimization with Python - Part II"
date: 2023-07-05
# weight: 1
# aliases: ["/first"]
tags: ["MOSEK", "Python", "Portfolio Optimization", "Efficient Frontier", "Sharpe Ratio"]
author: "Matias Macazaga"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Max Sharpe portfolio optimization using MOSEK's Fusion API."
# canonicalURL: "https://canonical.url/to/page"
disableHLJS: true # to disable highlightjs
disableShare: false
# disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "/img/portfolio_optimization_2.png" # image path/url
    alt: "Sharpe Ratio portfolio optimization" # alt text
    caption: "[Source](https://unsplash.com/photos/N__BnvQ_w18?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink)" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
# editPost:
#     URL: "https://github.com/<path_to_repo>/content"
#     Text: "Suggest Changes" # edit text
#     appendFilePath: true # to append file path to Edit link
---

This is the second part of the *Portfolio Optimization with Python* series. In [Part I](https://matimacazaga.github.io/posts/portfolio_optimization/), we introduced the Portfolio Optimization field, the MVO framework and the MOSEK python API for solving a constrained portfolio optimization problem efficiently. This time, we will find the portfolio that maximizes a performance metric called Sharpe Ratio. We will first introduce the Sharpe Ratio and its benefits and drawbacks as a portfolio performance metric. Finally, we will formulate the optimization problem theoretically as a conic problem and solve it using Python.

## Sharpe Ratio

Developed by Nobel laureate William F. Sharpe in 1966, the Sharpe Ratio serves as a cornerstone for portfolio evaluation and comparison. This metric enables investors to assess the risk-adjusted return of a portfolio, providing a comprehensive measure of its efficiency. By incorporating both the returns and volatility of a portfolio, the Sharpe Ratio assists investors in making informed decisions, aiming to maximize returns while minimizing risk.

The Sharpe Ratio is defined as the ratio of the excess return of an investment over the risk-free rate to the standard deviation of the investment's returns. Mathematically, the formula is expressed as:

\\[ SR = \dfrac{R_{p} - R_{f}}{\sigma_{p}}, \\]

where:

- \\(SR\\) is the Sharpe Ratio.
- \\(R_{p}\\) denotes the portfolio's expected return.
- \\(R_{f}\\) signifies the risk-free rate of return.
- \\(\sigma_{p}\\) represents the standard deviation of the portfolio's returns.

We can rewrite this formula in terms of the portfolio weights \\(\mathbf{x}\\), the expected return vector \\(\mathbf{\mu}\\) and the covariance matrix \\(\Sigma\\) as follows:

\\[SR = \dfrac{\mathbf{\mu}^{T}\mathbf{x} - r_{f}}{\sqrt{\mathbf{x}^{T}\Sigma \mathbf{x}}}\\]

### Benefits

The benefits of using the Sharpe Ratio are:

1. Risk-adjusted Performance: the Sharpe Ratio provides a more comprehensive assessment of portfolio performance. It accounts for the inherent risk associated with investments and helps investors evaluate the potential rewards relative to the level of risk undertaken.
2. Comparative Analysis: the Sharpe Ratio allows for straightforward comparisons between different investment portfolios by assessing their risk-adjusted returns.
3. Decision Support: this metric aids in decision-making by quantifying the trade-off between returns and risk. An investor can utilize the Sharpe Ratio to identify portfolios with higher expected returns per unit of risk.

### Drawbacks

1. Assumptions of Normality: the Sharpe Ratio assumes that returns follow a normal distribution, which might not always be the case in real-world scenarios.
2. Sensitivity to Risk-Free Rate: The choice of an appropriate risk-free rate can significantly impact the Sharpe Ratio. The risk-free rate acts as a benchmark, representing the return an investor would earn from a riskless investment. However, different market conditions and varying interpretations of the risk-free rate can introduce subjectivity and affect the interpretation of the Sharpe Ratio.
3. Dependency on Historical Data: the Sharpe Ratio relies on historical data to estimate the average returns and standard deviation of a portfolio. Consequently, it assumes that past performance is indicative of future outcomes. 

While it may not be possible to completely eliminate the drawbacks of the Sharpe Ratio, several approaches can help to mitigate their impact and enhance metric's effectiveness. Although we will not focus on these approaches, here are some strategies to address the drawbacks mentioned above:

- Use other risk-adjusted metrics that are more suitable for modeling non-normal return distributions, such as the Sortino Ratio or the Omega Ratio, which account for downside risk and skweness in return distributions.
- Incorporate Stress Testing and Scenario Analysis to account for extreme market events and analyze portfolio performance in adverse market conditions. This approach can help to refine the risk management strategies.
- Use Robust Portfolio Optimization techniques to incorporate uncertainty by considering a range of potential scenarios, minimizing the impact of extreme events and improving the reliability of risk and return estimates.

### Conic Formulation

The idea now is to use the Sharpe Ratio metric as the objective function in our optimization problem. In this way, the objective function of the portfolio optimization algorithm is

\\[\text{maximize}_{\mathbf{x}} \dfrac{\mathbf{\mu}^{T}\mathbf{x} - r_f }{||\mathbf{G}^{T}\mathbf{x}||_2}, \\]

where we reformulated the risk term as a 2-norm as in the previous post.

In order to derive a conic formulation for the objective function above, we should not that if we could fix the term in the numerator

\\[ \mathbf{\mu}^{T}\mathbf{x} - r_{f} = \text{const}, \\]

then the objective would be equivalent to minimizing \\(||\mathbf{G}^{T}\mathbf{x}||_2\\), which is a standard second-order cone problem.

We do not know in advance what the constant value should be. Solving this problem for all the possible constant values is equivalent to computing the efficient frontier. We do not want to do that, therefore let's denote the constant value using a new scalar variable \\(z\\), \\(z\geq 0\\), as \\(\text{const}=1/z\\). In this way, we now have

\\[ \mathbf{\mu}^{T}\mathbf{x} - r_{f} = \dfrac{1}{z} \\]

The reason for using \\(1/z\\) will become obvious in a moment. If we multiply both sides by \\(z\\), we obtain

\\[z \mathbf{\mu}^{T}\mathbf{x} - r_{f}z = 1 \\]

Denoting \\(\mathbf{y}=z\mathbf{x}\\), this equation becomes

\\[\mathbf{\mu}^{T}\mathbf{y} - r_{f}z = 1 \\]

Given that \\(\mathbf{x} = \mathbf{y}/z\\), the objective function becomes

\\[ \dfrac{\mathbf{\mathbf{\mu}^{T}\mathbf{x} - r_{f}}}{||\mathbf{G}^{T}\mathbf{x}||_2}=\dfrac{1/z}{||\mathbf{G}^{T}\mathbf{\frac{y}{z}}||_2} = \dfrac{1}{||\mathbf{G}^{T}\mathbf{y}||_2}, \\]

which allows us to rewrite the original optimization problem as

\\[ \begin{array}{lr} \text{minimize} & ||\mathbf{G}^{T}\mathbf{y}||_{2} \\\ \text{subject to}  & \mathbf{\mu}^{T}\mathbf{y} - r_f z = 1 \\\ &  z\geq 0 \end{array} \\]

The new problem involves variables \\(\mathbf{y}\\) and \\(z\\), so any additional constraints must be rewritten by substituting \\(x=y/z\\). For instance, the budget constraint \\( \mathbf{1}^{T}\mathbf{x} = 1\\) is reformulated as

\\[ \\mathbf{1}^{T}\\mathbf{y} = z, \\ \\ \\ y\geq 0. \\]

On the other hand, diversification constraints of the form \\(l_{i} \leq x_{i} \leq u_{i}\\) are rewritten as

\\[ zl_{i} \leq y_{i} \leq zu_{i} \\]

Finally, a solution \\(y, z\\) to the reformulated optimization problem gives a solution \\(x=y/z\\) to the original problem.

## Python Implementation

The implementation of the portfolio optimization problem in Python closely resembles the approach presented in our [previous post](https://matimacazaga.github.io/posts/portfolio_optimization/). The primary distinction lies in the modification of variables to reframe the problem as a conic optimization problem.

```python
import mosek.fusion as mf
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
alt.renderers.set_embed_options(actions=False, theme="dark")

def build_model(
    n_assets: int,
    asset_sectors: pd.DataFrame,
    constraints: List[Dict[str, Any]],
    risk_free: float = 0.0,
) -> mf.Model:
    """
    Function for building MOSEK model to solve the mean-variance optimization
    problem with diversification constraints.

    Parameters
    ----------
    n_assets : int
        Number of assets in the investment universe.
    asset_sectors : pd.DataFrame
        DataFrame containing assets' names (`"Asset"` column) and their
        corresponding sectors (`"Sector"` column).
    constraints : List[Dict[str, Any]]
        List of diversification constraints (dictionaries). The dictionaries
        must have the following keys:

        - `"Type"`: type of constraint. It can be `"All Assets"` (the constraint
        applies to all the assets), `"Sectors"` (the constraint applies only
        to assets from a particular sector) or `"Assets"` (the constraint
        applies to a particular asset).
        - `"Weight"`: limit value for the assets' weights.
        - `"Sign"`: domain of the constraint. It can be `">="` (greater than) or
        `"<="` (less than).
        - `"Position"`: indicates to which positions the constraint applies. It
        can be the name of a sector, the name of an asset or an empty string
        (`""`) if the constraint type is `"All Assets"`.
    risk_free: float
        Risk free rate.
        
    Returns
    -------
    model: mf.Model
        MOSEK model object.
    """
    # Creating the model
    model = mf.Model("sharpe_ratio")

    # Auxiliar vector variable y.
    y = model.variable("y", n_assets, mf.Domain.greaterThan(0.0))

    # Auxiliar scalar variable z.
    z = model.variable("z", 1, mf.Domain.greaterThan(0.0))

    # Variable for modeling the portfolio variance in the objective function
    s = model.variable("s", 1, mf.Domain.unbounded())

    # Parameter for cov matrix decomposition.
    G = model.parameter("G", [n_assets, n_assets])

    # Parameter for expected returns vector.
    mu = model.parameter("mu", n_assets)

    # Reformulation constraint
    model.constraint(
        "reformulation",
        mf.Expr.sub(mf.Expr.dot(mu, y), mf.Expr.mul(risk_free, z)),
        mf.Domain.equalsTo(1.0),
    )
    # Budget constraint (fully invested)
    model.constraint(
        "budget", mf.Expr.sub(mf.Expr.sum(y), z), mf.Domain.equalsTo(0)
    )

    # Iterate over the constraints list and add the constraints to the model.
    for c, constraint in enumerate(constraints):
        sign = (
            mf.Domain.greaterThan(0.0)
            if constraint["Sign"] == ">="
            else mf.Domain.lessThan(0.0)
        )

        if constraint["Type"] == "All Assets":
            A = np.identity(n_assets)

            model.constraint(
                f"c_{c}",
                mf.Expr.sub(
                    mf.Expr.mul(A, y),
                    mf.Expr.mul(
                        constraint["Weight"], mf.Var.vrepeat(z, n_assets)
                    ),
                ),
                sign,
            )

        elif constraint["Type"] == "Sectors":
            A = np.where(
                asset_sectors.loc[:, "Sector"] == constraint["Position"],
                1.0,
                0.0,
            )

            model.constraint(
                f"c_{c}",
                mf.Expr.sub(
                    mf.Expr.dot(A, y),
                    mf.Expr.mul(
                        constraint["Weight"], z
                    ),
                ),
                sign,
            )

        elif constraint["Type"] == "Assets":
            A = np.where(
                asset_sectors.loc[:, "Asset"] == constraint["Position"],
                1.0,
                0.0,
            )

            model.constraint(
                f"c_{c}",
                mf.Expr.sub(
                    mf.Expr.dot(A, y),
                    mf.Expr.mul(
                        constraint["Weight"], z
                    ),
                ),
                sign,
            )

    # Conic constraint for the portfolio variance
    model.constraint(
        "risk", mf.Expr.vstack(s, mf.Expr.mul(G, y)), mf.Domain.inQCone()
    )

    # Define objective function
    model.objective(
        "obj",
        mf.ObjectiveSense.Minimize,
        s,
    )

    return model
```

We will employ the identical investment universe and constraints as outlined in the previous post. Our objective is to obtain the vector variable \\(\mathbf{y}\\) and the scalar variable \\(z\\), which will be utilized to derive the optimal portfolio weights by calculating their ratio.

```python
# Get optimal weights from the Model object.
weights = pd.Series(
    constrained_model.getVariable("y").level()
    / constrained_model.getVariable("z").level()[0],
    index=asset_sectors.loc[:, "Asset"],
    name="Constrained",
).to_frame()

weights.loc[:, "Unconstrained"] = (
    unconstrained_model.getVariable("y").level() / 
    unconstrained_model.getVariable("z").level()[0]
)
```

The plot below illustrates the portfolio weights for both the constrained and unconstrained cases. Once again, we observe that the imposition of constraints leads to the selection of a greater number of assets. 

<iframe src="/altair_plots/weights_comparison_sharpe_ratio.html" width="600px" height="420px"></iframe>

We can utilize the following function to calculate the Sharpe Ratio of the portfolios and assess the influence of the constraints on its value.

```python
def get_portfolio_sharpe_ratio(
    weights: np.ndarray | pd.Series,
    mu: np.ndarray | pd.Series,
    sigma: np.ndarray | pd.DataFrame,
    ann_factor: int = 252,
):
    """
    Computes the Annualized Sharpe Ratio.

    Parameters
    ----------
    weights : np.ndarray | pd.Series
        Portfolio weights.
    mu : np.ndarray | pd.Series
        Expected return.
    sigma : np.ndarray | pd.DataFrame
        Covariance matrix.
    ann_factor : int, optional
        Annualization factor, by default 252.

    Returns
    -------
    float
        Sharpe Ratio.
    """
    std_dev = np.sqrt(
        np.dot(
            np.dot(
                weights,
                sigma,
            ),
            weights,
        )
    )

    expected_return = np.dot(
        weights,
        mu,
    )

    return (expected_return/std_dev)*np.sqrt(ann_factor)
```

The constrained portfolio exhibits a Sharpe Ratio of 0.95, whereas the unconstrained version achieves a ratio of 1.10. It is important to note that these calculations rely on historical data, and future portfolio performance may deviate from these values. While the constrained portfolio's Sharpe Ratio is lower than that of the unconstrained version, it is possible that the constraints have positively impacted other significant risk metrics. To further analyze this, a backtest will be performed, as detailed in the upcoming post.

# Conclusion

We have explored the use of the Sharpe Ratio in portfolio optimization and management, highlighting its benefits and limitations. The Sharpe Ratio provides a valuable metric for evaluating risk-adjusted returns and aiding investment decision-making. While it offers insights into portfolio efficiency, it is crucial to consider its assumptions and potential drawbacks.

To further deepen our analysis and gain a comprehensive understanding of the impact of constraints on portfolio performance, the next post will delve into the execution of a backtest. Using Python's `bt` library, we will demonstrate how to conduct a thorough evaluation by simulating the historical performance of constrained and unconstrained portfolios. This will enable us to assess their risk-return trade-offs and gain valuable insights into the efficacy of the portfolio optimization strategies employed.

You can find the full code [here](https://github.com/matimacazaga/portfolio_optimization_with_mosek) in the `sharpe_ratio.ipynb` notebook.

Stay tuned for the next post, where we will explore the intricacies of backtesting and how it can enhance our understanding of portfolio performance in real-world scenarios.
