# Modeling Social Unrest in Canada

This project aims to model and analyze social unrest in Canada using generalized linear models (GLM) and statistical techniques.

## Project Overview

The goal is to capture the relationship between various independent variables (year, month, Canadian provinces, and population) and the number of protests recorded. The project involves:

1. Model selection
2. GLM regression
3. Bootstrap methods
4. Monte Carlo simulation

## Files in the Project

-   `protestData.csv`: Contains the dataset used for analysis, provided by Canadian government
-   `fitModel.py`: Python script for fitting the negative binomial regression model
-   `social-unrest.tex`: LaTeX file containing the project report
-   `social-unrest.pdf`: PDF version of the project report

## Key Features

-   Negative binomial regression to model count data
-   Parametric bootstrapping for confidence intervals
-   Monte Carlo simulation for future protest predictions

## How to Run

1. Ensure you have Python installed
2. Install the required dependencies by running:
    ```
    pip install -r requirements.txt
    ```
3. Run the `fitModel.py` script to perform the analysis:
    ```
    python fitModel.py
    ```
4. The script will generate plots and output results

## Results

The analysis provides:

-   Model parameters and their significance
-   Bootstrapped confidence intervals for model parameters
-   Monte Carlo simulation results for expected protests in 2025

## Conclusion

This study offers a framework for modeling and understanding social unrest in Canada, potentially providing insights into future protest numbers across different provinces.

## Author

Matthew Neba
