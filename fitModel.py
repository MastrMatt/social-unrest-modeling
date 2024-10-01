import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

# Goal: Use negative binomial regression to fit the model, Independent variables are: year, month, prov, pop. Dependent variable is: number of protests

# Load the data
data = pd.read_csv("./protestData.csv")


def count_zeros(data):
    # count the number of zeros in the data
    return data[data["protests"] == 0].shape[0]


def sample_negative_binomial(muMatrix, alpha):
    """
    Generate new dependent variable samples from the negative binomial distribution
    :param mu:  mean of the values of the state matrix
    :param alpha: the dispersion parameter, constant for all values of the state matrix
    :return: an array of samples
    """

    # create matrix for p and r
    pMatrix = muMatrix / (muMatrix + alpha * (muMatrix**2))
    nMatrix = (pMatrix * muMatrix) / (1 - pMatrix)

    # generate samples from the negative binomial distribution
    samples = np.random.negative_binomial(nMatrix, pMatrix)

    return samples


def negative_binomial_regression(data):
    # fit a negative binomial regression model to the data

    # Convert "year" and "month" to ordinal variables, 1 indexed
    data["year"] = data["year"].apply(lambda x: 1 if x == 2022 else 2)

    # Convert "month" to integer, 1 indexed
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    data["month"] = data["month"].apply(lambda x: months.index(x) + 1)

    # Since prov has no order, need dummy variables with drop_first=True to avoid multicollinearity
    data = pd.get_dummies(data, columns=["prov"], drop_first=True)

    # convert the booleans to integers
    for col in data.columns:
        if col.startswith("prov_"):
            data[col] = data[col].astype(int)

    # add a constant to the data for proper regression
    data = sm.add_constant(data)

    # Create the independent variables matrix
    X = data[
        ["year", "month", "pop"]
        + [col for col in data.columns if col.startswith("prov_")]
    ]

    # Find an approximation to the best alpha for the model using a simple grid search for the log likelihood
    alphas = np.linspace(0.05, 20, 100)
    best_alpha = None
    best_model = None

    # try different alphas to determine best alpha for model, use likelihood as test
    for alpha in alphas:
        model = sm.GLM(
            data["protests"],
            X,
            family=sm.families.NegativeBinomial(alpha=alpha),
        ).fit()
        if best_alpha is None or model.llf > best_model.llf:
            best_alpha = alpha
            best_model = model

    # get the negative binomial mean and alpha parameters from the model

    # array that has the mean for every year,month,prov,pop combination
    stateMatrixMeans = best_model.mu

    # array that has the alpha, constant for every year,month,prov,pop combination
    alpha = best_model.family.alpha

    # time to do bootstrapping to determine significance of the independent variables
    n_bootstraps = 5000
    b_coefficients = []

    for i in range(n_bootstraps):
        # generate samples from the negative binomial distribution
        samples = sample_negative_binomial(stateMatrixMeans, alpha)

        # fit the model to the samples
        model = sm.GLM(
            samples,
            X,
            family=sm.families.NegativeBinomial(alpha=alpha),
        ).fit()

        # get the coefficients of the model
        b_coefficients.append(model.params)

    # get the confidence intervals for the coefficients
    b_coefficients = np.array(b_coefficients)

    # get the 95% confidence intervals for the coefficients
    lower = np.percentile(b_coefficients, 2.5, axis=0)
    upper = np.percentile(b_coefficients, 97.5, axis=0)

    provinceMap = {
        "prov_u": "AB",
        "prov_British Columbia": "BC",
        "prov_Manitoba": "MB",
        "prov_New Brunswick": "NB",
        "prov_Newfoundland and Labrador": "NL",
        "prov_Northwest Territories": "NT",
        "prov_Nova Scotia": "NS",
        "prov_Nunavut": "NU",
        "prov_Ontario": "ON",
        "prov_Prince Edward Island": "PE",
        "prov_Quebec": "QC",
        "prov_Saskatchewan": "SK",
        "prov_Yukon": "YT",
    }

    # abreviate province names to fit in plot
    plotCoulums = ["year", "month", "pop"] + [
        provinceMap[col] for col in X.columns if col.startswith("prov_")
    ]

    # plot the confidence intervals
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x=range(len(plotCoulums)),
        y=best_model.params,
        yerr=[
            best_model.params - lower,
            upper - best_model.params,
        ],
        fmt="o",
        color="black",
    )
    plt.xticks(range(len(X.columns)), plotCoulums, rotation=90)
    plt.ylabel("Coefficient")
    plt.title("Negative Binomial Regression Coefficients")
    plt.show()

    # Determine the significance of the coefficients from the bootstrap confidence intervals, see if CI contains 0
    for i in range(len(X.columns)):
        if lower[i] <= 0 and upper[i] >= 0:
            pass
        else:
            print(f"The coefficient for {X.columns[i]} is significant")

    print(best_model.summary())

    # do monte carlo simulation to predict the number of protests in 2025
    monte_carlo_simulation_2025(best_model, X)

    # Return the best model
    return best_model


def monte_carlo_simulation_2025(model, X):
    # simulate the number of protests in 2025 using the model

    # extract the independent variables for 2025, this is done by extrapolating the independent variables for 2022to 2025
    X_2025 = X[X["year"] == 1]

    # Increment the year by 3 to get to 2025 using .loc
    X_2025.loc[:, "year"] = 4

    # get the last 156 rows of the state matrix means, these are the means for 2022
    stateMatrixMeans = model.mu[-156:]

    # multiply the state matrix means by (e^beta)^2 to get the mean for 2025, remember that the effect of the coefficient is a multiplicative effect
    stateMatrixMeans = stateMatrixMeans * (np.exp(model.params["year"])) ** 3

    # get the alpha for the model
    alpha = model.family.alpha

    # construct 95% confidence intervals for number of protests in 2025
    n = 5000
    sampleMatrix = np.zeros((n, 13))

    for i in range(n):
        sample = sample_negative_binomial(stateMatrixMeans, alpha)

        # sum up the number of protests for every province, 13 provinces
        insertSampleMatrix = np.zeros(13)

        for j in range(13):
            for k in range(12):
                insertSampleMatrix[j] += sample[k + 12 * j]

        sampleMatrix[i] = insertSampleMatrix

    # get the 95% confidence intervals for the number of protests in 2025
    lower = np.percentile(sampleMatrix, 2.5, axis=0)
    upper = np.percentile(sampleMatrix, 97.5, axis=0)

    provinces = [
        "AB",
        "BC",
        "MB",
        "NB",
        "NL",
        "NT",
        "NS",
        "NU",
        "ON",
        "PE",
        "QC",
        "SK",
        "YT",
    ]

    # graph the 95% confidence intervals for each province for the number of protests in 2025
    plt.figure(figsize=(12, 6))
    for i, province in enumerate(provinces):
        plt.plot([i, i], [lower[i], upper[i]], color="blue", linewidth=2)
        plt.scatter([i], [lower[i]], color="blue", marker="_", s=100)
        plt.scatter([i], [upper[i]], color="blue", marker="_", s=100)

    plt.xticks(range(len(provinces)), provinces)
    plt.ylabel("Number of Protests")
    plt.title("95% Confidence Intervals for Number of Protests in 2025")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # get the number of samples
    n = data.shape[0]

    #run the experiment
    model = negative_binomial_regression(data)
