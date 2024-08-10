import pandas as pd


def main():
    dta = pd.read_csv("results_spirals.csv", sep=" ")
    # Sort data by `accuracy_val` in descending order
    dta = dta.sort_values(by="accuracy_val", ascending=False)
    # Get the top 5 results, except the last column
    top5 = dta.iloc[:5, :-1]
    print(top5)


if __name__ == "__main__":
    main()
