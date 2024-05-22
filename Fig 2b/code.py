import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import pandas as pd
import seaborn as sns


class Algorithms:
    def __init__(
            self, name, function, ratios, category, alpha=1, int_range=0.2, prob=0.2
    ):
        self.name = name
        self.function = function
        self.ratios = ratios
        self.category = category
        self.alpha = alpha
        self.int_range = int_range
        self.prob = prob


class Item:
    def __init__(self, number, weight, unit_value):
        self.number = number  # Represents the item number (0 for the first item, 1 for the second, etc.)
        self.weight = weight
        self.unit_value = (
            unit_value  # Represents the unit value of the item (value per unit weight)
        )


class Info:
    def __init__(
            self,
            L=1,
            U=500,
            n=150,
            capacity=1,
            name="no_name",
            advice=-1,
            a=-1,
            b=-1,
            sol_num=1,
            alpha=0,
            alg_list=None,
            num_runs=2000,
    ):
        self.advice = advice  # The advice
        self.capacity = capacity  # The capacity
        self.L = L  # Lower bound
        self.U = U  # Upper bound
        self.a = a  # Lower bound interval
        self.b = b  # Upper bound interval
        self.n = n
        self.name = name
        self.sol_num = sol_num
        self.alpha = alpha
        self.alg_list = alg_list
        self.num_runs = num_runs


class Solution:
    def __init__(self, inp):
        self.capacity = inp.capacity
        self.n = inp.n
        self.name = inp.name
        self.selection = [
                             0
                         ] * inp.n  # Initialize an array to store how much of each item is taken
        self.total_value = 0  # Initialize total_value to 0
        self.total_weight = 0
        self.w1 = 0  # temp values
        self.w2 = 0
        self.w3 = 0
        self.s1 = 0
        self.s2 = 0
        self.s3 = 0
        self.b1 = False
        self.b2 = False
        self.b3 = False

    # take min of possible and desired amount
    def set_x(self, w, item):
        amount = min(self.capacity - self.total_weight, w)
        self.selection[item.number] = amount
        self.total_weight += amount
        self.total_value += amount * item.unit_value

    def merge_x(self, sub_solution1, sub_solution2, alpha, item):
        self.set_x(
            alpha * sub_solution1.selection[item.number]
            + (1 - alpha) * sub_solution2.selection[item.number],
            item,
            )


def generatePLaw(n, xmax, alpha):
    s = []  # generate a list of samples from a power law distribution
    for _ in range(n):
        s.append((xmax * ((1 - np.random.uniform(0, 1)) ** (1 / (1 - alpha))) + 1))
    return s


def initialization(n, L, U):
    # Generate 10,000 items with random weights and unit values from a power law distribution
    unit_values = generatePLaw(n, U - 1, 0.8)  # Using generatePLaw for unit values
    dev_weight = np.random.normal(0, 10, 1)[0]  # Generate deviation for weights
    weights = generatePLaw(
        n, 50 + dev_weight, 0.8
    )  # Using generatePLaw for weights with deviation

    # Normalize the weights
    maxw = max(weights)
    weights = [abs(w / maxw) for w in weights]

    unit_values = [max(L, min(U, v)) for v in unit_values]

    items = []
    for i in range(n):
        item = Item(i, weights[i], unit_values[i])  # Use zero-based item numbering
        items.append(item)

    return items, n, L, U


def optimal_offline_knapsack(capacity, items_sorted, n):
    solution = Solution(
        Info(-1, -1, n, capacity, "Offline Optimal")
    )  # Create a Solution instance to store how much of each item is taken
    for item in items_sorted:
        if solution.total_weight + item.weight < capacity:
            # Take the whole item if there's enough capacity
            solution.selection[item.number] = item.weight
            solution.total_weight += solution.selection[item.number]
            solution.total_value += (
                    solution.selection[item.number] * item.unit_value
            )  # Update total_value
        else:
            # Take as much as possible of the item
            solution.selection[item.number] = capacity - solution.total_weight
            solution.total_weight = capacity
            solution.total_value += (
                    solution.selection[item.number] * item.unit_value
            )  # Update total_value
            break  # No more capacity left

    return solution


def smallest_unitvalue(capacity, items, items_sorted, n):
    # Use the optimalOfflineFractionalKnapsack function to find the solution
    solution = optimal_offline_knapsack(capacity, items_sorted, n)

    # Initialize the minimum unit value with a large value
    min_unit_value = float("inf")
    itn = 1

    # Iterate through the items to find the smallest unit value
    for i in range(n):
        selection = solution.selection[i]
        if selection > 0:
            item = items[i]
            unit_value = item.unit_value
            if unit_value < min_unit_value:
                min_unit_value = unit_value
                itn = item.number

    # print(f"1+WR = {1+items[itn].weight}")
    return items[itn].weight, min_unit_value


def solver(items, inp, policy):
    solutions = []
    for i in range(inp.sol_num):
        solutions.append(Solution(inp))
    for i in range(inp.n):
        item = items[i]
        args = tuple(solutions)
        policy(item, inp, *args)
    return solutions[0]


def greedy(item, inp, solution):
    # TODO just to not have zero
    if item.unit_value > inp.advice or item.number == solution.n - 1:
        # Take full item and any remaining of last item
        solution.set_x(item.weight, item)


def noadvice(item, inp, solution):
    if item.unit_value >= phi(solution.w2, inp.L, inp.U):
        solution.set_x(
            min(
                phi_inverse(item.unit_value, inp.L, inp.U) - solution.w2,
                item.weight,
                inp.capacity - solution.w2,
                ),
            item,
        )
        solution.w2 += solution.selection[item.number]


def two_cmpi(item, inp, solution):
    if item.unit_value >= inp.advice:
        # Take half of the item's weight
        solution.set_x(item.weight / 2, item)


def two_cmp(item, inp, solution):
    if item.unit_value > inp.advice:
        # Take half of the item's weight
        solution.set_x(item.weight / 2, item)
    elif item.unitvalue == inp.advice:
        # Take half of the item's weight and don't exceed 0.5
        solution.set_x(
            min(item.weight / 2, inp.capacity / 2 - solution.w1),
            item,
        )
        solution.w1 += solution.selection[item.number]


# TODO (1)not best (2)how to prove ?(3) not correct advice
def wr_cmpi(item, inp, solution):
    if item.unit_value > inp.advice and not solution.b1:
        solution.set_x(item.weight, item)
        solution.s1 += solution.selection[item.number] * item.unit_value
        # sum of all items value before advice
    elif item.unit_value == inp.advice:
        solution.b1 = True
        solution.w1 = item.weight
        # wr
        solution.set_x(
            max(
                (
                        solution.w1 * item.unit_value / (1 + solution.w1)
                        - solution.s1 * solution.w1 / (1 + solution.w1)
                )
                / (item.unit_value),
                0,
                ),
            item,
        )
    elif item.unit_value > inp.advice and solution.b1:
        solution.set_x(item.weight / (1 + solution.w1), item)


# TODO (1)noadvice with k/k+1 capacity input ?? (2)[] (3) best? (3) not correct advice
def interval_noadvice(item, inp, solution, sub_solution1):
    k = 1 + math.log(inp.b / inp.a)
    if item.unit_value > inp.b:
        solution.set_x(item.weight / (k + 1), item)
    elif item.unit_value >= inp.a:
        noadvice(item, inp, sub_solution1)
        solution.merge_x(sub_solution1, solution, k / (k + 1), item)
        # no lasy propagation check ?


# TODO noadvice with k/k+1 capacity input ??
def prob_two_noadvice(item, inp, solution, sub_solution1, sub_solution2):
    two_cmpi(item, inp, sub_solution1)
    noadvice(item, inp, sub_solution2)
    solution.merge_x(sub_solution1, sub_solution2, inp.alpha, item)


def prob_wr_noadvice(item, inp, solution, sub_solution1, sub_solution2):
    wr_cmpi(item, inp, sub_solution1)
    noadvice(item, inp, sub_solution2)
    solution.merge_x(sub_solution1, sub_solution2, inp.alpha, item)


def prob_int_noadvice(item, inp, solution, sub_solution1, sub_solution2, sub_solution3):
    interval_noadvice(item, inp, sub_solution1, sub_solution3)
    noadvice(item, inp, sub_solution2)
    solution.merge_x(sub_solution1, sub_solution2, inp.alpha, item)


# TODO not best how to proof ?
def wr_cmp(capacity, items, n, advice):
    solution = Solution(
        n, "1+w_r Competitive"
    )  # Create a Solution instance to store how much of each item is taken
    not_seen = True
    s = 0
    wr = 1
    for i in range(n):
        item = items[i]

        # Check if the unit value is greater than or equal to the advice
        if item.unit_value > advice and not_seen:
            solution.selection[item.number] = item.weight
            solution.total_value += solution.selection[item.number] * item.unit_value
            solution.total_weight += solution.selection[item.number]
            s += solution.selection[item.number] * item.unit_value
        if item.unit_value == advice:
            not_seen = False
            wr = item.weight
            solution.selection[item.number] = max(
                (wr * item.unit_value / (1 + wr) - s * wr / (1 + wr))
                / (item.unit_value),
                0,
                )
            solution.total_value += solution.selection[item.number] * item.unit_value
            solution.total_weight += solution.selection[item.number]
        if item.unit_value > advice and (not not_seen):
            solution.selection[item.number] = item.weight / (1 + wr)
            solution.total_value += solution.selection[item.number] * item.unit_value
            solution.total_weight += solution.selection[item.number]

    return solution


# TODO
def same_wr_cmp(capacity, items, n, advice):
    solution = Solution(
        n, "1+w_r Competitive"
    )  # Create a Solution instance to store how much of each item is taken
    not_seen = True
    s = 0
    # w = 0
    wr = 0
    # x = 0
    for i in range(n):
        item = items[i]

        # Check if the unit value is greater than or equal to the advice
        if item.unit_value > advice and not_seen:
            solution.selection[item.number] = item.weight
            solution.total_value += solution.selection[item.number] * item.unit_value
            solution.total_weight += solution.selection[item.number]
            # w += solution.selection[item.number]
            s += solution.selection[item.number] * item.unit_value
        if item.unit_value == advice:
            not_seen = False
            wr = item.weight
            solution.selection[item.number] = max(
                (wr * item.unit_value / (1 + wr) - s * wr / (1 + wr))
                / (item.unit_value),
                0,
                )
            solution.total_value += solution.selection[item.number] * item.unit_value
            solution.total_weight += solution.selection[item.number]
            # x = solution.selection[item.number]
        if item.unit_value > advice and (not not_seen):
            # solution.selection[item.number] = item.weight * (
            #   (1 - w - x) / (1 - w)
            # )  # s cannot be 1 so no worry
            solution.selection[item.number] = item.weight / (1 + wr)
            solution.total_value += solution.selection[item.number] * item.unit_value
            solution.total_weight += solution.selection[item.number]

    return solution


def phi(z, L, U):
    if 0 <= z < 1 / (1 + math.log(U / L)):
        return L
    elif 1 / (1 + math.log(U / L)) <= z <= 1:
        return L * math.exp((1 + math.log(U / L)) * z - 1)
    else:
        raise ValueError("z must be in the range [0, 1]")


def phi_inverse(v, L, U):
    if v < L:
        raise ValueError(f"v must be greater than or equal to L ({L})")
    elif v == L:
        return 1 / (1 + math.log(U / L))
    else:
        return (1 / (1 + math.log(U / L))) * (math.log(v / L) + 1)


def report(solutions, items, advice):
    print("Items:")
    for item in items:
        print(
            f"Item {item.number + 1}: Weight: {item.weight:.2f}, Unit Value: {item.unit_value:.2f}"
        )
    print(f"\nAdvice: {advice}")
    for solution in solutions:
        report_soultion(solution, items)


def report_soultion(solution, items):
    print(
        f"\nTotal Value ({solution.name}): {solution.total_value:.2f}: Total Weight Taken: {solution.total_weight:.2f}"
    )
    for i in range(solution.n):
        selection = solution.selection[i]
        if selection > 0:
            item = items[i]
            print(
                f"Item {item.number + 1}: Weight Taken: {selection:.2f}: limit: {item.weight:.2f}: unit_value: {item.unit_value:.2f} "
            )


def drawPlot(optimal, algos, legend, title):
    for result in algos:
        plt.plot(range(len(result)), result, alpha=0.5)
    plt.axhline(y=optimal, color="r", linestyle="-")
    plt.legend(legend)
    plt.title(title)
    plt.ylabel(title.split()[0])
    plt.xlabel("items presented to algorithm")
    plt.show()


def test():
    cap = 1
    items, n, L, U = initialization(30)

    items_sorted = copy.deepcopy(items)
    # Sort items by unit value in descending order
    items_sorted.sort(key=lambda item: item.unit_value, reverse=True)

    solutions = []

    # Example usage of the optimalOfflineFractionalKnapsack function:
    solutions.append(optimal_offline_knapsack(cap, items_sorted, n))

    # Example usage of the twoCompetitive function with advice from smallestUnitValue:
    advice = smallest_unitvalue(cap, items, items_sorted, n)
    solutions.append(solver(cap, items, n, advice, L, U, "Two Comp", two_cmpi))
    # soultion_cr = solution_two_competitive
    # solutions.append(wr_cmp(cap, items, n, advice))
    solutions.append(solver(cap, items, n, advice, L, U, "Noadvice", noadvice))

    # ra = random.uniform(0, advice - 1)
    # rb = random.uniform(0, 50 - advice)
    # solutions.append(interval_noadvice(cap, items, n, advice - ra, advice + rb))
    # print(
    #    f"interval a:{advice-ra} b:{advice+rb} predicted: {2+math.log((advice+rb)/(advice-ra))}"
    # )

    rn = random.uniform(0, 1)
    delta = 0.2  # 20% probability for A
    if rn < delta:
        ra = advice  # Select Advice with 20% probability
    else:
        ra = random.uniform(L, U)
    solutions.append(solver_p(cap, items, n, ra, L, U, "prob", prob_two_noadvice))

    report(
        solutions,
        items,
        advice,
    )


def calculate_algorithm_value(
        alg, items, L, U, n, cap, advice, optimal_solution, optimal_value
):
    if alg.category == "simple":
        alg_value = solver(
            items, Info(L, U, n, cap, alg.name, advice), alg.function
        ).total_value
    elif alg.category == "interval":
        q = 1 / (2 * alg.int_range)
        ra = ((q - 1) * advice + random.uniform(0, advice)) / q
        rb = ((q - 1) * advice + random.uniform(advice, U)) / q
        alg_value = solver(
            items,
            Info(L, U, n, cap, alg.name, advice, ra, rb, sol_num=2),
            alg.function,
        ).total_value
    elif alg.category == "prob":
        rn = random.uniform(0, 1)
        delta = alg.prob
        if rn < delta:
            ra = advice
        else:
            ra = random.uniform(L, U)
        alg_value = solver(
            items,
            Info(L, U, n, cap, alg.name, advice, sol_num=3, alpha=alg.alpha),
            alg.function,
        ).total_value
    elif alg.category == "prob_int":
        rn = random.uniform(0, 1)
        delta = alg.prob
        if rn < delta:
            q = 1 / (2 * alg.int_range)
            ra = ((q - 1) * advice + random.uniform(0, advice)) / q
            rb = ((q - 1) * advice + random.uniform(advice, U)) / q
        elif rn < delta / 2 + 0.5:
            ra = random.uniform(L, advice)
            rb = random.uniform(L, advice)
            if rb < ra:
                ra, rb = rb, ra
        else:
            ra = random.uniform(advice, U)
            rb = random.uniform(advice, U)
            if rb < ra:
                ra, rb = rb, ra

        alg_value = solver(
            items,
            Info(
                L,
                U,
                n,
                cap,
                alg.name,
                advice,
                ra,
                rb,
                sol_num=4,
                alpha=alg.alpha,
            ),
            alg.function,
        ).total_value

    if alg_value > optimal_value + 0.1:
        print(f"Error {alg.name}")
        report(
            optimal_solution,
            solver(cap, items, n, advice, L, U, alg.name, alg.function),
            advice,
        )
    if alg_value == 0:
        print(f"bad {alg.name}")
        alg.ratios.append(10000)
    else:
        alg.ratios.append(optimal_value / alg_value)


def main():
    info_list = inith()
    for info in info_list:
        alg_list = info.alg_list
        num_runs = info.num_runs
        avg_wr = 0
        ii = 0
        while ii < num_runs:
            U = info.U
            L = info.L
            n = info.n
            cap = info.capacity
            # Generate random item configurations for each input
            items, n, L, U = initialization(n, L, U)
            items_sorted = copy.deepcopy(items)
            items_sorted.sort(key=lambda item: item.unit_value, reverse=True)
            optimal_solution = optimal_offline_knapsack(cap, items_sorted, n)
            optimal_value = optimal_solution.total_value
            wr, advice = smallest_unitvalue(cap, items, items_sorted, n)

            # if wr > 0.53:
            #   continue

            avg_wr = (avg_wr * ii + wr) / (ii + 1)
            ii += 1

            # Run each algorithm on the same input and collect the ratios
            for alg in alg_list:
                calculate_algorithm_value(
                    alg, items, L, U, n, cap, advice, optimal_solution, optimal_value
                )

        # Sort ratios for each algorithm
        for alg in alg_list:
            alg.ratios.sort()

        print(avg_wr, len(alg.ratios))  # Assuming avg_wr is defined elsewhere
        #cdf_plot(num_runs, alg_list)

    box_plott(num_runs, [(info.alg_list,info.name) for info in info_list])


from matplotlib.container import ErrorbarContainer

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def box_plott(num_runs, alg_lists):
    # Initialize an empty list to store data
    data_list = []

    # Iterate through each alg_list with its associated aname
    for alg_list, aname in alg_lists:
        # Create a list of dictionaries for the current alg_list
        data_list.extend(
            [
                {"Name": alg.name, "Ratios": ratio, "aname": aname}
                for alg in alg_list
                for ratio in alg.ratios
            ]
        )

    # Create a DataFrame from the combined data
    df = pd.DataFrame(data_list)

    #~~~~~~~~~~~~~
    # Define hatch patterns for each box

    #~~~~~~~~~~~~~
    # Create a custom function to apply hatch patterns

    #~~~~~~~~~~~~~

    # Create a box plot using seaborn
    plt.figure(figsize=(10, 6))
    custom_colors = ["blue", "green", "red", "lime"]
    g = sns.boxplot(
        data=df, x="aname", y="Ratios", hue="Name", notch= True, palette=custom_colors, showfliers=False)
    #palette="Set3",


    plt.title("")
    plt.xlabel("")
    plt.ylabel("empirical competitive ratio", fontsize=21)
    plt.xticks(fontsize=21)
    plt.legend(title="", fontsize=22, loc='upper right')

    # Adjust plot limits as needed
    plt.ylim(1, 5)

    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# def cdf_plot(num_runs, alg_list):
#     # Generate CDF data
#     cdf_base = np.arange(1, num_runs + 1) / num_runs
#
#     # Create CDF plots
#     for alg in alg_list:
#         plt.plot(alg.ratios, cdf_base, label=alg.name)
#
#     plt.xlabel("emprical competitivel ratio")
#     plt.ylabel("emprical CDF")
#     plt.legend()
#     plt.title("")
#     plt.grid()
#
#     plt.xlim(1, 5)  # Set x-axis limits from 1 to 5
#     plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
#
#     plt.show()


def inith():
    alg_list1 = [
        Algorithms("ZCL", noadvice, [], "simple"),
        #  Algorithms("Greedy(A)", greedy, [], "simple"),
        Algorithms("PP-b", two_cmpi, [], "simple"),
        Algorithms("PP-a", wr_cmpi, [], "simple"),
        Algorithms(
            "IPA($10\%$)",
            interval_noadvice,
            [],
            "interval",
            int_range=0.1,
        ),
    ]
    """
    Algorithms(
            "Interval(25% range Advice)",
            interval_noadvice,
            [],
            "interval",
            int_range=0.25,
        ),
        Algorithms(
            "probabilistic Interval (20% probability of 20% range Advice) alpha=0.4",
            prob_int_noadvice,
            [],
            "prob_int",
            0.4,
            int_range=0.2,
        ),
        Algorithms(
            "probabilistic Interval (10% probability of 20% range Advice) alpha=0.9",
            prob_int_noadvice,
            [],
            "prob_int",
            0.9,
            int_range=0.2,
            prob = 0.1,
        ),
        Algorithms(
            "probabilistic Interval (20% probability of 20% range Advice) alpha=0.9",
            prob_int_noadvice,
            [],
            "prob_int",
            0.9,
            int_range=0.2,
        ),
        Algorithms(
            "probabilistic Interval (50% probability of 20% range Advice) alpha=0.9",
            prob_int_noadvice,
            [],
            "prob_int",
            0.9,
            int_range=0.2,
            prob = 0.5,
        ),
        Algorithms(
            "Interval(10% range Advice)",
            interval_noadvice,
            [],
            "interval",
            int_range=0.1,
        ),
        Algorithms(
            "Interval(5% range Advice)",
            interval_noadvice,
            [],
            "interval",
            int_range=0.05,
        ),
    """

    info_list = [
        Info(L=1, U=300, n=150, capacity=1, num_runs=2000,name = '$U/L$=300', alg_list=alg_list1),
        Info(L=1, U=1000, n=150, capacity=1, num_runs=2000, name = '$U/L$=1000', alg_list=copy.deepcopy(alg_list1)),
        Info(L=1, U=5000, n=150, capacity=1, num_runs=2000, name = '$U/L$=5000', alg_list=copy.deepcopy(alg_list1)),
        Info(L=1, U=20000, n=150, capacity=1, num_runs=2000, name = '$U/L$=20000', alg_list=copy.deepcopy(alg_list1)),
    ]

    return info_list


if __name__ == "__main__":
    main()