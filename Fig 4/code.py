from cmath import log
import random
from re import A
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from matplotlib.container import ErrorbarContainer


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


class Setting:
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
        wr=0.5,
        delta=0,
    ):
        self.L = L  # Lower bound
        self.U = U  # Upper bound
        self.n = n
        self.capacity = capacity  # The capacity
        self.name = name
        self.advice = advice  # The advice
        self.a = a  # Lower bound interval
        self.b = b  # Upper bound interval
        self.sol_num = sol_num
        self.alpha = alpha
        self.alg_list = alg_list
        self.num_runs = num_runs
        self.wr = wr
        self.delta = delta


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
        self.frac_range= {}
        self.integral_range= {}

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


def initialization2(n, L, U, info, delta):
    # Generate 10,000 items with random weights and unit values from a power law distribution

    delta = delta
    items = []
    k = 0
    up = 0
    low = 0
    info.b = 0
    info.a = 0
    for i in range(U, L - 1, -1):
        ww = 0.001
        lv = random.randint(50, 150)
        uv = math.floor(lv * (1 + delta))
        sv = random.randint(lv, uv)
        if up < 1:
            up += uv * ww
            if up >= 1:
                info.b = i

        if low < 1:
            low += lv * ww
            if low >= 1:
                info.a = i
        for j in range(sv):
            item = Item(k, ww, i)  # Use zero-based item numbering
            items.append(item)
            k += 1
    random.shuffle(items)
    ii = 0
    for it in items:
        it.number = ii
        ii += 1

    n = k
    info.n = k
    print(f"a:{info.a} b:{info.b} n:{len(items)} ")
    return items, n, L, U


def initialization3(n, L, U, info, delta, count_runs):
    filename = "random_numbers{}.txt".format(count_runs + 1)
    items_list = read_items_from_file(filename)
    n = len(items_list)
    info.n = n
    info.a = L
    info.U = U
    L = 700
    U = 20000
    return items_list, n, L, U


def initialization4(n, L, U, info, delta, count_runs):
    filename = "con_numbers_{}.txt".format(count_runs + 2)
    items_list = read_items_from_file(filename)
    n = len(items_list)
    info.n = n
    info.a = L
    info.U = U
    L = 700
    U = 20000
    return items_list, n, L, U


def initialization11(n, L, U, info, delta, count_runs):
    filename = "rand_numbers_{}.txt".format(count_runs + 1)
    items_list = read_items_from_file(filename)
    n = len(items_list)
    info.n = n
    info.a = L
    info.U = U
    L = 700
    U = 20000
    return items_list, n, L, U


def initialization12(n, L, U, info, delta, count_runs):
    filename = "ravi_{}_{}.txt".format(count_runs + 1, int(10 * delta))
    items_list = read_items_from_file12(filename)
    n = len(items_list)
    L = 1
    U = 100
    info.n = n
    info.a = L
    info.U = U
    return items_list, n, L, U

def initialization13(n, L, U, info, delta, count_runs):
    filename = "con_numbers_{}.txt".format(count_runs + 1)
    items_list = read_items_from_file(filename)
    n = len(items_list)
    info.n = n
    L = 700
    U = 20000
    info.a = L
    info.U = U
    return items_list, n, L, U

def read_items_from_file12(filename):
    items = []
    with open(filename, "r") as file:
        n = int(file.readline().strip())  # Read the number of items from the first line
        for idx, line in enumerate(file):
            unit_value = float(line.strip())
            number = idx
            w0 = 0.0001
            items.append(Item(number, w0, unit_value))
    return items


def read_numbers_from_file(filename):
    numbers = []
    with open(filename, "r") as file:
        for line in file:
            # Strip any leading/trailing whitespace and convert to integer
            number = int(line.strip())
            numbers.append(number)
    return numbers


def read_items_from_file(filename):
    items = []
    with open(filename, "r") as file:
        for idx, line in enumerate(file):
            # Assuming each line contains only the unit value
            unit_value = float(line.strip())
            # Set number as per file index
            number = idx
            # Set weight as w0
            w0 = 0.001
            items.append(Item(number, w0, unit_value))
    return items


def optimal_offline_knapsack(capacity, items_sorted, n):
    solution = Solution(
        Setting(-1, -1, n, capacity, "Offline Optimal")
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
    if item.unit_value > inp.advice:  # or item.number == solution.n - 1:
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


def im(item, inp, solution):
    print("CCCCCCCCCCCCCCC")
    # do nothing


def ppai_b(item, inp, solution, subsol):
    ppa_b(item, inp, subsol)
    j = log(item.univt_value / inp.advice, 1 + inp.d)
    solution.frac_range[j] += subsol.selection[item.number] * item.unit_value
    if solution.frac_range[j] > solution.integral_range[j]:
        solution.set_x(item.weight, item)
        solution.integral_range[j] += item.weight * item.unitvalue


def ppa_b(item, inp, solution):
    if item.unit_value > inp.advice:
        # Take half of the item's weight
        solution.set_x(item.weight / 2, item)
    elif item.unit_value == inp.advice:
        # Take half of the item's weight and not over cap
        temp = min(item.weight, inp.capacity - solution.w1)
        solution.set_x(
            temp / 2,
            item,
        )
        solution.w1 += temp


def conv_ppa_a(item, inp, solution, subsol):
    ppa_a(item, inp, subsol)
    j = log(item.unit_value /inp.L, 1 + 0.02)#d
    solution.frac_range[j] = solution.frac_range.get(j,0) + subsol.selection[item.number] * item.unit_value
    if solution.frac_range[j]*0.99 > solution.integral_range.get(j,0):
        solution.set_x(item.weight, item)
        solution.integral_range[j] = solution.integral_range.get(j,0) + solution.selection[item.number] * item.unit_value


def ppa_a(item, inp, solution):
    if item.unit_value > inp.advice:
        solution.set_x(item.weight / (1 + solution.w1), item)
    elif item.unit_value == inp.advice:
        temp = min(item.weight, inp.capacity - solution.w1)
        solution.w1 += temp
        # wr
        solution.set_x(
            (1 - solution.s1) * temp / (solution.w1 + 1),
            item,
        )
    solution.s1 += solution.selection[item.number]


def ipai(item, inp, solution, sub_solution1):
    t = inp.b / inp.a
    k = 1 + math.log(t)
    if item.unit_value > inp.b:
        solution.set_x(item.weight / (k + 1), item)
    elif item.unit_value >= inp.a:
        noadvice(item, inp, sub_solution1)
        solution.merge_x(sub_solution1, solution, k / (k + 1), item)
        # no lasy propagation check ?


def ipa(item, inp, solution, sub_solution1):
    t = inp.b / inp.a
    k = 1 + math.log(t)
    if item.unit_value > inp.b:
        solution.set_x(item.weight / (k + 1), item)
    elif item.unit_value >= inp.a:
        noadvice(item, inp, sub_solution1)
        solution.merge_x(sub_solution1, solution, k / (k + 1), item)
        # no lasy propagation check ?


# TODO noadvice with k/k+1 capacity input ??
def prob_two_noadvice(item, inp, solution, sub_solution1, sub_solution2):
    ppa_b(item, inp, sub_solution1)
    noadvice(item, inp, sub_solution2)
    solution.merge_x(sub_solution1, sub_solution2, inp.alpha, item)


def prob_wr_noadvice(item, inp, solution, sub_solution1, sub_solution2):
    ppa_a(item, inp, sub_solution1)
    noadvice(item, inp, sub_solution2)
    solution.merge_x(sub_solution1, sub_solution2, inp.alpha, item)


def prob_int_noadvice(item, inp, solution, sub_solution1, sub_solution2, sub_solution3):
    ipa(item, inp, sub_solution1, sub_solution3)
    noadvice(item, inp, sub_solution2)
    solution.merge_x(sub_solution1, sub_solution2, inp.alpha, item)


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


def get_third_number(file_path, first_number, second_number):
    try:
        with open(file_path, "r") as file:
            for line in file:
                numbers = line.split()
                if len(numbers) == 3:
                    num1, num2, num3 = map(float, numbers)
                    if num1 == first_number and num2 == second_number:
                        return num3
        # If the line with the given numbers is not found, return None
        return None
    except FileNotFoundError:
        print("File not found.")
        return None
    except ValueError:
        print("Invalid data format in the file.")
        return None


def calculate_algorithm_value(
    alg, items, L, U, n, cap, advice, optimal_solution, optimal_value, info, count_runs
):
    if alg.category == "simple":
        alg_res = solver(items, Setting(L, U, n, cap, alg.name, advice), alg.function)
        alg_value = alg_res.total_value
    elif alg.category == "simple2":
        alg_value = solver(
            items,
            Setting(L, U, n, cap, alg.name, advice, sol_num=2),
            alg.function,
        ).total_value
    elif alg.category == "interval":
        q = 1 / (2 * alg.int_range)
        ra = ((q - 1) * advice + random.uniform(0, advice)) / q
        rb = ((q - 1) * advice + random.uniform(advice, U)) / q
        alg_value = solver(
            items,
            Setting(L, U, n, cap, alg.name, advice, ra, rb, sol_num=2),
            alg.function,
        ).total_value
    elif alg.category == "input":
        alg_value = get_third_number("im1.txt", info.delta, count_runs)
    elif alg.category == "interval*":
        ra = info.a
        rb = info.b
        alg_value = solver(
            items,
            Setting(L, U, n, cap, alg.name, advice, ra, rb, sol_num=2),
            alg.function,
        ).total_value
    elif alg.category == "interval**":
        ra = info.a
        rb = info.b
        alg_value = solver(
            items,
            Setting(L, U, n, cap, alg.name, advice, ra, rb, sol_num=2),
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
            Setting(L, U, n, cap, alg.name, advice, sol_num=3, alpha=alg.alpha),
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
            Setting(
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
        # report(
        #     optimal_solution,
        #     solver(cap, items, n, advice, L, U, alg.name, alg.function),
        #     advice,
        # )
    if alg_value == 0:
        print(f"Bad {alg.name}")
        # print(alg_res.total_weight)
        alg.ratios.append(10000)
    else:
        print(alg_value)
        alg.ratios.append(optimal_value / alg_value)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mainX():
    info_list = inith()
    for info in info_list:
        alg_list, num_runs = info.alg_list, info.num_runs
        avg_wr, count_runs = 0, 0
        U, L, n, cap = info.U, info.L, info.n, info.capacity

        while count_runs < num_runs:
            # Generate random item configurations for each input
            items, n, L, U = initialization3(n, L, U, info, 2.5, count_runs)
            items_sorted = copy.deepcopy(items)
            items_sorted.sort(key=lambda item: item.unit_value, reverse=True)
            optimal_solution = optimal_offline_knapsack(cap, items_sorted, n)
            optimal_value = optimal_solution.total_value
            wr, advice = smallest_unitvalue(cap, items, items_sorted, n)
            # print(info.wr)
            print(wr, advice, optimal_value)
            # if  not(max(0,2*info.wr-1)<= wr <=min(1,2*info.wr)):
            #   continue

            avg_wr = (avg_wr * count_runs + wr) / (count_runs + 1)
            count_runs += 1

            # Run each algorithm on the same input and collect the ratios
            for alg in alg_list:
                calculate_algorithm_value(
                    alg,
                    items,
                    L,
                    U,
                    n,
                    cap,
                    advice,
                    optimal_solution,
                    optimal_value,
                    info,
                )

        # Sort ratios for each algorithm
        for alg in alg_list:
            alg.ratios.sort()
            print(alg.name)
            print(alg.ratios)

        # print(avg_wr, len(alg.ratios))  # Assuming avg_wr is defined elsewhere
        # cdf_plot(num_runs, alg_list)

    box_plott(num_runs, [(info.alg_list, info.name) for info in info_list])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

    # ~~~~~~~~~~~~~
    # Define hatch patterns for each box

    # ~~~~~~~~~~~~~
    # Create a custom function to apply hatch patterns

    # ~~~~~~~~~~~~~

    # Create a box plot using seaborn
    plt.figure(figsize=(10, 6))
    custom_colors = ["blue", "green", "red", "black"]
    g = sns.boxplot(
        data=df,
        x="aname",
        y="Ratios",
        hue="Name",
        notch=True,
        palette=custom_colors,
        showfliers=True,
    )
    # palette="Set3",

    plt.title("")
    plt.xlabel("")
    plt.ylabel("Ratios", fontsize=23)
    plt.xticks(fontsize=15)
    plt.legend(title="", fontsize=12.2, loc="upper right")

    # Adjust plot limits as needed
    plt.ylim(1, 5)

    plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def box_plot11(num_runs, alg_lists):
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

    # ~~~~~~~~~~~~~
    # Define hatch patterns for each box

    # ~~~~~~~~~~~~~
    # Create a custom function to apply hatch patterns

    # ~~~~~~~~~~~~~

    # Create a box plot using seaborn
    plt.figure(figsize=(10, 6))
    custom_colors = ["blue", "green", "red", "lime"]
    g = sns.boxplot(
        data=df,
        x="aname",
        y="Ratios",
        hue="Name",
        notch=True,
        palette=custom_colors,
        showfliers=False,
    )
    # palette="Set3",

    plt.title("")
    plt.xlabel("")
    plt.ylabel("empirical competitive ratio", fontsize=21)
    plt.xticks(fontsize=21)
    
    plt.legend(title="", fontsize=22, loc="upper left")

    # Adjust plot limits as needed
    plt.ylim(1, 1.1)

    plt.show()


def cdf_plot(num_runs, alg_list):
    # Generate CDF data
    cdf_base = np.arange(1, num_runs + 1) / num_runs

    # Create CDF plots
    # Define line styles

    # line style for Alpha
    line_styles = ["-.", ":", "--"]
    custom_line_style = (5, (3, 10))
    custom_line_style_2 = (0, (3, 1, 1, 1))
    custom_line_style_3 = (0, (3, 1, 1, 1, 1))
    line_colors = ["blue", "green", "red", "black", "pink"]

    for i, alg in enumerate(alg_list):
        plt.plot(
            alg.ratios,
            cdf_base,
            label=alg.name,
            linestyle=line_styles[i],
            linewidth=2,
            color=line_colors[i],
        )
        line_styles.append(custom_line_style)
        line_styles.append(custom_line_style_2)
        line_styles.append(custom_line_style_3)

    plt.xlabel("empirical competitive ratio", fontsize=23)
    plt.ylabel("empirical CDF", fontsize=23)
    """  plt.legend()"""
    plt.title("")
    plt.grid()

    plt.xlim(1, 5)  # Set x-axis limits from 1 to 5
    # Increase the font size of the x-axis tick labels
    plt.xticks(fontsize=23)  # Adjust the fontsize as needed
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    # Increase the font size of the x-axis tick labels
    plt.yticks(fontsize=23)  # Adjust the fontsize as needed

    # legend for alpha
    legend = ["TA", "PPA-n", "PPA-b", "PPA-a", r"PIPA$_{\delta}$(50%)", "IPA(25%)"]

    """
    plt.legend(legend, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=len(legend))
   """

    plt.legend(legend, loc="lower right", fontsize=21)
    plt.show()


def inith():
    alg_list1 = [
        # Algorithms("PPA-b", ppa_b, [], "simple"),
        Algorithms("PP-a", ppa_a, [], "simple"),
        Algorithms("Fr2Int-PP-a", conv_ppa_a, [], "simple2"),
        # Algorithms("PPA-n", greedy, [], "simple"),
        Algorithms("SENTINEL", im, [], "input"),
        #Algorithms("ZCL", noadvice, [], "simple"),
    ]
    """
    Algorithms(
            "Interval(10% range Advice)",
            interval_noadvice,
            [],
            "interval**",
            int_range=0.10,
        ),
    Algorithms(
            "Interval(10% range Advice)",
            interval_noadvice,
            [],
            "interval*",
            int_range=0.10,
        ),
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
        Setting(
            L=1,
            U=100,
            delta=0,
            n=10000,
            capacity=1,
            num_runs=24,
            name="δ=0",
            alg_list=copy.deepcopy(alg_list1),
        ),
        Setting(
            L=1,
            U=100,
            delta=0.5,
            n=10000,
            capacity=1,
            num_runs=24,
            name="δ=0.5",
            alg_list=copy.deepcopy(alg_list1),
        ),
        Setting(
            L=1,
            U=100,
            delta=1,
            n=10000,
            capacity=1,
            num_runs=24,
            name="δ=1",
            alg_list=copy.deepcopy(alg_list1),
        ),
        Setting(
            L=1,
            U=100,
            delta=1.5,
            n=10000,
            capacity=1,
            num_runs=24,
            name="δ=1.5",
            alg_list=copy.deepcopy(alg_list1),
        ),
        Setting(
            L=1,
            U=100,
            delta=2,
            n=10000,
            capacity=1,
            num_runs=24,
            name="δ=2",
            alg_list=copy.deepcopy(alg_list1),
        ),
    ]

    """
        Info(L=1, wr=0.45, n=150, capacity=1, num_runs=2000, name = 'wr=0.5', alg_list=copy.deepcopy(alg_list1)),
        Info(L=1, wr=0.6, n=150, capacity=1, num_runs=2000, name = 'wr=0.7', alg_list=copy.deepcopy(alg_list1)),
        Info(L=1, wr=0.75, n=150, capacity=1, num_runs=2000, name = 'wr=0.9', alg_list=copy.deepcopy(alg_list1)),
    """

    return info_list


def test4():
    advlist = []
    info_list = inith()
    for info in info_list:
        alg_list, num_runs = info.alg_list, info.num_runs
        avg_wr, count_runs = 0, 0
        U, L, n, cap = info.U, info.L, info.n, info.capacity

        while count_runs < num_runs:
            # Generate random item configurations for each input
            items, n, L, U = initialization4(n, L, U, info, 2.5, count_runs)
            items_sorted = copy.deepcopy(items)
            items_sorted.sort(key=lambda item: item.unit_value, reverse=True)
            optimal_solution = optimal_offline_knapsack(cap, items_sorted, n)
            optimal_value = optimal_solution.total_value

            wr, advice = smallest_unitvalue(cap, items, items_sorted, n)
            # print(info.wr)
            print(wr, advice, optimal_value)
            # if  not(max(0,2*info.wr-1)<= wr <=min(1,2*info.wr)):
            #   continue

            avg_wr = (avg_wr * count_runs + wr) / (count_runs + 1)
            count_runs += 1

            print("XXX:")
            print(optimal_value)
            if count_runs != 1:
                print("guess:")
                print(sad_advice + items[0].unit_value)
                print("intial value:")
                print(items[0].unit_value)
                print("advice")
                print(advice)

                info.a = np.min(advlist) + items[0].unit_value
                info.b = np.max(advlist) + items[0].unit_value
                # Run each algorithm on the same input and collect the ratios
                for alg in alg_list:
                    print(alg.name)
                    calculate_algorithm_value(
                        alg,
                        items,
                        L,
                        U,
                        n,
                        cap,
                        sad_advice + items[0].unit_value,
                        optimal_solution,
                        optimal_value,
                        info,
                    )

            advlist.append(advice - items[0].unit_value)
            sad_advice = np.average(advlist)

        # Sort ratios for each algorithm
        for alg in alg_list:
            print(alg.name)
            print(alg.ratios)
            alg.ratios.sort()

        # print(avg_wr, len(alg.ratios))  # Assuming avg_wr is defined elsewhere
        # cdf_plot(num_runs, alg_list)

    box_plott(num_runs, [(info.alg_list, info.name) for info in info_list])


def test11():
    info_list = inith()
    for info in info_list:
        alg_list, num_runs = info.alg_list, info.num_runs
        avg_wr, count_runs = 0, 0
        U, L, n, cap = info.U, info.L, info.n, info.capacity

        while count_runs < num_runs:
            # Generate random item configurations for each input
            items, n, L, U = initialization11(n, L, U, info, info.delta, count_runs)
            items_sorted = copy.deepcopy(items)
            items_sorted.sort(key=lambda item: item.unit_value, reverse=True)
            optimal_solution = optimal_offline_knapsack(cap, items_sorted, n)
            optimal_value = optimal_solution.total_value

            wr, advice = smallest_unitvalue(cap, items, items_sorted, n)

            # Calculate the lower and upper bounds for the noisy advice
            # lower_bound = max(items_sorted[-1].unit_value, advice // (info.delta + 1))
            # upper_bound = min(advice * (), items_sorted[0].unit_value)
            random_noise_factor = random.uniform(1, 1 + info.delta)
            r1 = random.randint(0, 1)
            if r1 == 1:
                noisy_advice = max(
                    (1 / random_noise_factor) * advice, items_sorted[-1].unit_value
                )
            else:
                noisy_advice = min(
                    (random_noise_factor + 1) / 2 * advice, items_sorted[1].unit_value
                )
            # Generate noisy advice within the specified range
            # if int(lower_bound)>int(upper_bound):
            #     noisy_advice = advice

            # else:
            #     noisy_advice = random.randint(int(lower_bound), int(upper_bound))

            print(wr, advice, optimal_value, noisy_advice)

            avg_wr = (avg_wr * count_runs + wr) / (count_runs + 1)
            count_runs += 1

            # Run each algorithm on the same input and collect the ratios
            for alg in alg_list:
                print(alg.name)
                calculate_algorithm_value(
                    alg,
                    items,
                    L,
                    U,
                    n,
                    cap,
                    noisy_advice,
                    optimal_solution,
                    optimal_value,
                    info,
                    count_runs,
                )

        # Sort ratios for each algorithm
        for alg in alg_list:
            print(alg.name)
            print(alg.ratios)
            alg.ratios.sort()

        # print(avg_wr, len(alg.ratios))  # Assuming avg_wr is defined elsewhere
        # cdf_plot(num_runs, alg_list)

    box_plot11(num_runs, [(info.alg_list, info.name) for info in info_list])


def im0():
    info_list = inith()
    for info in info_list:
        alg_list, num_runs = info.alg_list, info.num_runs
        avg_wr, count_runs = 0, 0
        U, L, n, cap = info.U, info.L, info.n, info.capacity

        while count_runs < num_runs:
            # Generate random item configurations for each input
            items, n, L, U = initialization12(n, L, U, info, info.delta, count_runs)
            items_sorted = copy.deepcopy(items)
            items_sorted.sort(key=lambda item: item.unit_value, reverse=True)
            optimal_solution = optimal_offline_knapsack(cap, items_sorted, n)
            optimal_value = optimal_solution.total_value

            wr, advice = smallest_unitvalue(cap, items, items_sorted, n)

            # Calculate the lower and upper bounds for the noisy advice
            # lower_bound = max(items_sorted[-1].unit_value, advice // (info.delta + 1))
            # upper_bound = min(advice * (), items_sorted[0].unit_value)
            # random_noise_factor = random.uniform(1, 1 + info.delta)
            # r1 = random.randint(0, 1)
            # if r1 == 1:
            #     noisy_advice = max(
            #         (1 / random_noise_factor) * advice, items_sorted[-1].unit_value
            #     )
            # else:
            #     noisy_advice = min(
            #         (random_noise_factor + 1) / 2 * advice, items_sorted[1].unit_value
            #     )
            # Generate noisy advice within the specified range
            # if int(lower_bound)>int(upper_bound):
            #     noisy_advice = advice

            # else:
            #     noisy_advice = random.randint(int(lower_bound), int(upper_bound))
            # noisy_advice = 100 - 100/(1+(info.delta/2))
            noisy_advice = get_third_number("adv0.txt", info.delta, count_runs + 1)
            print(advice, optimal_value, noisy_advice)

            avg_wr = (avg_wr * count_runs + wr) / (count_runs + 1)
            count_runs += 1

            # Run each algorithm on the same input and collect the ratios
            for alg in alg_list:
                print(alg.name)
                calculate_algorithm_value(
                    alg,
                    items,
                    L,
                    U,
                    n,
                    cap,
                    noisy_advice,
                    optimal_solution,
                    optimal_value,
                    info,
                    count_runs,
                )

        # Sort ratios for each algorithm
        for alg in alg_list:
            print(alg.name)
            print(alg.ratios)
            alg.ratios.sort()

        # print(avg_wr, len(alg.ratios))  # Assuming avg_wr is defined elsewhere
        # cdf_plot(num_runs, alg_list)

    box_plot11(num_runs, [(info.alg_list, info.name) for info in info_list])


def im1():
    info_list = inith()
    for info in info_list:
        alg_list, num_runs = info.alg_list, info.num_runs
        avg_wr, count_runs = 0, 0
        U, L, n, cap = info.U, info.L, info.n, info.capacity

        while count_runs < num_runs:
            # Generate random item configurations for each input
            items, n, L, U = initialization13(n, L, U, info, info.delta, count_runs)
            items_sorted = copy.deepcopy(items)
            items_sorted.sort(key=lambda item: item.unit_value, reverse=True)
            optimal_solution = optimal_offline_knapsack(cap, items_sorted, n)
            optimal_value = optimal_solution.total_value

            wr, advice = smallest_unitvalue(cap, items, items_sorted, n)

            # Calculate the lower and upper bounds for the noisy advice
            # lower_bound = max(items_sorted[-1].unit_value, advice // (info.delta + 1))
            # upper_bound = min(advice * (), items_sorted[0].unit_value)
            # random_noise_factor = random.uniform(1, 1 + info.delta)
            # r1 = random.randint(0, 1)
            # if r1 == 1:
            #     noisy_advice = max(
            #         (1 / random_noise_factor) * advice, items_sorted[-1].unit_value
            #     )
            # else:
            #     noisy_advice = min(
            #         (random_noise_factor + 1) / 2 * advice, items_sorted[1].unit_value
            #     )
            # Generate noisy advice within the specified range
            # if int(lower_bound)>int(upper_bound):
            #     noisy_advice = advice

            # else:
            #     noisy_advice = random.randint(int(lower_bound), int(upper_bound))
            # noisy_advice = 100 - 100/(1+(info.delta/2))
            noisy_advice = get_third_number("adv1.txt", info.delta, count_runs + 1)
            print(advice, optimal_value, noisy_advice)

            avg_wr = (avg_wr * count_runs + wr) / (count_runs + 1)
            count_runs += 1

            # Run each algorithm on the same input and collect the ratios
            for alg in alg_list:
                print(alg.name)
                calculate_algorithm_value(
                    alg,
                    items,
                    L,
                    U,
                    n,
                    cap,
                    noisy_advice,
                    optimal_solution,
                    optimal_value,
                    info,
                    count_runs,
                )

        # Sort ratios for each algorithm
        for alg in alg_list:
            print(alg.name)
            print(alg.ratios)
            alg.ratios.sort()

        # print(avg_wr, len(alg.ratios))  # Assuming avg_wr is defined elsewhere
        # cdf_plot(num_runs, alg_list)

    box_plot11(num_runs, [(info.alg_list, info.name) for info in info_list])



def sum_top_10000(file_path):
    # Step 1: Read the file and store items in a list
    items = []
    with open(file_path, "r") as file:
        next(file)
        for line in file:
            # Step 2: Convert each line to a number and add to the list
            items.append(float(line.strip()))

    # Step 3: Sort the items in descending order
    items.sort(reverse=True)

    # Step 4: Get the top 10,000 items and sum them
    top_10000_sum = sum(items[:10000])

    return top_10000_sum


def pythonTEST():
    file_path = "ravi_3_0.txt"  # Replace with the path to your file
    result = sum_top_10000(file_path)
    print(f"The sum of the top 10,000 items is: {result}")


if __name__ == "__main__":
    # mainX()
    # test4()
    # test11()
    # pythonTEST()
    #im0()
    im1()