"""
Contains a simple PageRank implementation.
"""

import json
from typing import Dict, List
from argparse import ArgumentParser

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm


N_ITER: int = 500
ALPHAS: List[float] = [0.5, 0.75, 0.8, 0.85, 0.90, 0.95, 0.975, 0.99, 0.999]


def build_google_matrix(
    adj_list: Dict[int, List[int]],
    alpha: float = 0.85,
) -> npt.NDArray:
    """
    Constructs the dense Google Matrix
    based on the adjacency list of the web graph

    G = alpha * H + 1/N * (alpha*a + (1-alpha)*e)*e.T

    Args:
        - adj_list: Dict[int, List[int]], adjacency list
                    of the web graph with the stucture
                    <page_id>: [<outbound_links>]
        - alpha: float, dampening factor
    Return:
        - G: numpy.NDArray, google matrix
    """

    # Dimensionality of Transition Matrix
    N = len(adj_list.keys())

    # Init H as zero matrix
    global H
    H = np.zeros(shape=(N, N))

    # Fill 1/|P| for outgoing links
    for row, entries in adj_list.items():
        # Remove redundant links
        entries = set(entries)
        m = len(entries)
        if m > 0:
            for col in entries:
                H[row, col] = 1 / m

    # Compute the google matrix
    # G = alpha * H + 1/N * (alpha*a + (1-alpha)*e)*e.T

    # vector of all 0-s - stochasticity adjustment vector
    a = (H.sum(axis=1) == 0).astype(int).reshape(-1, 1)
    e = np.ones(shape=(N, 1))

    S = H + (1 / N) * (a @ e.T)
    E = (1 / N) * e @ e.T

    G = alpha * S + (1 - alpha) * E
    return G


if __name__ == "__main__":
    # Load the dataset containing adjacency list
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="source file containing the adjacency matrix",
        metavar="FILE",
    )
    parser.add_argument(
        "-d", "--dest", dest="destination", help="destination folder", metavar="DIR"
    )
    parser.parse_args()
    adj_list_path = parser.filename

    with open(adj_list_path, "r") as fp:
        dataset = json.load(fp)

        index = {int(k): v for k, v in dataset["index"].items()}
        inv_index = dataset["inverseIndex"]
        adj_list = {int(k): v for k, v in dataset["adjacencyList"].items()}
        titles = dataset["titles"]

    # For collection of resulting pageranks and their convergence
    pageranks: Dict[float, npt.NDArray] = {}
    deltas: Dict[float, List[float]] = {}
    N = len(adj_list.keys())

    # Iterate for each value of alpha
    for a in tqdm(ALPHAS):
        G = build_google_matrix(adj_list, alpha=a)

        # Init the uniformely distributed pagerank vector
        pi = 1 / N * np.ones(N)

        delta = []  # Init collection of deltas

        # Power Method
        for i in range(N_ITER):
            pi_next = pi.T @ G

            # Compute total variation
            d = np.abs(pi_next - pi).sum()
            delta.append(d)

            pi = pi_next

        # Append obtained
        pageranks[a] = pi
        deltas[a] = delta

    # Calculate the rankings depending on alpha
    pi50 = pageranks[0.50]
    ranks50 = {index.get(i): s for i, s in list(enumerate(pi50))}
    series50 = pd.Series(ranks50)

    pi85 = pageranks[0.85]
    ranks85 = {index.get(i): s for i, s in list(enumerate(pi85))}
    series85 = pd.Series(ranks85)

    pi99 = pageranks[0.99]
    ranks99 = {index.get(i): s for i, s in list(enumerate(pi99))}
    series99 = pd.Series(ranks99)

    # Count inbound and outbound links
    P = (H != 0).astype(int)
    refs = P.sum(axis=0)
    ref_counts = {index.get(i): s for i, s in list(enumerate(refs))}
    sorted_refs = {k: v for k, v in sorted(ref_counts.items(), key=lambda x: -x[1])}
    refs_series = pd.Series(sorted_refs)

    out_refs = P.sum(axis=1)
    out_ref_counts = {index.get(i): s for i, s in list(enumerate(out_refs))}
    out_sorted_refs = {
        k: v for k, v in sorted(out_ref_counts.items(), key=lambda x: -x[1])
    }
    out_refs_series = pd.Series(out_sorted_refs)

    # combine in a dataset
    ranks_df = pd.concat(
        [series50, series85, series99, refs_series, out_refs_series], axis=1
    )
    ranks_df.columns = [
        "pr_50",
        "pr_85",
        "pr_99",
        "N_inbound_links",
        "N_outbound_links",
    ]
    ranks_df["rank_99"] = ranks_df["pr_99"].rank(ascending=False).astype(int)
    ranks_df["rank_85"] = ranks_df["pr_85"].rank(ascending=False).astype(int)
    ranks_df["rank_50"] = ranks_df["pr_50"].rank(ascending=False).astype(int)

    ranks_df["rank_by_inbound_links"] = ranks_df.N_inbound_links.rank(
        ascending=False
    ).astype(int)
    ranks_df = ranks_df.reset_index().rename(mapper={"index": "link"}, axis=1)
    ranks_df["title"] = ranks_df["link"].apply(titles.get)
    ranks_df["idx"] = ranks_df["link"].apply(inv_index.get)
    ranks_df = ranks_df.sort_values(by="rank_85")

    # Save the obtained pagerank df as CSV
    dest = parser.dest
    ranks_df.to_csv(f"{dest}/pagerank.csv")
    pd.DataFrame(deltas).to_csv(f"{dest}/convergence.csv")
