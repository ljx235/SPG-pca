"""
Pathway-guided Nonnegative Matrix Factorization with Graph Filtering
for single-cell RNA-seq data analysis.

Usage:
    python main.py --data_path /path/to/counts.txt \\
                   --gmt_path  /path/to/pathways.gmt \\
                   --label_file /path/to/labels.csv \\
                   --lambda_1 -0.001 --lambda_2 0.5 \\
                   --alpha 1.5 --h 15 --k 5 --s 20

Tunable hyperparameters: --lambda_1, --h, --k, --alpha
Fixed defaults:          --lambda_2 0.5, --s 20, --max_it 1000, --seed 42
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix, diags
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ==================== Data I/O ====================

def read_gmt(path):
    """Parse a GMT-format pathway file into a dict {pathway_name: [gene, ...]}."""
    with open(path, 'r') as f:
        lines = f.readlines()
    gmt = {}
    for line in lines:
        parts = line.strip().split('\t')
        gmt[parts[0]] = parts[2:]
    return gmt


def gmt_df(gmt):
    """Convert a GMT dict to a long-format DataFrame with columns ['gene', 'pathway']."""
    frames = []
    for pathway, genes in gmt.items():
        df = pd.DataFrame(genes, columns=['gene'])
        df['pathway'] = pathway
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


# ==================== Core Math ====================

def compute_laplacian(X, h=10, k=10):
    """
    Construct the graph Laplacian from a PCA-reduced KNN graph.

    Returns L (graph Laplacian) and X_reduced (PCA embedding).
    """
    X_reduced = PCA(n_components=h).fit_transform(X)
    A = kneighbors_graph(X_reduced, n_neighbors=k, mode='connectivity',
                         include_self=False).toarray()
    W = 0.5 * (A + A.T)
    D = np.diag(W.sum(axis=0))
    return D - W, X_reduced


def compute_norm(V, d, alpha):
    """Compute the exact Omega norm: (sum_j ||V_j * d_j||_2^alpha)^(1/alpha)."""
    d_dense = d.toarray() if sp.issparse(d) else d
    l2_norms = np.linalg.norm(V * d_dense, axis=0)
    return np.sum(l2_norms ** alpha) ** (1.0 / alpha)


def compute_omega(V, d, eta, Lambda_1, beta):
    """Compute the majorization-based approximation of Omega."""
    m, r = V.shape
    quadratic_term = sum(
        V[:, j].T @ diags((d[:, j].toarray().flatten() ** 2) / eta[j]) @ V[:, j]
        for j in range(r)
    )
    quadratic_term *= Lambda_1 / 2
    norm_eta = Lambda_1 / 2 * (np.sum(np.abs(eta) ** beta) ** (1.0 / beta))
    return quadratic_term + norm_eta


def compute_eta_inv_zeta(V, d, alpha, epsilon=1e-5):
    """
    Compute auxiliary variables eta (shape: r,) and inv_Zeta (shape: m x r)
    used in the majorization step for V.
    """
    m, r = V.shape
    eta = np.zeros(r)
    inv_Zeta = sp.lil_matrix((m, r))
    d_dense = d.toarray()

    for j in range(r):
        V_d_j = V[:, j] * d_dense[:, j]
        norm_j = np.linalg.norm(V_d_j, ord=2)
        eta_j_star = norm_j ** (2 - alpha)
        sum_norm = np.sum(np.linalg.norm(V * d_dense, axis=0) ** alpha)
        norm_factor = (max(sum_norm ** ((alpha - 1) / alpha), epsilon)
                       if sum_norm != 0 else epsilon)
        eta[j] = max(eta_j_star * norm_factor, epsilon)
        inv_Zeta[:, j] = (d[:, j].power(2) / eta[j]).toarray().flatten()

    return eta, inv_Zeta


# ==================== Update Steps ====================

def update_U_with_filter(V, X, W, U, max_it_U, pos_U, L, s=20, seed=42):
    """
    Update the cell factor matrix U with graph-filter projection.

    W : eigenvector matrix of the graph Laplacian (low-pass filter basis).
    """
    n, r = U.shape
    XV = X @ V
    VtV = V.T @ V
    U_out = U.copy()
    np.random.seed(seed)

    for _ in range(max_it_U):
        for col in np.random.permutation(r):
            if VtV[col, col] > 1e-12:
                inv_norm_sq = 1.0 / np.linalg.norm(V[:, col]) ** 2
                U_out[:, col] += inv_norm_sq * (XV[:, col] - U_out @ VtV[:, col])
                # Graph-filter projection
                U_out[:, col] = W @ (W.T @ U_out[:, col])
                if pos_U:
                    U_out[:, col] = np.maximum(U_out[:, col], 0)
                norm_u = np.linalg.norm(U_out[:, col])
                if norm_u > 0:
                    U_out[:, col] /= norm_u

    return U_out


def update_V(V, X, U, inv_Zeta, Lambda_1, max_it_V, pos_V, seed=42):
    """Update the pathway factor matrix V with the regularized coordinate-descent step."""
    n, m = X.shape
    r = V.shape[1]
    V_out = V.copy()
    XtU = X.T @ U
    UtU = U.T @ U
    np.random.seed(seed)

    for _ in range(max_it_V):
        for col in np.random.permutation(r):
            UtU_kk = UtU[col, col]
            if UtU_kk > 1e-12:
                tmp = XtU[:, col] - V_out @ UtU[:, col] + UtU_kk * V_out[:, col]
                for j in range(m):
                    V_out[j, col] = tmp[j] / (-Lambda_1 * n * m * inv_Zeta[j, col] + UtU_kk)

    return V_out


# ==================== Objective Function ====================

def display_cost_function(t, it0, X, U, V, Lambda_1, Lambda_2, L,
                           past_cost, d, alpha, beta, epsilon):
    """Compute and print the objective value; return relative change and updated past_cost."""
    n = X.shape[0]
    delta_cost = np.inf

    if t % it0 == 0:
        loss = np.linalg.norm(X - U @ V.T, 'fro') ** 2 / (2 * n)
        eta, _ = compute_eta_inv_zeta(V, d, alpha, epsilon)
        Omega_approx = compute_omega(V, d, eta, Lambda_1, beta)
        L_laplace = np.trace(U.T @ L @ U)
        cost_approx = loss - Lambda_1 * Omega_approx + Lambda_2 * L_laplace

        if t > it0:
            delta_cost = abs((past_cost - cost_approx) / past_cost)

        print(f"Iteration {t}: loss={loss:.4f}, "
              f"cost={cost_approx:.4f}, delta={delta_cost:.6f}")
        past_cost = cost_approx

    return delta_cost, past_cost


# ==================== Evaluation ====================

# ---------------------------------------------------------------------------
# CELL_TYPE_MAPPING: maps each cell-type label to the set of GMT pathway names
# that are considered "ground-truth" markers for that cell type.
#
# The example below is for the human pancreas dataset (GSE84133) using the
# c8 cell-type signature gene sets from MSigDB.  Pathway names must exactly
# match those in your GMT file.
#
# When applying to a different dataset, replace the keys and pathway sets
# according to the cell types and GMT gene sets you are using.
# Example for a PBMC dataset:
#
#   CELL_TYPE_MAPPING = {
#       'T_cell':   {"GOLDRATH_NAIVE_VS_EFF_CD8_TCELL_UP", ...},
#       'B_cell':   {"REACTOME_B_CELL_RECEPTOR_SIGNALING", ...},
#       'monocyte': {"HALLMARK_INTERFERON_GAMMA_RESPONSE", ...},
#   }
# ---------------------------------------------------------------------------
CELL_TYPE_MAPPING = {
    'acinar':      {"DESCARTES_FETAL_PANCREAS_ACINAR_CELLS",
                    "MURARO_PANCREAS_ACINAR_CELL"},
    'alpha':       {"MURARO_PANCREAS_ALPHA_CELL",
                    "VANGURP_PANCREATIC_ALPHA_CELL"},
    'beta':        {"MURARO_PANCREAS_BETA_CELL",
                    "VANGURP_PANCREATIC_BETA_CELL"},
    'delta':       {"MURARO_PANCREAS_DELTA_CELL",
                    "VANGURP_PANCREATIC_DELTA_CELL"},
    'ductal':      {"DESCARTES_FETAL_PANCREAS_DUCTAL_CELLS",
                    "MURARO_PANCREAS_DUCTAL_CELL"},
    'endothelial': {"DESCARTES_FETAL_PANCREAS_LYMPHATIC_ENDOTHELIAL_CELLS",
                    "DESCARTES_FETAL_PANCREAS_VASCULAR_ENDOTHELIAL_CELLS",
                    "MURARO_PANCREAS_ENDOTHELIAL_CELL"},
    'epsilon':     {"MURARO_PANCREAS_EPSILON_CELL"},
    'gamma':       {"VANGURP_PANCREATIC_GAMMA_CELL"},
}


def load_cell_labels(label_file):
    """Return a dict {cell_id: cell_type} from a two-column CSV."""
    df = pd.read_csv(label_file)
    return dict(zip(df["cell_id"].astype(str), df["cell_type"].astype(str)))


def compute_cell_types_topk_scores(U, pathways, cell_names, label_file, max_k=5):
    """
    For each cell type defined in CELL_TYPE_MAPPING, compute the fraction of
    cells whose top-k pathway scores include at least one ground-truth pathway,
    for k = 1 ... max_k.

    Returns
    -------
    scores_dict_by_k    : {k: {cell_type: score}}
    cell_type_counts    : {cell_type: n_cells}
    matched_counts_by_k : {k: {cell_type: n_matched}}
    overall_scores_by_k : {k: overall_score}
    total_matched_by_k  : {k: total_matched}
    total_cells         : int
    """
    cell_to_type = load_cell_labels(label_file)
    cell_groups = {ct: [] for ct in CELL_TYPE_MAPPING}

    for i, cell_name in enumerate(cell_names):
        ct = cell_to_type.get(str(cell_name), "Unknown")
        if ct in cell_groups:
            cell_groups[ct].append((i, cell_name))

    print("\nCell type distribution:")
    for ct, cells in cell_groups.items():
        if cells:
            print(f"  {ct}: {len(cells)} cells")

    scores_dict_by_k    = {k: {} for k in range(1, max_k + 1)}
    matched_counts_by_k = {k: {} for k in range(1, max_k + 1)}
    cell_type_counts    = {}
    total_cells         = 0
    total_matched_by_k  = {k: 0 for k in range(1, max_k + 1)}

    for ct, target_pathways in CELL_TYPE_MAPPING.items():
        cells = cell_groups[ct]
        if not cells:
            continue
        cell_type_counts[ct] = len(cells)
        total_cells += len(cells)

        for k in range(1, max_k + 1):
            matched = sum(
                1 for cell_idx, _ in cells
                if any(pathways[j] in target_pathways
                       for j in np.argsort(U[cell_idx])[-k:][::-1])
            )
            scores_dict_by_k[k][ct]    = matched / len(cells)
            matched_counts_by_k[k][ct] = matched
            total_matched_by_k[k]     += matched

    overall_scores_by_k = {
        k: total_matched_by_k[k] / total_cells if total_cells > 0 else 0.0
        for k in range(1, max_k + 1)
    }
    return (scores_dict_by_k, cell_type_counts, matched_counts_by_k,
            overall_scores_by_k, total_matched_by_k, total_cells)


def print_topk_cells_by_type(U, pathways, cell_names, label_file,
                              num_cells=3, max_k=5):
    """Print the top-k pathways for the first num_cells cells of each cell type."""
    cell_to_type = load_cell_labels(label_file)
    cell_groups = {ct: [] for ct in CELL_TYPE_MAPPING}

    for i, cell_name in enumerate(cell_names):
        ct = cell_to_type.get(str(cell_name), "Unknown")
        if ct in cell_groups:
            cell_groups[ct].append((i, cell_name))

    for ct, target_pathways in CELL_TYPE_MAPPING.items():
        cells = cell_groups[ct]
        if not cells:
            continue
        print(f"\n{'='*80}")
        print(f"Top-{max_k} pathways for the first "
              f"{min(num_cells, len(cells))} '{ct}' cells:")
        print(f"{'='*80}")
        for cell_idx, cell_name in cells[:num_cells]:
            scores = U[cell_idx]
            topk_idx = np.argsort(scores)[-max_k:][::-1]
            print(f"\n  {cell_name} ({ct}):")
            for rank, j in enumerate(topk_idx, 1):
                marker = "v" if pathways[j] in target_pathways else " "
                print(f"    [{marker}] {rank}. {pathways[j]}: {scores[j]:.4f}")


def evaluate_clustering(U, cell_ids, label_file):
    """
    Run KMeans on U and evaluate against ground-truth labels.

    Returns ARI, NMI, RI.
    """
    meta_df = pd.read_csv(label_file)
    meta_df.columns = ["cell_id", "cell_type"]
    id2type = dict(zip(meta_df["cell_id"].astype(str), meta_df["cell_type"].astype(str)))
    labels_true = np.array([id2type.get(str(cid), "Unknown") for cid in cell_ids])

    known_mask = labels_true != "Unknown"
    unique_labels = [l for l in np.unique(labels_true[known_mask]) if l != "Unknown"]
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        print("  Warning: fewer than 2 clusters found; skipping clustering evaluation.")
        return 0.0, 0.0, 0.0

    U_scaled = StandardScaler().fit_transform(U)
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=42,
                            n_init="auto").fit_predict(U_scaled)

    ari = adjusted_rand_score(labels_true[known_mask], cluster_labels[known_mask])
    nmi = normalized_mutual_info_score(labels_true[known_mask], cluster_labels[known_mask])
    ri  = rand_score(labels_true[known_mask], cluster_labels[known_mask])

    print(f"  Clusters : {n_clusters}  |  Subtypes: {unique_labels}")
    print(f"  ARI: {ari:.4f}  |  NMI: {nmi:.4f}  |  RI: {ri:.4f}")
    return ari, nmi, ri


# ==================== Main Model ====================

def run_model(X, d, pathways, lambda_1, lambda_2, s, cell_names, label_file,
              seed=42, h=10, k=10, alpha=0.5, max_it=1000):
    """
    Run the pathway-guided NMF model with a single set of hyperparameters.

    Parameters
    ----------
    X          : ndarray, shape (n_cells, n_genes)
    d          : csr_matrix, shape (n_genes, n_pathways)
                 Gene-pathway indicator matrix (1 = gene NOT in pathway).
    pathways   : array-like of str, length n_pathways
    lambda_1   : float  Pathway regularization weight (typically negative).
    lambda_2   : float  Graph regularization weight.
    s          : int    Number of eigenvectors retained for the graph filter.
    cell_names : list[str]
    label_file : str    Path to cell-type label CSV (columns: cell_id, cell_type).
    seed       : int    Random seed.
    h          : int    Number of PCA components used to build the KNN graph.
    k          : int    Number of nearest neighbors for graph construction.
    alpha      : float  Norm exponent in the Omega regularizer (0 < alpha < 2).
    max_it     : int    Maximum number of outer iterations.

    Returns
    -------
    U, V, scores_dict_by_k, overall_scores_by_k, ari, nmi, ri
    """
    n, m = X.shape
    r    = d.shape[1]
    beta    = alpha / (2 - alpha)
    epsilon = 1e-5

    max_it_U = 3
    max_it_V = 3
    pos_U    = True
    pos_V    = True   # kept for extensibility; not applied in update_V currently

    if sparse.issparse(X):
        X = X.toarray()

    # Initialization
    np.random.seed(seed)
    U = np.abs(np.random.randn(n, r))
    V = np.abs(np.random.randn(m, r))

    # Graph Laplacian and low-pass filter basis
    L, _ = compute_laplacian(X, h=h, k=k)
    eigenvalues, eigenvectors = eigh(L, check_finite=False)
    W = eigenvectors[:, np.argsort(eigenvalues)[:s]]   # s smallest eigenvectors

    print(f"\nParameters — lambda_1={lambda_1}, lambda_2={lambda_2}, "
          f"s={s}, h={h}, k={k}, alpha={alpha}")

    # Optimization loop
    t         = 1
    past_cost = 1e10
    count     = 0
    min_delta = 0.05

    while t <= max_it:
        eta, inv_Zeta = compute_eta_inv_zeta(V, d, alpha, epsilon)
        V = update_V(V, X, U, inv_Zeta, lambda_1, max_it_V, pos_V, seed=seed)
        U = update_U_with_filter(V, X, W, U, max_it_U, pos_U, L, s=s, seed=seed)
        delta_cost, past_cost = display_cost_function(
            t, 1, X, U, V, lambda_1, lambda_2, L, past_cost, d, alpha, beta, epsilon)

        count = 0 if delta_cost > min_delta else count + 1
        if count == 5:
            print(f"Converged at iteration {t}.")
            break
        t += 1

    if t > max_it:
        print(f"Reached maximum iterations ({max_it}).")

    # Evaluation — top-k pathway matching
    (scores_dict_by_k, cell_type_counts, matched_counts_by_k,
     overall_scores_by_k, total_matched_by_k, total_cells) = \
        compute_cell_types_topk_scores(U, pathways, cell_names, label_file, max_k=5)

    print(f"\n{'='*80}")
    print("Top-k Pathway Matching Scores")
    print(f"{'='*80}")
    for k_val in range(1, 6):
        print(f"\nTop-{k_val}:")
        for ct in CELL_TYPE_MAPPING:
            if ct in scores_dict_by_k[k_val]:
                cnt     = cell_type_counts.get(ct, 0)
                matched = matched_counts_by_k[k_val].get(ct, 0)
                print(f"  {ct:15s}: {scores_dict_by_k[k_val][ct]:.4f}"
                      f"  ({matched}/{cnt})")
        print(f"  {'Overall':15s}: {overall_scores_by_k[k_val]:.4f}"
              f"  ({total_matched_by_k[k_val]}/{total_cells})")

    print_topk_cells_by_type(U, pathways, cell_names, label_file,
                              num_cells=3, max_k=5)

    # Evaluation — clustering
    print(f"\n{'='*80}")
    print("Clustering Evaluation")
    print(f"{'='*80}")
    ari, nmi, ri = evaluate_clustering(U, cell_names, label_file)

    return U, V, scores_dict_by_k, overall_scores_by_k, ari, nmi, ri


# ==================== Preprocessing ====================

def preprocess(data_path, gmt_path):
    """
    Load and preprocess count data and pathway annotations.

    Steps: deduplication -> cell/gene filtering -> MT-gene filter ->
           library-size normalization -> log1p -> build gene-pathway matrix d.

    Returns X (ndarray), d (csr_matrix), filtered_pathways (ndarray), cell_names (list).
    """
    import anndata
    import scanpy as sc

    gmt = read_gmt(gmt_path)
    DF  = gmt_df(gmt)

    gene_cell_df = pd.read_csv(data_path, sep='\t', index_col=0)
    X_raw      = gene_cell_df.values.T          # shape: (n_cells, n_genes)
    cell_names = gene_cell_df.columns.tolist()

    adata = anndata.AnnData(csr_matrix(X_raw))
    adata.var_names = gene_cell_df.index
    adata.obs_names = cell_names
    print(f"Raw data  : {adata.shape[0]} cells x {adata.shape[1]} genes")

    adata = adata[:, ~adata.var_names.duplicated(keep='first')]
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'],
                                percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cell_names = adata.obs_names.tolist()
    print(f"Processed : {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Build gene-pathway indicator matrix d:
    #   d[i, j] = 1  if gene i is NOT in pathway j  (penalized by Omega)
    #   d[i, j] = 0  if gene i IS  in pathway j      (not penalized)
    df_gene  = pd.DataFrame(range(len(adata.var_names)),
                             index=adata.var_names, columns=['row'])
    DF_filt  = DF[DF['gene'].isin(set(df_gene.index))]
    pathways = np.unique(DF_filt['pathway'])
    n_pw     = len(pathways)
    print(f"Filtered pathways: {n_pw}")

    df_pw = pd.DataFrame(range(n_pw), index=pathways, columns=['col'])
    rows  = df_gene.loc[DF_filt['gene'].dropna(), 'row']
    cols  = df_pw.loc[DF_filt['pathway'].dropna(), 'col']

    ones = coo_matrix(np.ones((len(adata.var_names), n_pw)))
    d    = coo_matrix((np.ones(len(rows)), (rows.values, cols.values)),
                       shape=(len(adata.var_names), n_pw))
    d    = (ones - d).tocsr()

    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    return X, d, pathways, cell_names


# ==================== CLI Entry Point ====================

def parse_args():
    p = argparse.ArgumentParser(
        description="Pathway-guided NMF with graph filtering for scRNA-seq.")
    # Paths
    p.add_argument("--data_path",  required=True,
                   help="Gene-cell count matrix (.txt, tab-separated, genes as rows)")
    p.add_argument("--gmt_path",   required=True,
                   help="GMT pathway file")
    p.add_argument("--label_file", required=True,
                   help="Cell-type label CSV (columns: cell_id, cell_type)")
    p.add_argument("--output_dir", default="./results",
                   help="Directory to save U.npy, V.npy, and metrics.csv")
    # Tunable hyperparameters
    p.add_argument("--lambda_1", type=float, default=-0.001,
                   help="Pathway regularization coefficient (should be negative) [tunable]")
    p.add_argument("--alpha",    type=float, default=1.5,
                   help="Norm exponent in the Omega regularizer (0 < alpha < 2) [tunable]")
    p.add_argument("--h",        type=int,   default=15,
                   help="Number of PCA components for KNN graph construction [tunable]")
    p.add_argument("--k",        type=int,   default=5,
                   help="Number of nearest neighbors for graph construction [tunable]")
    # Fixed defaults
    p.add_argument("--lambda_2", type=float, default=0.5,
                   help="Graph (Laplacian) regularization coefficient")
    p.add_argument("--s",        type=int,   default=20,
                   help="Number of eigenvectors retained in the graph filter")
    p.add_argument("--max_it",   type=int,   default=1000,
                   help="Maximum number of outer iterations")
    p.add_argument("--seed",     type=int,   default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args       = parse_args()
    start_time = time.time()

    print("=" * 60)
    print("Loading and preprocessing data ...")
    print("=" * 60)
    X, d, pathways, cell_names = preprocess(args.data_path, args.gmt_path)
    print(f"X shape: {X.shape} | d shape: {d.shape}")

    print("\n" + "=" * 60)
    print("Running model ...")
    print("=" * 60)
    U, V, scores_dict_by_k, overall_scores_by_k, ari, nmi, ri = run_model(
        X, d, pathways,
        lambda_1   = args.lambda_1,
        lambda_2   = args.lambda_2,
        s          = args.s,
        cell_names = cell_names,
        label_file = args.label_file,
        seed       = args.seed,
        h          = args.h,
        k          = args.k,
        alpha      = args.alpha,
        max_it     = args.max_it,
    )

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "U.npy"), U)
    np.save(os.path.join(args.output_dir, "V.npy"), V)

    summary = {
        "lambda_1": args.lambda_1, "lambda_2": args.lambda_2,
        "alpha": args.alpha, "h": args.h, "k": args.k, "s": args.s,
        "ARI": ari, "NMI": nmi, "RI": ri,
        **{f"overall_top{k}": overall_scores_by_k[k] for k in range(1, 6)},
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output_dir, "metrics.csv"), index=False)

    elapsed = time.time() - start_time
    print(f"\nFinished. Runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results saved to : {args.output_dir}")


if __name__ == "__main__":
    main()