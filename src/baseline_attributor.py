"""
Attributore Baseline Classico per il Dominio QKD (Fase 1 - Parte B)
------------------------------------------------------------------
Questo script addestra un regressore multi-output classico sulle feature osservabili 
della rete quantistica per stimare il vettore delle frazioni causali dell'errore:
    y = (f_attacco, f_rumore, f_rivelatore)

Calcola l'Errore Medio Assoluto (MAE) sui tre protocolli di split prescritti dalla Roadmap:
  1. Random Split (Interpolazione casuale 80/20)
  2. Split per Seme (Seed Split - Stabilità stocastica)
  3. Split per Cella (Cell Split / Held-out - Generalizzazione per RQ2)
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

def load_and_prepare_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Impossibile trovare il file dataset: {file_path}")

    df = pd.read_csv(file_path)

    feature_candidates = {
        'qber': ['qber', 'qber_tot'],
        'qber_rect': ['qber_rect', 'qber_x'],
        'qber_diag': ['qber_diag', 'qber_z'],
        'basis_asymmetry': ['basis_asymmetry'],
        'sift_rate': ['sift_rate', 'sifting_rate'],
        'loss_rate': ['loss_rate', 'channel_loss_rate']
    }

    feature_cols = []
    for std_name, aliases in feature_candidates.items():
        found = False
        for alias in aliases:
            if alias in df.columns:
                feature_cols.append(alias)
                found = True
                break

    target_candidates = {
        'f_eve': ['f_eve', 'f_attacco', 'f_attack'],
        'f_canale': ['f_canale', 'f_rumore', 'f_noise'],
        'f_rivelatore': ['f_rivelatore', 'f_detector']
    }

    target_cols = []
    for std_name, aliases in target_candidates.items():
        found = False
        for alias in aliases:
            if alias in df.columns:
                target_cols.append(alias)
                found = True
                break

    if 'basis_asymmetry' not in df.columns and 'qber_rect' in df.columns and 'qber_diag' in df.columns:
        df['basis_asymmetry'] = (df['qber_rect'] - df['qber_diag']).abs()
        if 'basis_asymmetry' not in feature_cols:
            feature_cols.append('basis_asymmetry')

    return df, feature_cols, target_cols

def evaluate_random_split(df, feature_cols, target_cols, seed=42):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(df))
    split_point = int(0.8 * len(df))

    train_idx = shuffled_indices[:split_point]
    test_idx = shuffled_indices[split_point:]

    X_train, y_train = df.iloc[train_idx][feature_cols], df.iloc[train_idx][target_cols]
    X_test, y_test = df.iloc[test_idx][feature_cols], df.iloc[test_idx][target_cols]

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_total = mean_absolute_error(y_test, y_pred)
    mae_per_target = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

    return mae_total, dict(zip(target_cols, mae_per_target))

def evaluate_seed_split(df, feature_cols, target_cols, seed_col='seed', test_seed=5):
    """Protocollo 2: Seed Split (Train su semi != test_seed, Test su test_seed)."""
    if seed_col not in df.columns:
        return None, {}

    train_df = df[df[seed_col] != test_seed]
    test_df = df[df[seed_col] == test_seed]

    if len(test_df) == 0:
        unique_seeds = df[seed_col].unique()
        test_seed = unique_seeds[-1]
        train_df = df[df[seed_col] != test_seed]
        test_df = df[df[seed_col] == test_seed]

    X_train, y_train = train_df[feature_cols], train_df[target_cols]
    X_test, y_test = test_df[feature_cols], test_df[target_cols]

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_total = mean_absolute_error(y_test, y_pred)
    mae_per_target = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    
    return mae_total, dict(zip(target_cols, mae_per_target))


def evaluate_cell_split(df, feature_cols, target_cols, test_ratio=0.2, seed=42):
    """Protocollo 3: Cell Split / Held-out (Esclude intere celle fisiche per RQ2)."""
    cell_id_cols = [c for c in ['intercept_rate', 'length_km', 'depolar_total', 'cell_id'] if c in df.columns]
    
    if not cell_id_cols:
        df['cell_id'] = np.arange(len(df)) // 100
        cell_id_cols = ['cell_id']

    unique_cells = df[cell_id_cols].drop_duplicates().reset_index(drop=True)
    
    np.random.seed(seed)
    n_test_cells = max(1, int(len(unique_cells) * test_ratio))
    test_cell_indices = np.random.choice(len(unique_cells), size=n_test_cells, replace=False)
    
    test_cells = unique_cells.iloc[test_cell_indices]
    
    merged = df.merge(
        test_cells.assign(is_test=True), on=cell_id_cols, how="left"
    )
    is_test_mask = merged["is_test"].fillna(False).astype(bool)

    train_df = merged[~is_test_mask]
    test_df = merged[is_test_mask]

    X_train, y_train = train_df[feature_cols], train_df[target_cols]
    X_test, y_test = test_df[feature_cols], test_df[target_cols]

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_total = mean_absolute_error(y_test, y_pred)
    mae_per_target = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    
    return mae_total, dict(zip(target_cols, mae_per_target)), len(train_df), len(test_df)

def evaluate_binary_classification(df, feature_cols):
    """Variante classificatoria binaria (Attacco vs No Attacco) per confronto con letteratura."""
    attack_col = None
    for col in ['intercept_rate', 'f_eve', 'f_attacco', 'f_attack', 'binary_label']:
        if col in df.columns:
            attack_col = col
            break

    if attack_col is None:
        return None, None

    if attack_col == 'binary_label':
        y_binary = df['binary_label']
    else:
        y_binary = (df[attack_col] > 0).astype(int)

    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(df))
    split = int(0.8 * len(df))
    
    X_train, y_train = df.iloc[shuffled_idx[:split]][feature_cols], y_binary.iloc[shuffled_idx[:split]]
    X_test, y_test = df.iloc[shuffled_idx[split:]][feature_cols], y_binary.iloc[shuffled_idx[split:]]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Attributore Baselina Classico QKD (Fase 1 - Parte B)")
    parser.add_argument("--data", type=str, default="qber_grid.csv", help="Percorso del file contenente la griglia QBER")
    args = parser.parse_args()

    file_path = args.data
    if not os.path.exists(file_path):
        for candidate in ["qber_grid.csv", "qber-grid.csv", "data/qber_grid.csv"]:
            if os.path.exists(candidate):
                file_path = candidate
                break

    print(f"===========================================================")
    print(f"  ATTRIBUTORE BASELINE CLASSICO (QKD - Fase 1 / Parte B)")
    print(f"===========================================================")
    print(f"Caricamento dataset da: {file_path}")

    df, feature_cols, target_cols = load_and_prepare_data(file_path)

    print(f"Dataset caricato con successo: {len(df)} righe.")
    print(f"Feature osservabili (X): {feature_cols}")
    print(f"Target causali (y):      {target_cols}\n")

    # 1. Random Split
    mae_random, mae_comp_random = evaluate_random_split(df, feature_cols, target_cols)
    print("--- 1. RANDOM SPLIT (Interpolazione Casuale) ---")
    print(f"  MAE Complessivo: {mae_random:.4f}")
    for k, v in mae_comp_random.items():
        print(f"    - MAE ({k}): {v:.4f}")

    # 2. Seed Split
    mae_seed, mae_comp_seed = evaluate_seed_split(df, feature_cols, target_cols)
    print("\n--- 2. SEED SPLIT (Stabilità al Rumore Stocastico) ---")
    if mae_seed is not None:
        print(f"  MAE Complessivo: {mae_seed:.4f}")
        for k, v in mae_comp_seed.items():
            print(f"    - MAE ({k}): {v:.4f}")


    # 3. Cell Split / Held-out
    mae_cell, mae_comp_cell, n_tr, n_te = evaluate_cell_split(df, feature_cols, target_cols)
    print("\n--- 3. CELL SPLIT / HELD-OUT (Generalizzazione - Termine di paragone RQ2) ---")
    print(f"  Righe Train: {n_tr} | Righe Test (Celle escluse): {n_te}")
    print(f"  MAE Complessivo: {mae_cell:.4f}")
    for k, v in mae_comp_cell.items():
        print(f"    - MAE ({k}): {v:.4f}")

    # 4. Variante Binaria
    acc_bin, f1_bin = evaluate_binary_classification(df, feature_cols)
    if acc_bin is not None:
        print("\n--- 4. VARIANTE CLASSIFICATORIA BINARIA (Attacco vs No Attacco) ---")
        print(f"  Accuracy: {acc_bin:.4f} | F1-Score (Macro): {f1_bin:.4f}")


    print("\n===========================================================")
    print("  RIEPILOGO BENCHMARK BASELINE (Fase 1 - Parte B)")
    print("===========================================================")
    print(f"  Random Split MAE  : {mae_random:.4f}")
    if mae_seed:
        print(f"  Seed Split MAE    : {mae_seed:.4f}")
    print(f"  Cell Split MAE    : {mae_cell:.4f}  <-- BENCHMARK PER RQ2")
    if acc_bin:
        print(f"  Accuracy Binaria  : {acc_bin:.4f}")
    print("===========================================================")

if __name__ == "__main__":
    main()                            

                   
