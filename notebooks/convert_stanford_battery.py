"""
convert_stanford_battery.py
============================
Converte todos os arquivos .mat do dataset Stanford EV Real-Driving
(INR21700-M50T) para .csv, detectando automaticamente o tipo de cada arquivo.

Tipos reconhecidos:
  - capacity_test_crate_data  →  CAPACI_1.MAT  (C-rate characterization, formato único)
  - capacity_test_10cells     →  CAPACI_2.MAT  (diagnósticos periódicos, 15 diag × 10 células)
  - hppc_test_10cells         →  HPPC (15 diag × 10 células)
  - EIS_test_10cells          →  EIS  (15 diag × 10 células, 19 freq × 3 SOC)

Uso:
  python convert_stanford_battery.py --input_dir ./data/raw --output_dir ./data/csv

  # Ou converter um arquivo específico:
  python convert_stanford_battery.py --input_dir ./data/raw --output_dir ./data/csv --file CAPACI_2.MAT
"""

import argparse
import os
import glob
import scipy.io as sio
import pandas as pd
import numpy as np


# ── Detecção de tipo ──────────────────────────────────────────────────────────

def detect_type(mat: dict) -> str:
    keys = set(mat.keys()) - {'__header__', '__version__', '__globals__'}
    if 'capacity_test_crate_data' in keys:
        return 'crate'
    if 'col_cell_label' in keys and 'freq' in keys:
        return 'eis'
    if 'col_cell_label' in keys and 'vcell' in keys and 'curr' in keys:
        return 'capacity_or_hppc'
    return 'unknown'


# ── Conversores ───────────────────────────────────────────────────────────────

def convert_crate(mat: dict, output_dir: str, stem: str):
    """
    CAPACI_1: capacity_test_crate_data com sub-chaves chg_0_050C, dis_0_025C, etc.
    Cada sub-chave tem arrays 1-D: vcell, curr, time.
    Saída: um CSV por sub-teste  →  stem_chg_0_050C.csv, etc.
    """
    data = mat['capacity_test_crate_data']
    saved = []
    for test_name, arrays in data.items():
        rows = pd.DataFrame({
            'time_s':    arrays['time'],
            'voltage_V': arrays['vcell'],
            'current_A': arrays['curr'],
        })
        # decode se for bytes
        tname = test_name.decode() if isinstance(test_name, bytes) else test_name
        out_path = os.path.join(output_dir, f"{stem}__{tname}.csv")
        rows.to_csv(out_path, index=False)
        saved.append(out_path)
        print(f"  [crate] {tname}: {len(rows)} linhas -> {os.path.basename(out_path)}")
    return saved


def convert_multi_cell(mat: dict, output_dir: str, stem: str):
    """aa
    CAPACI_2 / HPPC: matriz (15 diag × 10 células) com arrays de séries temporais.
    Colunas: diag_number, cell, time_s, voltage_V, current_A
    Saída: um CSV único  →  stem.csv
    """
    cell_labels  = [str(c) for c in mat['col_cell_label']]
    diag_numbers = mat['row_diag_number']            # array 1-D de ints
    time_mat     = mat['time']
    vcell_mat    = mat['vcell']
    curr_mat     = mat['curr']

    rows = []
    n_diag, n_cell = time_mat.shape

    for di in range(n_diag):
        for ci in range(n_cell):
            t = time_mat[di][ci]
            v = vcell_mat[di][ci]
            c = curr_mat[di][ci]

            # Células que não chegaram a esse diagnóstico ficam vazias
            if t is None or (hasattr(t, '__len__') and len(t) == 0):
                continue

            t = np.asarray(t).ravel()
            v = np.asarray(v).ravel()
            c = np.asarray(c).ravel()

            n = len(t)
            chunk = pd.DataFrame({
                'diag_number': np.full(n, int(diag_numbers[di])),
                'cell':        np.full(n, cell_labels[ci]),
                'time_s':      t,
                'voltage_V':   v,
                'current_A':   c,
            })
            rows.append(chunk)

    df = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(output_dir, f"{stem}.csv")
    df.to_csv(out_path, index=False)
    print(f"  [multi-cell] {df['cell'].nunique()} celulas, "
          f"{df['diag_number'].nunique()} diags, "
          f"{len(df)} linhas -> {os.path.basename(out_path)}")
    return [out_path]


def convert_eis(mat: dict, output_dir: str, stem: str):
    """
    EIS: re_z e im_z com shape (n_freq, 3_soc) por célula/diag.
    Colunas: diag_number, cell, soc_pct, freq_Hz, re_z_ohm, im_z_ohm
    Saída: um CSV único  →  stem.csv
    """
    cell_labels  = [str(c) for c in mat['col_cell_label']]
    diag_numbers = mat['row_diag_number']
    soc_levels   = mat['soc_level']          # [20, 50, 80]
    re_mat       = mat['re_z']
    im_mat       = mat['im_z']

    # freq pode existir ou não — caso não exista, usamos índice
    has_freq = 'freq' in mat
    freq_mat = mat.get('freq', None)

    rows = []
    n_diag, n_cell = re_mat.shape

    for di in range(n_diag):
        for ci in range(n_cell):
            re = re_mat[di][ci]
            im = im_mat[di][ci]

            re = np.asarray(re)
            im = np.asarray(im)

            # Células sem dados ficam como array escalar vazio (ndim==0 ou size==0)
            if re.ndim < 2 or re.size == 0:
                continue

            if has_freq and freq_mat[di][ci] is not None:
                f_raw = np.asarray(freq_mat[di][ci])
                # freq tem shape (n_freq, n_soc) — as colunas são iguais, pegar a primeira
                freq = f_raw[:, 0] if f_raw.ndim == 2 else f_raw.ravel()
            else:
                freq = np.arange(re.shape[0], dtype=float)

            n_freq = re.shape[0]

            for si, soc in enumerate(soc_levels):
                chunk = pd.DataFrame({
                    'diag_number': np.full(n_freq, int(diag_numbers[di])),
                    'cell':        np.full(n_freq, cell_labels[ci]),
                    'soc_pct':     np.full(n_freq, int(soc)),
                    'freq_Hz':     freq,
                    're_z_ohm':    re[:, si],
                    'im_z_ohm':    im[:, si],
                })
                rows.append(chunk)

    df = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(output_dir, f"{stem}.csv")
    df.to_csv(out_path, index=False)
    print(f"  [EIS] {df['cell'].nunique()} celulas, "
          f"{df['diag_number'].nunique()} diags, "
          f"{df['soc_pct'].nunique()} SOCs, "
          f"{len(df)} linhas -> {os.path.basename(out_path)}")
    return [out_path]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def convert_file(path: str, output_dir: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    print(f"\nConvertendo: {os.path.basename(path)}")

    mat = sio.loadmat(path, simplify_cells=True)
    kind = detect_type(mat)

    if kind == 'crate':
        return convert_crate(mat, output_dir, stem)
    elif kind == 'eis':
        return convert_eis(mat, output_dir, stem)
    elif kind == 'capacity_or_hppc':
        return convert_multi_cell(mat, output_dir, stem)
    else:
        print(f"  [AVISO] Tipo desconhecido - chaves: {set(mat.keys()) - {'__header__','__version__','__globals__'}}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def _long_path(p: str) -> str:
    r"""Adiciona prefixo \\?\ para suportar caminhos longos no Windows (>260 chars)."""
    p = os.path.abspath(p)
    if os.name == 'nt' and not p.startswith('\\\\?\\'):
        p = '\\\\?\\' + p
    return p


def main():
    parser = argparse.ArgumentParser(description='Converte dataset Stanford battery .mat → .csv')
    parser.add_argument('--input_dir',  required=True,  help='Pasta com os arquivos .mat')
    parser.add_argument('--output_dir', required=True,  help='Pasta de destino dos .csv')
    parser.add_argument('--file',       default=None,   help='Converter apenas este arquivo (opcional)')
    args = parser.parse_args()

    input_dir  = _long_path(args.input_dir)
    output_dir = _long_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.file:
        targets = [os.path.join(input_dir, args.file)]
    else:
        targets = sorted(
            glob.glob(os.path.join(input_dir, '*.mat')) +
            glob.glob(os.path.join(input_dir, '*.MAT'))
        )

    if not targets:
        print('Nenhum arquivo .mat encontrado.')
        print(f'  Diretório procurado: {input_dir}')
        return

    all_outputs = []
    for path in targets:
        all_outputs.extend(convert_file(path, output_dir))

    print(f'\nPronto. {len(all_outputs)} CSV(s) gerado(s) em: {args.output_dir}')


if __name__ == '__main__':
    main()