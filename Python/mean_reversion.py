"""Pipeline completo de treinamento e exportação.

Executa, em sequência: carga dos dados → features → imagens ilustrativas
da rotulagem → clusterização de regimes (K-Means) → rotulagem → treino
CatBoost (principal + meta) por cluster → seleção do melhor por R² →
gráfico de desempenho → exportação ONNX + include MQL5.

Uso:
    cd Python
    python mean_reversion.py                     # EURGBP_H1, método 1
    python mean_reversion.py --method 4          # outro método de rotulagem
    python mean_reversion.py --symbol AUDNZD_H1  # outro par (CSV em Python/)
"""

import argparse
import math
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from data_lib import find_symbol_csv, get_features, load_prices
from export_lib import export_model_to_ONNX
from labeling_lib import (
    get_labels_filter,
    get_labels_filter_bidirectional,
    get_labels_mean_reversion,
    get_labels_mean_reversion_multi,
    get_labels_mean_reversion_v,
    get_labels_multiple_filters,
)
from tester_lib import tester
from viz_lib import (BASELINE, COLOR_PRICE, COLOR_STRATEGY, COLOR_ZONE,
                     apply_style, save_fig)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IMG_DIR = os.path.join(REPO_ROOT, "Imagens")

hyper_params = {
    'symbol': 'EURGBP_H1',
    # Por padrão exporta para a pasta MQL5 do próprio repositório; para exportar
    # direto para o MetaTrader 5, substitua pelo caminho da pasta de dados do MT5
    # (Arquivo → Abrir Pasta de Dados → MQL5/Include/Mean_Reversion/).
    'export_path': os.path.join(REPO_ROOT, 'MQL5', 'Include', 'Mean_Reversion') + os.sep,
    'model_number': 0,
    'markup': 0.00010,
    'stop_loss':  0.02000,
    'take_profit': 0.00200,
    'periods': [i for i in range(5, 300, 30)],
    'periods_meta': [10],
    'backward': datetime(2000, 1, 1),
    'forward': datetime(2021, 1, 1),
    'n_clusters': 10,
    'rolling': 200,
}

# Métodos de rotulagem disponíveis (selecionáveis via --method)
LABELING_METHODS = {
    1: ('Filtro Savitzky-Golay com bandas de quantis',
        lambda d: get_labels_filter(
            d, rolling=hyper_params['rolling'], quantiles=[0.45, 0.55], polyorder=3)),
    2: ('Múltiplos filtros com quantis dinâmicos',
        lambda d: get_labels_multiple_filters(
            d, rolling_periods=[50, 100, 200], quantiles=[0.45, 0.55], window=100, polyorder=3)),
    3: ('Filtro bidirecional',
        lambda d: get_labels_filter_bidirectional(
            d, rolling1=50, rolling2=200, quantiles=[0.45, 0.55], polyorder=3)),
    4: ('Reversão à média com restrições de lucratividade',
        lambda d: get_labels_mean_reversion(
            d, markup=hyper_params['markup'], min_l=1, max_l=15, rolling=0.5,
            quantiles=[0.45, 0.55], method='spline', shift=0)),
    5: ('Reversão à média multi-janela',
        lambda d: get_labels_mean_reversion_multi(
            d, markup=hyper_params['markup'], min_l=1, max_l=15,
            windows=[0.2, 0.3, 0.5], quantiles=[0.45, 0.55])),
    6: ('Rotulagem ajustada por volatilidade',
        lambda d: get_labels_mean_reversion_v(
            d, markup=hyper_params['markup'], min_l=1, max_l=15, rolling=0.2,
            quantiles=[0.45, 0.55], method='spline', shift=0, volatility_window=100)),
}


def get_prices():
    return load_prices(find_symbol_csv(hyper_params['symbol']))


def make_features(data):
    return get_features(data, hyper_params['periods'], hyper_params['periods_meta'])


# ──────────────────────────────────────────────────────────────────────────────
# Imagens ilustrativas da rotulagem (README)
# ──────────────────────────────────────────────────────────────────────────────

def plot_filter_illustration(dataset):
    """Filtro Savitzky-Golay com bandas de quantis — 2 meses (até o mês final).

    As bandas 20%/80% são ilustrativas (mais fáceis de enxergar);
    o treinamento usa os quantis 45%/55%.
    """
    end = dataset.index[-1].to_period('M').to_timestamp()
    sample = dataset.loc[end - pd.DateOffset(months=2):end]
    rolling = hyper_params['rolling']
    smoothed = savgol_filter(sample['close'].values, window_length=rolling, polyorder=3)
    diff = sample['close'].values - smoothed
    q_low, q_high = np.quantile(diff, 0.20), np.quantile(diff, 0.80)

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(sample.index, sample['close'], color=COLOR_PRICE, linewidth=0.7,
            alpha=0.9, label='Preço de fechamento')
    ax.plot(sample.index, smoothed, color=COLOR_STRATEGY, linewidth=1.6,
            label=f'Savitzky-Golay (janela={rolling})')
    ax.plot(sample.index, smoothed + q_high, color=COLOR_ZONE, linestyle='--',
            linewidth=1, label='Zona de reversão (quantis 20%/80%, ilustrativos)')
    ax.plot(sample.index, smoothed + q_low, color=COLOR_ZONE, linestyle='--', linewidth=1)
    ax.fill_between(sample.index, smoothed + q_low, smoothed + q_high,
                    color=COLOR_ZONE, alpha=0.06, linewidth=0)
    ax.set_title('Filtro Savitzky-Golay com bandas de quantis — '
                 f'{sample.index[0]:%m/%Y} a {sample.index[-1]:%m/%Y}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(IMG_DIR, '01_Quantiles.png'))
    plt.close(fig)


def plot_spline_illustration(dataset):
    """Suavização spline com bandas de quantis — 1 ano (até o mês final)."""
    end = dataset.index[-1].to_period('M').to_timestamp()
    sample = dataset.loc[end - pd.DateOffset(years=1):end]
    x = np.arange(sample.shape[0])
    y = sample['close'].values
    spl = UnivariateSpline(x, y, k=3, s=0.5)
    yHat = spl(np.linspace(x.min(), x.max(), x.shape[0]))
    diff = y - yHat
    q_low, q_high = np.quantile(diff, 0.20), np.quantile(diff, 0.80)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sample.index, y, color=COLOR_PRICE, linewidth=0.6, alpha=0.85,
            label='Preço de fechamento')
    ax.plot(sample.index, yHat, color=COLOR_STRATEGY, linewidth=1.6,
            label='Spline suavizado (s=0.5)')
    ax.plot(sample.index, yHat + q_high, color=COLOR_ZONE, linestyle='--',
            linewidth=1, label='Zona de reversão (quantis 20%/80%, ilustrativos)')
    ax.plot(sample.index, yHat + q_low, color=COLOR_ZONE, linestyle='--', linewidth=1)
    ax.fill_between(sample.index, yHat + q_low, yHat + q_high,
                    color=COLOR_ZONE, alpha=0.06, linewidth=0)
    ax.set_title('Suavização spline com bandas de quantis — '
                 f'{sample.index[0]:%m/%Y} a {sample.index[-1]:%m/%Y}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(IMG_DIR, '02_Spline.png'))
    plt.close(fig)


def plot_clusters(data):
    """Mapa dos regimes de mercado — recorte de 2 meses da janela de treinamento.

    Linha de preço em cinza ao fundo; cada ponto colorido pelo cluster
    atribuído no treino. Escala discreta, uma cor por regime.
    """
    # Janela de 2 meses terminando 5 meses antes do fim do treino
    # (com forward=2021 → 06/2020 a 08/2020)
    end = data.index[-1].to_period('M').to_timestamp()
    view = data.loc[end - pd.DateOffset(months=7):end - pd.DateOffset(months=5)]

    n = hyper_params['n_clusters']
    cmap = plt.get_cmap('tab10', n)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(view.index, view['close'], color=BASELINE, linewidth=1.0,
            alpha=0.9, zorder=1)
    scatter = ax.scatter(view.index, view['close'], c=view['clusters'],
                         cmap=cmap, vmin=-0.5, vmax=n - 0.5,
                         s=14, alpha=0.9, zorder=2)
    cbar = plt.colorbar(scatter, ax=ax, label='Cluster', ticks=range(n))
    cbar.outline.set_visible(False)
    ax.set_title('Regimes de mercado (K-Means) — período de treino, '
                 f'{view.index[0]:%m/%Y} a {view.index[-1]:%m/%Y}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    fig.tight_layout()
    save_fig(fig, os.path.join(IMG_DIR, '03_Clusters.png'))
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Treinamento e avaliação
# ──────────────────────────────────────────────────────────────────────────────

def test_model(result: list, stop: float, take: float, plt=False):
    """Avalia o par de modelos no dataset completo (treino + validação OOS).

    `plt` aceita False (sem gráfico), True (exibe) ou um caminho de arquivo
    (exibe e salva o PNG).
    """
    pr_tst = make_features(get_prices())
    X = pr_tst[pr_tst.columns[1:]]
    X_meta = X.copy()
    X = X.loc[:, ~X.columns.str.contains('meta_feature')]
    X_meta = X_meta.loc[:, X_meta.columns.str.contains('meta_feature')]

    pr_tst['labels'] = result[0].predict_proba(X)[:, 1]
    pr_tst['meta_labels'] = result[1].predict_proba(X_meta)[:, 1]
    pr_tst['labels'] = pr_tst['labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    pr_tst['meta_labels'] = pr_tst['meta_labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    return tester(pr_tst, stop, take, hyper_params['forward'], hyper_params['backward'], hyper_params['markup'], plt)


def clustering(dataset, n_clusters: int):
    """K-Means sobre as meta-features, restrito à janela de treino."""
    data = dataset[(dataset.index < hyper_params['forward']) & (dataset.index > hyper_params['backward'])].copy()
    meta_X = data.loc[:, data.columns.str.contains('meta_feature')]
    data['clusters'] = KMeans(n_clusters=n_clusters, n_init='auto').fit(meta_X).labels_
    return data


def fit_final_models(clustered, meta) -> list:
    """Treina modelo principal (direção) e meta-modelo (regime) para um cluster."""
    X, X_meta = clustered[clustered.columns[:-1]], meta[meta.columns[:-1]]
    X = X.loc[:, ~X.columns.str.contains('meta_feature')]
    X_meta = X_meta.loc[:, X_meta.columns.str.contains('meta_feature')]

    y = clustered['labels'].astype('int16')
    y_meta = meta['clusters'].astype('int16')

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.7, test_size=0.3, shuffle=True)

    train_X_m, test_X_m, train_y_m, test_y_m = train_test_split(
        X_meta, y_meta, train_size=0.7, test_size=0.3, shuffle=True)

    model = CatBoostClassifier(iterations=1000,
                               custom_loss=['Accuracy'],
                               eval_metric='Accuracy',
                               verbose=False,
                               use_best_model=False,
                               task_type='CPU',
                               thread_count=-1)
    model.fit(train_X, train_y, eval_set=(test_X, test_y),
              early_stopping_rounds=30, plot=False)

    meta_model = CatBoostClassifier(iterations=500,
                                    custom_loss=['F1'],
                                    eval_metric='F1',
                                    verbose=False,
                                    use_best_model=True,
                                    task_type='CPU',
                                    thread_count=-1)
    meta_model.fit(train_X_m, train_y_m, eval_set=(test_X_m, test_y_m),
                   early_stopping_rounds=25, plot=False)

    R2 = test_model([model, meta_model], hyper_params['stop_loss'], hyper_params['take_profit'])
    if math.isnan(R2):
        R2 = -1.0
        print('R2 is fixed to -1.0')
    print('R2: ' + str(R2))

    return [R2, model, meta_model]


def run_training(dataset, labeling_fn, iterations: int = 1) -> list:
    """LEARNING LOOP — treina um par de modelos (principal + meta) por cluster.

    Aumente `iterations` para gerar mais candidatos (ex.: 10 → até 100 pares).
    """
    models = []
    for i in range(iterations):
        data = clustering(dataset, n_clusters=hyper_params['n_clusters'])
        if i == 0:
            plot_clusters(data)
        for clust in sorted(data['clusters'].unique()):
            clustered_data = data[data['clusters'] == clust].copy()
            if len(clustered_data) < 500:
                print('too few samples: {}'.format(len(clustered_data)))
                continue

            clustered_data = labeling_fn(clustered_data)
            if len(clustered_data) < 100:
                print(f'Cluster {clust}: exemplos insuficientes após rotulagem, pulando.')
                continue

            print(f'Iteration: {i}, Cluster: {clust}')
            clustered_data = clustered_data.drop(['close', 'clusters'], axis=1)

            meta_data = data.copy()
            meta_data['clusters'] = meta_data['clusters'].apply(lambda x: 1 if x == clust else 0)
            models.append(fit_final_models(clustered_data, meta_data.drop(['close'], axis=1)))
    return models


def main():
    parser = argparse.ArgumentParser(description='Treinamento e exportação da estratégia de reversão à média.')
    parser.add_argument('--symbol', default=hyper_params['symbol'],
                        help='Símbolo do CSV em Python/ (padrão: %(default)s)')
    parser.add_argument('--method', type=int, default=1, choices=sorted(LABELING_METHODS),
                        help='Método de rotulagem 1–6 (padrão: %(default)s)')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Repetições do loop de treinamento (padrão: %(default)s)')
    args = parser.parse_args()

    hyper_params['symbol'] = args.symbol
    method_name, labeling_fn = LABELING_METHODS[args.method]

    apply_style()
    os.makedirs(IMG_DIR, exist_ok=True)

    print(f"Símbolo: {hyper_params['symbol']}")
    print(f"Método de rotulagem: {args.method} — {method_name}")

    # 1. Dados + features
    dataset = make_features(get_prices())
    print(f'Dataset pronto: {dataset.shape[0]:,} linhas × {dataset.shape[1]} colunas '
          f'({dataset.index[0].date()} → {dataset.index[-1].date()})')

    # 2. Imagens ilustrativas da rotulagem
    plot_filter_illustration(dataset)
    plot_spline_illustration(dataset)

    # 3. Treinamento (clusterização + rotulagem + CatBoost por cluster)
    models = run_training(dataset, labeling_fn, iterations=args.iterations)
    if not models:
        raise SystemExit('Nenhum modelo treinado — verifique dados e parâmetros.')

    # 4. Seleção do melhor por R² + gráfico de desempenho
    models.sort(key=lambda x: x[0])
    best_model = models[-1]
    print(f"\nMelhor R²: {best_model[0]:.4f}")
    test_model(best_model[1:], hyper_params['stop_loss'], hyper_params['take_profit'],
               plt=os.path.join(IMG_DIR, '04_Desempenho.png'))

    # 5. Exportação ONNX + include MQL5
    os.makedirs(hyper_params['export_path'], exist_ok=True)
    export_model_to_ONNX(model=best_model,
                         symbol=hyper_params['symbol'],
                         periods=hyper_params['periods'],
                         periods_meta=hyper_params['periods_meta'],
                         model_number=hyper_params['model_number'],
                         export_path=hyper_params['export_path'])

    # Snapshot do melhor modelo (ignorado pelo git)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    print('best_model.pkl salvo. Pipeline concluído.')


if __name__ == '__main__':
    main()
