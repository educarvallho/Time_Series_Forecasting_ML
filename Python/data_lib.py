"""Carregamento de dados e engenharia de features.

Fonte única usada pelo script `mean reversion.py` e pelos notebooks —
qualquer mudança de formato de dados ou de features deve ser feita aqui.
"""

import os

import pandas as pd


def load_prices(csv_path: str) -> pd.DataFrame:
    """Carrega uma série de preços de fechamento a partir de um CSV OHLC.

    Dois formatos são detectados automaticamente pelo cabeçalho:

    1. **Genérico** (separado por vírgula)::

        Date,Time,Open,High,Low,Close,Volume
        20100104,00:00:00,0.88822,...

    2. **Export padrão do MetaTrader 5** (separado por tab) — retrocompatível::

        <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
        2010.01.04\t00:00:00\t0.88822\t...

    Args:
        csv_path: Caminho do arquivo CSV.

    Returns:
        DataFrame com índice datetime (`time`) e uma única coluna `close`,
        ordenado no tempo e sem valores ausentes.
    """
    with open(csv_path, "r") as f:
        header = f.readline()

    if "<DATE>" in header:
        # Formato MetaTrader 5
        p = pd.read_csv(csv_path, sep=r"\s+")
        time = pd.to_datetime(
            p["<DATE>"].astype(str) + " " + p["<TIME>"].astype(str),
            format="%Y.%m.%d %H:%M:%S",
        )
        close = p["<CLOSE>"]
    else:
        # Formato genérico (Date,Time,...,Close,...) — nomes sem distinção
        # de maiúsculas/minúsculas
        p = pd.read_csv(csv_path)
        p.columns = [c.strip().lower() for c in p.columns]
        time = pd.to_datetime(
            p["date"].astype(str) + " " + p["time"].astype(str),
            format="%Y%m%d %H:%M:%S",
        )
        close = p["close"]

    prices = pd.DataFrame({"close": close.values}, index=pd.DatetimeIndex(time, name="time"))
    return prices.dropna().sort_index()


def find_symbol_csv(symbol: str, base_dir: str = None) -> str:
    """Resolve o caminho do CSV de um símbolo (ex.: ``EURGBP_H1`` → ``EURGBP_H1.csv``).

    Procura na pasta ``base_dir`` (por padrão, a pasta desta biblioteca —
    ``Python/``), o que funciona tanto para o script quanto para os notebooks.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, symbol + ".csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV do símbolo não encontrado: {csv_path}\n"
            "Coloque o arquivo de dados na pasta Python/ com o nome <SIMBOLO>.csv"
        )
    return csv_path


def get_features(data: pd.DataFrame, periods: list, periods_meta: list) -> pd.DataFrame:
    """Calcula as features usadas pelos dois modelos.

    - **Features principais** (modelo direcional): médias móveis do fechamento,
      uma coluna por período em ``periods``.
    - **Meta-features** (modelo de regime): skewness (assimetria) rolante do
      fechamento, uma coluna por período em ``periods_meta``. As colunas levam
      o sufixo ``meta_feature``, usado pelo restante do pipeline para separá-las.

    Args:
        data: DataFrame com a coluna `close` (saída de :func:`load_prices`).
        periods: Janelas das médias móveis (features principais).
        periods_meta: Janelas da skewness (meta-features).

    Returns:
        DataFrame original + colunas de features, sem as linhas iniciais (NaN).
    """
    pFixed = data.copy()
    pFixedC = data.copy()
    count = 0

    for i in periods:
        pFixed[str(count)] = pFixedC.rolling(i).mean()
        count += 1

    for i in periods_meta:
        pFixed[str(count) + "meta_feature"] = pFixedC.rolling(i).skew()
        count += 1

    # Variantes de meta-feature para experimentação (manter comentadas):
    # for i in periods_meta:
    #     pFixed[str(count) + 'meta_feature'] = pFixedC.rolling(i).std()
    #     count += 1
    # for i in periods_meta:
    #     pFixed[str(count) + 'meta_feature'] = pFixedC - pFixedC.rolling(i).mean()
    #     count += 1

    return pFixed.dropna()
