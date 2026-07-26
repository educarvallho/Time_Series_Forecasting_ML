"""Tema visual dos gráficos do projeto.

Paleta categórica validada para daltonismo (deutan/protan/tritan) e
contraste sobre superfície clara. Todos os gráficos do repositório
(script e notebooks) usam estas cores e este estilo.
"""

import matplotlib.pyplot as plt

# Séries (identidade fixa — não trocar a ordem)
COLOR_STRATEGY = "#2a78d6"   # azul    — curva de equity / linhas suavizadas
COLOR_BENCHMARK = "#eb6834"  # laranja — variação bruta do preço (benchmark)
COLOR_ZONE = "#4a3aa7"       # violeta — zona de reversão
COLOR_PRICE = "#008300"      # verde   — preço de fechamento (ilustrações)

# Anotações e chrome
COLOR_TREND = "#898781"      # cinza   — linha de tendência linear
INK = "#0b0b0b"              # texto primário
INK_SOFT = "#52514e"         # texto secundário (rótulos de eixo)
MUTED = "#898781"            # ticks
GRID = "#e1e0d9"             # linhas de grade
BASELINE = "#c3c2b7"         # eixos
SURFACE = "#fcfcfb"          # fundo


def apply_style() -> None:
    """Aplica o tema do projeto ao matplotlib (idempotente)."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
    })


def save_fig(fig, path: str, dpi: int = 150) -> None:
    """Salva a figura em PNG com o fundo do tema (para o README)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Imagem salva em: {path}")
