"""Funções de visualização para análise de carga elétrica do SIN.

Este módulo implementa as figuras da análise exploratória (Fase 3),
seguindo as convenções do curso (Aula 10 e guia de visualização):

- Título indica a conclusão; subtítulo descreve o que é medido
- Anos em rampa ordinal de um só tom (claro → escuro)
- Subsistemas em cores categóricas fixas
- Paletas validadas para contraste e daltonismo
- Fonte no rodapé de cada figura

Figuras disponíveis:
    - Fig. 1: Perfil horário médio de carga e carga residual
    - Fig. 2: Heatmap mês × hora da carga residual
    - Fig. 3/4: Distribuições por ano (boxplots e linhas mensais)
    - Fig. 5: Coeficientes de regressão com intervalos de confiança
    - Fig. 6: Intercâmbio por condição de carga residual

Example:
    >>> from src.visualizacao import estilo, fig_perfil_horario, perfil_horario
    >>> estilo()
    >>> perfil = perfil_horario(base_horaria, nivel="SIN")
    >>> fig = fig_perfil_horario(perfil, "Título", "Subtítulo")
    >>> fig.savefig("figura.png")
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# =============================================================================
# Constantes de estilo
# =============================================================================

FONTE: str = "Fonte: ONS, Balanço de Energia nos Subsistemas (2023-2025). Elaboração própria."
"""Texto de atribuição exibido no rodapé de todas as figuras."""

CORES_ANO: dict[int, str] = {2023: "#86b6ef", 2024: "#2a78d6", 2025: "#104281"}
"""Paleta ordinal para anos: azul claro (2023) → azul escuro (2025)."""

CORES_SUBSISTEMA: dict[str, str] = {
    "N": "#2a78d6",
    "NE": "#eb6834",
    "S": "#1baf7a",
    "SE": "#eda100",
    "SIN": "#52514e",
}
"""Paleta categórica para subsistemas e SIN agregado."""

NOMES: dict[str, str] = {
    "N": "Norte",
    "NE": "Nordeste",
    "S": "Sul",
    "SE": "Sudeste/Centro-Oeste",
    "SIN": "SIN",
}
"""Nomes por extenso dos subsistemas para rótulos de figuras."""

ORDEM: list[str] = ["SIN", "SE", "S", "NE", "N"]
"""Ordem de exibição dos níveis em painéis múltiplos."""

TINTA: str = "#0b0b0b"
"""Cor principal para texto (preto suave)."""

TINTA_2: str = "#52514e"
"""Cor secundária para texto (cinza)."""

GRADE: str = "#e4e3df"
"""Cor da grade dos gráficos."""

SEQUENCIAL: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)
"""Paleta sequencial azul para heatmaps."""

TOPO: float = 0.84
"""Fração da altura reservada aos painéis (acima ficam título e subtítulo)."""

INDICADORES: dict[str, tuple[str, float, int]] = {
    "indice_vale": ("Índice de vale (carga 10–15h ÷ média do dia)", 1, 3),
    "carga_residual_min": ("Carga residual mínima 10–15h (MWmed)", 1, 0),
    "rampa_tarde": ("Rampa do fim da tarde (MWmed)", 1, 0),
    "maior_rampa_horaria": ("Maior rampa horária 16–21h (MWmed/h)", 1, 0),
    "horas_residual_baixa": ("Horas/dia com carga residual baixa", 1, 2),
}
"""Configuração dos indicadores: (nome, escala, casas decimais)."""

UNIDADES_COEF: dict[str, tuple[str, str, float, int]] = {
    "indice_vale": ("Índice de vale", "diferença no índice (adimensional)", 1, 3),
    "carga_residual_min": ("Carga residual mínima 10–15h", "GWmed", 1000, 1),
    "rampa_tarde": ("Rampa do fim da tarde", "GWmed", 1000, 1),
    "maior_rampa_horaria": ("Maior rampa horária 16–21h", "GWmed/h", 1000, 1),
    "horas_residual_baixa": ("Horas/dia com carga residual baixa", "horas por dia", 1, 1),
}
"""Configuração de unidades para gráfico de coeficientes: (nome, unidade, escala, casas)."""


# =============================================================================
# Funções utilitárias
# =============================================================================


def estilo() -> None:
    """Configura o estilo global do matplotlib para as figuras do projeto.

    Aplica configurações de:
        - Resolução (DPI)
        - Tipografia e tamanhos de fonte
        - Cores de eixos, grade e ticks
        - Remoção de bordas superiores e direitas
        - Largura de linhas

    Esta função deve ser chamada uma vez no início do notebook,
    antes de criar qualquer figura.

    Example:
        >>> from src.visualizacao import estilo
        >>> estilo()  # Aplica configurações globais
    """
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": GRADE,
            "axes.labelcolor": TINTA_2,
            "axes.grid": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TINTA_2,
            "ytick.color": TINTA_2,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def br(valor: float, casas: int = 0) -> str:
    """Formata número no padrão brasileiro (separador de milhar: ponto; decimal: vírgula).

    Args:
        valor: Número a ser formatado.
        casas: Quantidade de casas decimais. Padrão: 0.

    Returns:
        String formatada no padrão brasileiro (ex: "1.234,56").

    Example:
        >>> br(1234.5, 1)
        '1.234,5'
        >>> br(1000000, 0)
        '1.000.000'
    """
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def eixo_br(eixo: plt.Axes, casas: int = 0, escala: float = 1.0) -> None:
    """Aplica formatação brasileira ao eixo Y de um gráfico.

    Args:
        eixo: Objeto Axes do matplotlib a ser formatado.
        casas: Quantidade de casas decimais nos rótulos. Padrão: 0.
        escala: Divisor para os valores (ex: 1000 para converter MW em GW). Padrão: 1.0.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1000, 2000, 3000])
        >>> eixo_br(ax, casas=1, escala=1000)  # Mostra "1,0", "2,0", "3,0"
    """
    eixo.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: br(v / escala, casas))
    )


def rodape(fig: Figure, texto_extra: str = "") -> None:
    """Adiciona rodapé com fonte e atribuição à figura.

    Args:
        fig: Objeto Figure do matplotlib.
        texto_extra: Texto adicional a ser concatenado após a fonte. Padrão: "".

    Example:
        >>> fig, ax = plt.subplots()
        >>> rodape(fig, "Nota: exclui dias atípicos.")
    """
    fig.text(
        0.01,
        0.005,
        f"{FONTE} {texto_extra}".strip(),
        fontsize=8,
        color=TINTA_2,
        ha="left",
        va="bottom",
    )


def titulo(fig: Figure, conclusao: str, subtitulo: str) -> None:
    """Adiciona título (conclusão) e subtítulo à figura.

    O título aparece em negrito no topo; o subtítulo logo abaixo,
    em tamanho menor e cor secundária.

    Args:
        fig: Objeto Figure do matplotlib.
        conclusao: Texto principal do título (a conclusão do gráfico).
        subtitulo: Descrição do que é medido, nível e recorte temporal.

    Example:
        >>> fig, ax = plt.subplots()
        >>> titulo(fig, "A carga residual caiu 8,6 GW ao meio-dia",
        ...        "Perfil horário médio, SIN, 2023–2025")
    """
    fig.text(
        0.01,
        0.985,
        conclusao,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TINTA,
    )
    fig.text(
        0.01, 0.915, subtitulo, ha="left", va="top", fontsize=9.5, color=TINTA_2
    )


# =============================================================================
# Figura 1: Perfil horário
# =============================================================================


def perfil_horario(base: pd.DataFrame, nivel: str = "SIN") -> pd.DataFrame:
    """Calcula estatísticas horárias de carga e carga residual por ano.

    Agrupa os dados por ano e hora, calculando média, P25 e P75
    para carga observada e carga residual, convertendo para GWmed.

    Args:
        base: DataFrame com colunas 'id_subsistema', 'ano', 'hora',
              'val_carga' e 'carga_residual' (em MWmed).
        nivel: Subsistema ou 'SIN' para filtrar. Padrão: "SIN".

    Returns:
        DataFrame com colunas:
            - ano: int (2023, 2024, 2025)
            - hora: int (0–23)
            - variavel: str ('val_carga' ou 'carga_residual')
            - media: float (GWmed)
            - p25: float (percentil 25, GWmed)
            - p75: float (percentil 75, GWmed)

    Example:
        >>> perfil = perfil_horario(base_horaria, nivel="SE")
        >>> perfil[perfil["hora"] == 12]  # Estatísticas ao meio-dia
    """
    dados = base[base["id_subsistema"] == nivel]
    linhas: list[dict[str, Any]] = []
    for (ano, hora), grupo in dados.groupby(["ano", "hora"]):
        for variavel in ("val_carga", "carga_residual"):
            serie = grupo[variavel] / 1000
            linhas.append(
                {
                    "ano": ano,
                    "hora": hora,
                    "variavel": variavel,
                    "media": serie.mean(),
                    "p25": serie.quantile(0.25),
                    "p75": serie.quantile(0.75),
                }
            )
    return pd.DataFrame(linhas)


def fig_perfil_horario(
    perfil: pd.DataFrame,
    conclusao: str,
    subtitulo: str,
    rodape_extra: str = "",
) -> Figure:
    """Cria Figura 1: perfil horário médio de carga e carga residual.

    Dois painéis lado a lado:
        - Esquerda: carga observada
        - Direita: carga residual (carga − eólica − solar)

    Cada painel mostra uma linha por ano com banda P25–P75.
    A janela do vale (10h–15h) é destacada em cinza.

    Args:
        perfil: DataFrame gerado por `perfil_horario()`.
        conclusao: Título principal da figura (a conclusão).
        subtitulo: Descrição do que é medido.
        rodape_extra: Texto adicional para o rodapé. Padrão: "".

    Returns:
        Objeto Figure do matplotlib pronto para exibição ou salvamento.

    Example:
        >>> perfil = perfil_horario(base_horaria, nivel="SIN")
        >>> fig = fig_perfil_horario(
        ...     perfil,
        ...     "A carga residual ao meio-dia caiu 8,6 GW de 2023 para 2025",
        ...     "Perfil horário médio de carga e carga residual, SIN, maio–dez"
        ... )
        >>> fig.savefig("resultados/figuras/fig1_perfil_horario.png")
    """
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    paineis = {
        "val_carga": "Carga observada",
        "carga_residual": "Carga residual (carga − eólica − solar)",
    }
    for eixo, (variavel, nome) in zip(eixos, paineis.items()):
        for ano, cor in CORES_ANO.items():
            p = perfil[(perfil["variavel"] == variavel) & (perfil["ano"] == ano)]
            eixo.fill_between(
                p["hora"], p["p25"], p["p75"], color=cor, alpha=0.15, linewidth=0
            )
            eixo.plot(p["hora"], p["media"], color=cor, label=str(ano))
            if variavel == "carga_residual":
                # Rótulo direto no vale, onde as linhas se separam
                y12 = p.loc[p["hora"] == 12, "media"].iloc[0]
                eixo.annotate(
                    f"{ano}: {br(y12, 1)}",
                    (12, y12),
                    xytext=(-8, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    color=TINTA,
                )
        eixo.axvspan(10, 15, color="#f0efec", zorder=0)
        eixo.set_title(nome, fontsize=11)
        eixo.set_xticks(range(0, 24, 3), [f"{h}h" for h in range(0, 24, 3)])
        eixo.set_xlim(0, 23)
        eixo.set_xlabel("Hora do dia (faixa cinza: janela do vale, 10h–15h)")
        eixo_br(eixo)
    eixos[0].set_ylabel("GWmed (média; faixa: P25–P75)")
    eixos[0].legend(title="Ano", loc="lower right")
    titulo(fig, conclusao, subtitulo)
    rodape(fig, rodape_extra)
    fig.tight_layout(rect=(0, 0.03, 1, TOPO))
    return fig


# =============================================================================
# Figura 2: Heatmap mês × hora
# =============================================================================


def matriz_mes_hora(base: pd.DataFrame, nivel: str, ano: int) -> pd.DataFrame:
    """Calcula matriz de carga residual média por mês e hora.

    Args:
        base: DataFrame com colunas 'id_subsistema', 'ano', 'mes',
              'hora' e 'carga_residual' (em MWmed).
        nivel: Subsistema ou 'SIN' para filtrar.
        ano: Ano para filtrar (2023, 2024 ou 2025).

    Returns:
        DataFrame pivotado com meses nas linhas (1–12),
        horas nas colunas (0–23) e valores em GWmed.

    Example:
        >>> matriz = matriz_mes_hora(base_horaria, "SIN", 2025)
        >>> matriz.loc[7, 12]  # Julho, 12h
    """
    dados = base[(base["id_subsistema"] == nivel) & (base["ano"] == ano)]
    return (
        dados.pivot_table(
            index="mes", columns="hora", values="carga_residual", aggfunc="mean"
        )
        / 1000
    )


def fig_heatmap(
    base: pd.DataFrame,
    nivel: str,
    conclusao: str,
    subtitulo: str,
    rodape_extra: str = "",
) -> Figure:
    """Cria Figura 2: heatmap mês × hora da carga residual.

    Três painéis lado a lado (2023, 2024, 2025) com escala de cor comum.
    O período jan–abr/2023 é hachurado por incompletude da série solar.

    Args:
        base: DataFrame com dados horários completos.
        nivel: Subsistema ou 'SIN' para exibir.
        conclusao: Título principal da figura (a conclusão).
        subtitulo: Descrição do que é medido.
        rodape_extra: Texto adicional para o rodapé. Padrão: "".

    Returns:
        Objeto Figure do matplotlib.

    Example:
        >>> fig = fig_heatmap(
        ...     base_horaria, "SIN",
        ...     "O vale diurno se aprofundou de jul a out entre 10h e 15h",
        ...     "Carga residual média por mês e hora, SIN"
        ... )
    """
    matrizes = {ano: matriz_mes_hora(base, nivel, ano) for ano in CORES_ANO}
    vmin = min(m.min().min() for m in matrizes.values())
    vmax = max(m.max().max() for m in matrizes.values())
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for eixo, (ano, matriz) in zip(eixos, matrizes.items()):
        imagem = eixo.imshow(
            matriz,
            aspect="auto",
            cmap=SEQUENCIAL,
            vmin=vmin,
            vmax=vmax,
            extent=(-0.5, 23.5, 12.5, 0.5),
        )
        if ano == 2023:
            eixo.axhspan(
                0.5,
                4.5,
                facecolor="white",
                edgecolor=TINTA_2,
                hatch="///",
                alpha=0.85,
                linewidth=0,
            )
            eixo.text(
                11.5,
                2.5,
                "jan–abr/2023: série solar incompleta",
                ha="center",
                va="center",
                fontsize=8.5,
                color=TINTA,
            )
        eixo.set_title(str(ano), fontsize=11)
        eixo.set_xticks(range(0, 24, 3), [f"{h}h" for h in range(0, 24, 3)])
        eixo.set_xlabel("Hora do dia")
        eixo.grid(False)
    eixos[0].set_yticks(
        range(1, 13),
        [
            "jan",
            "fev",
            "mar",
            "abr",
            "mai",
            "jun",
            "jul",
            "ago",
            "set",
            "out",
            "nov",
            "dez",
        ],
    )
    fig.subplots_adjust(left=0.05, right=0.9, top=0.8, bottom=0.17, wspace=0.06)
    barra = fig.colorbar(imagem, cax=fig.add_axes([0.915, 0.17, 0.012, 0.63]))
    barra.set_label("Carga residual média (GWmed)", color=TINTA_2)
    barra.formatter = mticker.FuncFormatter(lambda v, _: br(v))
    barra.update_ticks()
    titulo(fig, conclusao, subtitulo)
    rodape(fig, rodape_extra)
    return fig


# =============================================================================
# Figuras 3 e 4: Distribuições por ano
# =============================================================================


def fig_distribuicao_por_ano(
    diarios: pd.DataFrame,
    coluna: str,
    rotulo_y: str,
    conclusao: str,
    subtitulo: str,
    escala: float = 1000,
    casas: int = 0,
    rodape_extra: str = "",
) -> Figure:
    """Cria boxplots de um indicador diário por ano, em painéis por nível.

    Cinco painéis (SIN, SE, S, NE, N), cada um com boxplots para
    2023, 2024 e 2025. Outliers são omitidos; medianas são rotuladas.

    Args:
        diarios: DataFrame com indicadores diários (uma linha por dia × nível).
        coluna: Nome da coluna do indicador a plotar.
        rotulo_y: Rótulo do eixo Y.
        conclusao: Título principal da figura.
        subtitulo: Descrição do que é medido.
        escala: Divisor para os valores (ex: 1000 para MW→GW). Padrão: 1000.
        casas: Casas decimais nos rótulos das medianas. Padrão: 0.
        rodape_extra: Texto adicional para o rodapé. Padrão: "".

    Returns:
        Objeto Figure do matplotlib.

    Example:
        >>> fig = fig_distribuicao_por_ano(
        ...     diarios, "rampa_tarde", "GWmed",
        ...     "A rampa do fim da tarde cresceu 45% de 2023 para 2025",
        ...     "Distribuição da rampa 18h–20h por ano, todos os níveis"
        ... )
    """
    fig, eixos = plt.subplots(1, 5, figsize=(15, 4.4))
    for eixo, nivel in zip(eixos, ORDEM):
        dados = [
            diarios[(diarios["id_subsistema"] == nivel) & (diarios["ano"] == a)][
                coluna
            ]
            / escala
            for a in CORES_ANO
        ]
        caixas = eixo.boxplot(
            dados,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=2),
        )
        for caixa, cor in zip(caixas["boxes"], CORES_ANO.values()):
            caixa.set(facecolor=cor, edgecolor=cor)
        for elemento in ("whiskers", "caps"):
            for linha, cor in zip(
                caixas[elemento], np.repeat(list(CORES_ANO.values()), 2)
            ):
                linha.set(color=cor, linewidth=1.2)
        medianas = [d.median() for d in dados]
        for x, m in enumerate(medianas, start=1):
            eixo.annotate(
                br(m, casas),
                (x, m),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=TINTA,
            )
        eixo.set_xticks([1, 2, 3], [str(a) for a in CORES_ANO])
        eixo.set_title(NOMES[nivel], fontsize=10.5)
        eixo_br(eixo, casas)
        eixo.grid(axis="x", visible=False)
    eixos[0].set_ylabel(rotulo_y)
    titulo(fig, conclusao, subtitulo)
    rodape(fig, rodape_extra)
    fig.tight_layout(rect=(0, 0.03, 1, TOPO))
    return fig


def fig_mensal_por_ano(
    diarios: pd.DataFrame,
    coluna: str,
    rotulo_y: str,
    conclusao: str,
    subtitulo: str,
    rodape_extra: str = "",
    casas: int = 1,
) -> Figure:
    """Cria gráfico de linhas mensais de um indicador, uma linha por ano.

    Cinco painéis (SIN, SE, S, NE, N) com média mensal do indicador.
    Jan–abr/2023 é omitido por incompletude da série solar.

    Args:
        diarios: DataFrame com indicadores diários.
        coluna: Nome da coluna do indicador a plotar.
        rotulo_y: Rótulo do eixo Y.
        conclusao: Título principal da figura.
        subtitulo: Descrição do que é medido.
        rodape_extra: Texto adicional para o rodapé. Padrão: "".
        casas: Casas decimais nos rótulos do eixo Y. Padrão: 1.

    Returns:
        Objeto Figure do matplotlib.

    Example:
        >>> fig = fig_mensal_por_ano(
        ...     diarios, "horas_residual_baixa", "horas/dia",
        ...     "As horas de carga residual baixa triplicaram de 2023 para 2025",
        ...     "Média mensal de horas/dia com carga residual abaixo do P10"
        ... )
    """
    fig, eixos = plt.subplots(1, 5, figsize=(15, 4.2), sharey=False)
    meses = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    for eixo, nivel in zip(eixos, ORDEM):
        dados = diarios[diarios["id_subsistema"] == nivel]
        mensal = dados.groupby(["ano", "mes"])[coluna].mean().unstack("ano")
        for ano, cor in CORES_ANO.items():
            serie = mensal[ano]
            if ano == 2023:
                serie = serie.where(serie.index >= 5)
            eixo.plot(
                serie.index,
                serie.values,
                color=cor,
                marker="o",
                markersize=4,
                label=str(ano),
            )
        eixo.set_xticks(range(1, 13), meses)
        eixo.set_title(NOMES[nivel], fontsize=10.5)
        eixo_br(eixo, casas)
    eixos[0].set_ylabel(rotulo_y)
    eixos[-1].legend(title="Ano", loc="upper left", bbox_to_anchor=(1.0, 1.0))
    titulo(fig, conclusao, subtitulo)
    rodape(fig, rodape_extra)
    fig.tight_layout(rect=(0, 0.03, 1, TOPO))
    return fig


# =============================================================================
# Figura 6: Intercâmbio por condição
# =============================================================================


def intercambio_por_condicao(base: pd.DataFrame) -> pd.DataFrame:
    """Calcula intercâmbio médio nas horas 10–15h por condição de carga residual.

    Separa as horas em dois grupos:
        - Horas com carga residual baixa (residual_baixa=True)
        - Demais horas

    Args:
        base: DataFrame com colunas 'id_subsistema', 'hora',
              'val_intercambio' e 'residual_baixa'.

    Returns:
        DataFrame com colunas:
            - id_subsistema: str
            - ano: int
            - residual_baixa: bool
            - intercambio_medio: float (MWmed)
            - horas: int (contagem de observações)

    Example:
        >>> tabela = intercambio_por_condicao(base_horaria)
        >>> tabela[tabela["residual_baixa"] == True]
    """
    dados = base[(base["id_subsistema"] != "SIN") & base["hora"].between(10, 15)]
    return (
        dados.groupby(["id_subsistema", "ano", "residual_baixa"])["val_intercambio"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "intercambio_medio", "size": "horas"})
        .reset_index()
    )


def fig_intercambio(
    tabela: pd.DataFrame,
    conclusao: str,
    subtitulo: str,
    rodape_extra: str = "",
) -> Figure:
    """Cria Figura 6: intercâmbio por condição de carga residual.

    Quatro painéis (SE, S, NE, N) com barras agrupadas por ano,
    comparando intercâmbio em horas de carga residual baixa vs. demais horas.
    Valores positivos indicam exportação; negativos, importação.

    Args:
        tabela: DataFrame gerado por `intercambio_por_condicao()`.
        conclusao: Título principal da figura.
        subtitulo: Descrição do que é medido.
        rodape_extra: Texto adicional para o rodapé. Padrão: "".

    Returns:
        Objeto Figure do matplotlib.

    Example:
        >>> tabela = intercambio_por_condicao(base_horaria)
        >>> fig = fig_intercambio(
        ...     tabela,
        ...     "O Sudeste exporta mais nas horas de carga residual baixa",
        ...     "Intercâmbio médio 10–15h por subsistema e condição"
        ... )
    """
    fig, eixos = plt.subplots(1, 4, figsize=(15, 4.3))
    largura = 0.38
    for eixo, nivel in zip(eixos, ["SE", "S", "NE", "N"]):
        t = tabela[tabela["id_subsistema"] == nivel]
        x = np.arange(len(CORES_ANO))
        for deslocamento, condicao, cor, nome in (
            (-largura / 2, False, "#c3c2b7", "demais horas 10–15h"),
            (largura / 2, True, "#eb6834", "horas de carga residual baixa"),
        ):
            valores = [
                t[(t["ano"] == a) & (t["residual_baixa"] == condicao)][
                    "intercambio_medio"
                ].mean()
                / 1000
                for a in CORES_ANO
            ]
            barras = eixo.bar(
                x + deslocamento, valores, width=largura - 0.03, color=cor, label=nome
            )
            eixo.bar_label(
                barras, [br(v, 1) for v in valores], fontsize=7.5, color=TINTA, padding=2
            )
        eixo.axhline(0, color=TINTA_2, linewidth=0.8)
        eixo.set_xticks(x, [str(a) for a in CORES_ANO])
        eixo.set_title(NOMES[nivel], fontsize=10.5)
        eixo_br(eixo, 1)
        eixo.grid(axis="x", visible=False)
    eixos[0].set_ylabel("Intercâmbio médio, GWmed\n(positivo = exportação)")
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(
        alcas, rotulos, loc="upper right", bbox_to_anchor=(0.995, 0.93), ncols=2, fontsize=9
    )
    titulo(fig, conclusao, subtitulo)
    rodape(fig, rodape_extra)
    fig.tight_layout(rect=(0, 0.03, 1, TOPO))
    return fig


# =============================================================================
# Tabela-resumo de indicadores
# =============================================================================


def tabela_resumo(diarios: pd.DataFrame) -> pd.DataFrame:
    """Calcula médias anuais e variação 2023→2025 por nível e indicador.

    Usa apenas o período maio–dezembro (sem dias atípicos) para
    comparabilidade entre anos.

    Args:
        diarios: DataFrame com indicadores diários filtrados
                 (maio–dez, sem dias atípicos).

    Returns:
        DataFrame com colunas:
            - nivel: str (SIN, SE, S, NE, N)
            - indicador: str (nome da coluna)
            - 2023, 2024, 2025: float (médias anuais)
            - var_abs_23_25: float (variação absoluta)
            - var_pct_23_25: float (variação percentual)

    Example:
        >>> resumo = tabela_resumo(diarios_filtrados)
        >>> resumo[resumo["indicador"] == "rampa_tarde"]
    """
    medias = diarios.groupby(["id_subsistema", "ano"])[list(INDICADORES)].mean()
    linhas: list[dict[str, Any]] = []
    for nivel in ORDEM:
        for coluna in INDICADORES:
            m = medias.loc[nivel, coluna]
            variacao = m[2025] - m[2023]
            base_rel = abs(m[2023]) if m[2023] != 0 else np.nan
            linhas.append(
                {
                    "nivel": nivel,
                    "indicador": coluna,
                    "2023": m[2023],
                    "2024": m[2024],
                    "2025": m[2025],
                    "var_abs_23_25": variacao,
                    "var_pct_23_25": 100 * variacao / base_rel,
                }
            )
    return pd.DataFrame(linhas)


# =============================================================================
# Figura 5: Coeficientes de regressão
# =============================================================================


def fig_coeficientes(
    coef: pd.DataFrame,
    conclusao: str,
    subtitulo: str,
    rodape_extra: str = "",
) -> Figure:
    """Cria Figura 5: diferença ajustada 2024/2025 vs. 2023 com IC 95%.

    Cinco painéis (um por indicador), cada um mostrando os coeficientes
    de ano (diferença em relação a 2023) com barras de erro (IC 95%)
    para cada nível (SIN, SE, S, NE, N).

    Args:
        coef: DataFrame com colunas 'indicador', 'nivel', 'ano',
              'coef', 'ic_inf', 'ic_sup' (resultados da regressão).
        conclusao: Título principal da figura.
        subtitulo: Descrição do que é medido.
        rodape_extra: Texto adicional para o rodapé. Padrão: "".

    Returns:
        Objeto Figure do matplotlib.

    Example:
        >>> fig = fig_coeficientes(
        ...     coeficientes,
        ...     "Todos os indicadores pioraram significativamente de 2023 para 2025",
        ...     "Diferença ajustada (MQO com erros HAC) vs. 2023, IC 95%"
        ... )
    """
    fig, eixos = plt.subplots(1, 5, figsize=(16, 4.6), sharey=True)
    niveis = ORDEM[::-1]  # SIN no topo
    for eixo, (indicador, (nome, unidade, escala, casas)) in zip(
        eixos, UNIDADES_COEF.items()
    ):
        dados = coef[coef["indicador"] == indicador]
        for ano, deslocamento in ((2024, -0.15), (2025, 0.15)):
            d = dados[dados["ano"] == ano].set_index("nivel").loc[niveis]
            y = np.arange(len(niveis)) + deslocamento
            eixo.errorbar(
                d["coef"] / escala,
                y,
                xerr=[
                    (d["coef"] - d["ic_inf"]) / escala,
                    (d["ic_sup"] - d["coef"]) / escala,
                ],
                fmt="o",
                color=CORES_ANO[ano],
                markersize=6,
                elinewidth=1.6,
                capsize=0,
                label=f"{ano} × 2023",
            )
            if ano == 2025:
                for yi, valor in zip(y, d["coef"] / escala):
                    eixo.annotate(
                        br(valor, casas),
                        (valor, yi),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7.5,
                        color=TINTA,
                    )
        eixo.axvline(0, color=TINTA_2, linewidth=0.9)
        eixo.set_title(nome, fontsize=10)
        eixo.set_xlabel(unidade, fontsize=8.5)
        eixo.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda v, _, c=casas: br(v, max(c - 1, 0) if c > 1 else c)
            )
        )
        eixo.grid(axis="y", visible=False)
    eixos[0].set_yticks(range(len(niveis)), [NOMES[n] for n in niveis])
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(
        alcas, rotulos, loc="upper right", bbox_to_anchor=(0.995, 0.93), ncols=2, fontsize=9
    )
    titulo(fig, conclusao, subtitulo)
    rodape(fig, rodape_extra)
    fig.tight_layout(rect=(0, 0.03, 1, TOPO))
    return fig
