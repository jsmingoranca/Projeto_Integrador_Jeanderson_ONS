"""Teste estatístico da variação entre anos dos indicadores diários.

Este módulo implementa a análise estatística principal do projeto:
comparar os indicadores diários entre 2023, 2024 e 2025, controlando
por mês e tipo de dia.

Modelo utilizado (por nível e indicador, estimado por MQO):

    indicador ~ C(ano, Treatment(2023)) + C(mes) + dia_util

Características:
- Recorte principal: maio a dezembro (DA-018), sem dias atípicos (DA-019, DA-020)
- Erros-padrão HAC (Newey-West, 7 defasagens) para lidar com autocorrelação
- Os coeficientes de 2024 e 2025 medem a diferença média em relação a 2023,
  no mesmo mês e mesmo tipo de dia
- São associações descritivas ajustadas, NÃO efeitos causais

Análise de sensibilidade:
- Janelas horárias deslocadas ±1 hora
- Apenas dias úteis
- Incluindo dias atípicos
- HAC com 30 defasagens
- Comparação 2025 × 2024 (ano inteiro)

Exemplo de uso:
    >>> from src.modelagem import recorte_principal, estimar_todos, cenarios_sensibilidade
    >>> dados = recorte_principal(diarios)
    >>> resultados = estimar_todos(dados)
    >>> sensibilidade = cenarios_sensibilidade(diarios, base_horaria)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.stattools import durbin_watson

# =============================================================================
# CONSTANTES
# =============================================================================

# Indicadores analisados (devem existir no DataFrame de indicadores diários)
INDICADORES = [
    "indice_vale",           # Razão carga no vale / carga média do dia
    "carga_residual_min",    # Mínimo da carga residual nas horas solares (MWmed)
    "rampa_tarde",           # Subida da carga residual até a noite (MWmed)
    "maior_rampa_horaria",   # Maior subida em 1 hora (MWmed/h)
    "horas_residual_baixa",  # Horas do dia abaixo do P10 (contagem)
]

# Níveis de agregação (SIN + 4 subsistemas)
NIVEIS = ["SIN", "SE", "S", "NE", "N"]

# Número de defasagens para erros HAC (Newey-West)
# 7 dias é razoável para capturar autocorrelação semanal
MAXLAGS_HAC = 7

# Fórmula do modelo (o {y} é substituído pelo nome do indicador)
FORMULA = "{y} ~ C(ano, Treatment(2023)) + C(mes) + dia_util"


# =============================================================================
# FUNÇÕES - Recorte dos dados
# =============================================================================

def recorte_principal(diarios: pd.DataFrame) -> pd.DataFrame:
    """Aplica o recorte principal: maio a dezembro, sem dias atípicos.
    
    Este é o recorte usado na análise principal. Exclui:
    - Janeiro a abril (série solar incompleta em 2023)
    - Dias com incidentes de registro ou eventos operativos
    
    Args:
        diarios: DataFrame de indicadores diários com colunas
                 ``mes_comparavel`` e ``dia_atipico``.
                 
    Returns:
        Subconjunto do DataFrame com o recorte aplicado.
    """
    return diarios[diarios["mes_comparavel"] & ~diarios["dia_atipico"]]


# =============================================================================
# FUNÇÕES - Estimação do modelo
# =============================================================================

def ajustar(
    dados: pd.DataFrame,
    indicador: str,
    ano_base: int = 2023,
    maxlags: int = MAXLAGS_HAC
) -> RegressionResultsWrapper:
    """Ajusta o modelo MQO com erros HAC para um indicador em um nível.
    
    O modelo compara os anos usando o ano_base como referência (Treatment).
    Os coeficientes dos outros anos representam a diferença média em relação
    ao ano base, controlando por mês e tipo de dia.
    
    Erros HAC (Heteroskedasticity and Autocorrelation Consistent) são usados
    porque observações de dias consecutivos são correlacionadas.
    
    Args:
        dados: DataFrame de um único nível (SIN ou subsistema), ordenado por data.
        indicador: Nome da coluna do indicador a modelar.
        ano_base: Ano de referência para as comparações (padrão: 2023).
        maxlags: Número de defasagens para o estimador HAC (padrão: 7).
        
    Returns:
        Objeto de resultados do statsmodels com o modelo ajustado.
    """
    # Ordena por data e converte dia_util para inteiro (exigido pelo statsmodels)
    dados = dados.sort_values("data").assign(dia_util=lambda d: d["dia_util"].astype(int))
    
    # Monta a fórmula substituindo o ano base
    formula = FORMULA.format(y=indicador).replace("Treatment(2023)", f"Treatment({ano_base})")
    
    # Se todos os dias são úteis (cenário "só dias úteis"), remove o controle
    if dados["dia_util"].nunique() == 1:
        formula = formula.replace(" + dia_util", "")
    
    # Ajusta o modelo com erros HAC
    return smf.ols(formula, data=dados).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def coeficientes_ano(
    modelo: RegressionResultsWrapper,
    dados: pd.DataFrame,
    indicador: str,
    ano_base: int = 2023
) -> list[dict]:
    """Extrai os coeficientes de ano com IC 95%, p-valor e variação relativa.
    
    Para cada ano (exceto o base), extrai:
    - coef: diferença média em relação ao ano base
    - ic_inf, ic_sup: intervalo de confiança de 95%
    - p_valor: significância estatística
    - media_ano_base: média do indicador no ano base (para calcular %)
    - var_pct: variação percentual em relação ao ano base
    
    Args:
        modelo: Resultado do ajuste do modelo.
        dados: DataFrame usado no ajuste.
        indicador: Nome do indicador modelado.
        ano_base: Ano de referência.
        
    Returns:
        Lista de dicionários, um por ano comparado.
    """
    # Calcula a média do indicador no ano base (denominador da variação %)
    media_base = dados.loc[dados["ano"] == ano_base, indicador].mean()
    
    # Obtém os intervalos de confiança
    ic = modelo.conf_int()
    
    linhas = []
    for nome in modelo.params.index:
        # Filtra apenas os coeficientes de ano (começam com "C(ano")
        if not nome.startswith("C(ano"):
            continue
        
        # Extrai o ano do nome do coeficiente (ex: "C(ano, Treatment(2023))[T.2025]" -> 2025)
        ano = int(nome.split("[T.")[1].rstrip("]"))
        coef = modelo.params[nome]
        
        linhas.append({
            "ano": ano,
            "coef": coef,
            "ic_inf": ic.loc[nome, 0],
            "ic_sup": ic.loc[nome, 1],
            "p_valor": modelo.pvalues[nome],
            "media_ano_base": media_base,
            "var_pct": 100 * coef / abs(media_base) if media_base else np.nan,
        })
    
    return linhas


def estimar_todos(
    diarios: pd.DataFrame,
    cenario: str = "principal",
    ano_base: int = 2023,
    maxlags: int = MAXLAGS_HAC
) -> pd.DataFrame:
    """Estima o modelo para todos os níveis e indicadores.
    
    Itera sobre cada combinação de nível (SIN, SE, S, NE, N) e indicador,
    ajustando o modelo e extraindo os coeficientes de ano.
    
    Args:
        diarios: DataFrame de indicadores diários (já deve estar recortado
                 se for o cenário principal).
        cenario: Nome do cenário para identificação nos resultados.
        ano_base: Ano de referência para as comparações.
        maxlags: Defasagens para o estimador HAC.
        
    Returns:
        DataFrame com uma linha por combinação nível × indicador × ano,
        contendo coeficiente, IC, p-valor, R², Durbin-Watson e metadados.
    """
    linhas = []
    
    for nivel in NIVEIS:
        # Filtra os dados do nível
        dados = diarios[diarios["id_subsistema"] == nivel]
        
        for indicador in INDICADORES:
            # Ajusta o modelo
            modelo = ajustar(dados, indicador, ano_base, maxlags)
            
            # Extrai os coeficientes de cada ano
            for linha in coeficientes_ano(modelo, dados, indicador, ano_base):
                linha.update(
                    cenario=cenario,
                    nivel=nivel,
                    indicador=indicador,
                    n=int(modelo.nobs),
                    r2=modelo.rsquared,
                    durbin_watson=durbin_watson(modelo.resid),
                )
                linhas.append(linha)
    
    # Organiza as colunas na ordem desejada
    colunas = [
        "cenario", "nivel", "indicador", "ano", "coef", "ic_inf", "ic_sup", 
        "p_valor", "media_ano_base", "var_pct", "n", "r2", "durbin_watson"
    ]
    return pd.DataFrame(linhas)[colunas]


# =============================================================================
# FUNÇÕES - Análises complementares
# =============================================================================

def spearman_anomalias(diarios: pd.DataFrame) -> pd.DataFrame:
    """Correlação de Spearman entre ano e anomalia mensal do indicador.
    
    Calcula a correlação de Spearman (não-paramétrica, robusta a outliers)
    entre o ano e a anomalia do indicador (desvio da média do mês).
    
    Usa médias mensais (até 24 pontos por nível) para reduzir a autocorrelação
    diária. Remove a média de cada mês (anomalia) antes de correlacionar,
    para isolar o efeito do ano do efeito sazonal.
    
    Args:
        diarios: DataFrame de indicadores diários.
        
    Returns:
        DataFrame com ρ de Spearman, p-valor e n para cada nível × indicador.
    """
    linhas = []
    
    for nivel in NIVEIS:
        dados = diarios[diarios["id_subsistema"] == nivel]
        
        for indicador in INDICADORES:
            # Calcula média mensal
            mensal = dados.groupby(["ano", "mes"])[indicador].mean().reset_index()
            
            # Remove a média do mês (anomalia = desvio da sazonalidade)
            mensal["anomalia"] = mensal[indicador] - mensal.groupby("mes")[indicador].transform("mean")
            
            # Correlação de Spearman entre ano e anomalia
            rho, p = stats.spearmanr(mensal["ano"], mensal["anomalia"])
            
            linhas.append({
                "nivel": nivel,
                "indicador": indicador,
                "rho": rho,
                "p_valor": p,
                "n_meses": len(mensal)
            })
    
    return pd.DataFrame(linhas)


def estabilidade(cenarios: pd.DataFrame, ano: int = 2025, alfa: float = 0.05) -> pd.DataFrame:
    """Avalia a robustez dos resultados através dos cenários de sensibilidade.
    
    Para cada combinação nível × indicador, verifica se:
    - O sinal do coeficiente é o mesmo em todos os cenários
    - O resultado é significativo (p < alfa) em todos os cenários
    
    Um resultado é considerado "robusto" se mantém o mesmo sinal E é
    significativo em todos os cenários testados.
    
    Args:
        cenarios: DataFrame com resultados de múltiplos cenários.
        ano: Ano a avaliar (padrão: 2025).
        alfa: Nível de significância (padrão: 0.05).
        
    Returns:
        DataFrame com métricas de estabilidade por nível × indicador.
    """
    dados = cenarios[cenarios["ano"] == ano].copy()
    
    # Determina o sinal e significância de cada resultado
    dados["sinal"] = np.sign(dados["coef"])
    dados["significativo"] = dados["p_valor"] < alfa
    
    # Agrupa por nível e indicador
    agrupado = dados.groupby(["nivel", "indicador"])
    
    resultado = pd.DataFrame({
        "cenarios": agrupado.size(),
        "mesmo_sinal": agrupado["sinal"].nunique() == 1,
        "significativo_em": agrupado["significativo"].sum(),
        "coef_min": agrupado["coef"].min(),
        "coef_max": agrupado["coef"].max(),
    })
    
    # Robusto = mesmo sinal em todos E significativo em todos
    resultado["robusto"] = (
        resultado["mesmo_sinal"] & 
        (resultado["significativo_em"] == resultado["cenarios"])
    )
    
    return resultado.reset_index()


# =============================================================================
# FUNÇÕES - Cenários de sensibilidade
# =============================================================================

def cenarios_sensibilidade(diarios: pd.DataFrame, base_horaria: pd.DataFrame) -> pd.DataFrame:
    """Reestima o modelo em cenários alternativos para teste de robustez.
    
    Cenários testados:
    1. principal: maio a dezembro, sem dias atípicos, HAC com 7 defasagens
    2. janelas -1h: indicadores recalculados com janelas deslocadas -1 hora
    3. janelas +1h: indicadores recalculados com janelas deslocadas +1 hora
    4. só dias úteis: exclui fins de semana e feriados
    5. com dias atípicos: inclui dias com incidentes e eventos
    6. HAC 30 defasagens: erro padrão com mais defasagens
    
    Args:
        diarios: DataFrame de indicadores diários (cenário principal).
        base_horaria: DataFrame horário (para recalcular indicadores nas
                      janelas deslocadas).
                      
    Returns:
        DataFrame consolidado com resultados de todos os cenários.
    """
    from src import indicadores as ind

    resultados = []
    
    # Cenário 1: principal
    resultados.append(estimar_todos(recorte_principal(diarios), "principal"))
    
    # Cenários 2 e 3: janelas deslocadas
    for nome, desloc in (("janelas -1h", -1), ("janelas +1h", 1)):
        janelas = {
            "janela_vale": range(ind.JANELA_VALE.start + desloc, ind.JANELA_VALE.stop + desloc),
            "janela_tarde": range(ind.JANELA_TARDE.start + desloc, ind.JANELA_TARDE.stop + desloc),
            "janela_rampa": range(ind.JANELA_RAMPA.start + desloc, ind.JANELA_RAMPA.stop + desloc),
        }
        # Recalcula indicadores com as novas janelas
        alternativo = ind.consolidar_dias_atipicos(ind.indicadores_diarios(base_horaria, **janelas))
        resultados.append(estimar_todos(recorte_principal(alternativo), nome))
    
    # Cenário 4: só dias úteis
    principal = recorte_principal(diarios)
    resultados.append(estimar_todos(principal[principal["dia_util"]], "só dias úteis"))
    
    # Cenário 5: com dias atípicos
    resultados.append(estimar_todos(diarios[diarios["mes_comparavel"]], "com dias atípicos"))
    
    # Cenário 6: HAC com mais defasagens
    resultados.append(estimar_todos(principal, "HAC 30 defasagens", maxlags=30))
    
    return pd.concat(resultados, ignore_index=True)


def comparacao_2024_2025(diarios: pd.DataFrame) -> pd.DataFrame:
    """Compara 2025 contra 2024 usando o ano inteiro.
    
    Diferente do cenário principal (que usa 2023 como base e só maio-dezembro),
    esta comparação usa 2024 como base e inclui o ano inteiro. Isso é possível
    porque ambos os anos têm cobertura solar completa.
    
    Útil para verificar se a tendência continua entre 2024 e 2025.
    
    Args:
        diarios: DataFrame de indicadores diários.
        
    Returns:
        DataFrame com os resultados da comparação 2025 × 2024.
    """
    # Filtra apenas 2024 e 2025, excluindo dias atípicos
    dados = diarios[diarios["ano"].isin([2024, 2025]) & ~diarios["dia_atipico"]]
    
    return estimar_todos(dados, "2025 × 2024, ano inteiro", ano_base=2024)
