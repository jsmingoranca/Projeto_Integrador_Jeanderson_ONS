"""Variáveis derivadas, sinalização de valores suspeitos e indicadores diários.

Este módulo é o coração da análise. Ele transforma a base horária bruta em:
1. Variáveis temporais (ano, mês, hora, dia útil, feriado)
2. Variáveis derivadas (carga residual, participação renovável)
3. Flags de qualidade (valores suspeitos, cobertura solar parcial)
4. Indicadores diários (vale, rampa, horas de carga residual baixa)

Nenhuma função altera os valores originais: as derivações e flags são colunas novas.

Janelas horárias utilizadas (hora de início do intervalo, horário de Brasília):
- JANELA_VALE: 10h a 15h — horas de maior geração solar, onde se mede o vale diurno
- JANELA_TARDE: 18h a 20h — pós-pôr do sol, onde se mede o nível da carga residual na ponta
- JANELA_RAMPA: 16h a 21h — onde se busca a maior subida horária

As janelas são escolhas analíticas (DA-007, DA-008) e têm análise de sensibilidade.

Exemplo de uso:
    >>> from src.indicadores import preparar_base_horaria, indicadores_diarios
    >>> base_horaria = preparar_base_horaria(base)
    >>> diarios = indicadores_diarios(base_horaria)
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from dateutil.easter import easter

from src.carregar_dados import COLUNAS_GERACAO

# =============================================================================
# CONSTANTES - Janelas horárias para os indicadores
# =============================================================================

# Janela do vale: 10h às 15h (6 horas de maior incidência solar)
JANELA_VALE = range(10, 16)

# Janela da tarde: 18h às 20h (3 horas após o pôr do sol)
JANELA_TARDE = range(18, 21)

# Janela da rampa: 16h às 21h (6 horas onde ocorre a maior subida)
JANELA_RAMPA = range(16, 22)

# Horas consideradas noturnas (para validar zeros de geração solar)
HORAS_NOTURNAS = [21, 22, 23, 0, 1, 2, 3, 4]

# =============================================================================
# CONSTANTES - Limiares de qualidade
# =============================================================================

# Limiar para solar noturna alta: 2% do máximo anual do subsistema
# Valores acima disso à noite indicam erro de registro
LIMIAR_SOLAR_NOTURNA = 0.02

# Limiar para solar "praticamente zero" ao meio-dia: 1% do máximo anual
# Valores abaixo disso em horário solar indicam possível falha
LIMIAR_SOLAR_DIURNA = 0.01

# Mínimo de horas consecutivas com valor idêntico para considerar "congelado"
MIN_HORAS_CONGELADA = 4

# Percentil usado para definir "carga residual baixa"
PERCENTIL_RESIDUAL_BAIXA = 0.10

# Primeiro ano completo com cobertura solar consistente
ANO_REFERENCIA = 2024

# =============================================================================
# CONSTANTES - Tratamento da quebra na série solar
# =============================================================================

# Data da mudança de patamar na série solar do ONS
# A partir de 29/04/2023 00h, a solar passa a ser contabilizada em N e S
# (antes era ~0-4 MWmed), quase triplica no SE e sobe ~50% no NE.
# Não há salto visível na carga, então a carga residual antes dessa data
# não é comparável com os anos seguintes.
QUEBRA_SOLAR = pd.Timestamp("2023-04-29 00:00")

# Subsistemas afetados pela quebra (todos, incluindo o SIN agregado)
SUBSISTEMAS_COM_QUEBRA = {"N", "NE", "S", "SE", "SIN"}

# Meses que podem ser comparados entre anos (maio a dezembro)
MESES_COMPARAVEIS = range(5, 13)

# =============================================================================
# CONSTANTES - Eventos operativos
# =============================================================================

# Eventos reais (não são erro de registro) que afetam os dados
EVENTOS_OPERATIVOS = {
    pd.Timestamp("2023-08-15"): "Apagão no SIN às 8h30, com interrupção de carga em 25 estados e no DF",
}


# =============================================================================
# FUNÇÕES - Calendário e feriados
# =============================================================================

def feriados_nacionais(anos: Iterable[int]) -> pd.DataFrame:
    """Gera lista de feriados nacionais e pontos facultativos com efeito na carga.

    Inclui feriados fixos e móveis (calculados pela data da Páscoa).
    
    Feriados fixos:
    - 01/01: Confraternização Universal
    - 21/04: Tiradentes
    - 01/05: Dia do Trabalho
    - 07/09: Independência
    - 12/10: Nossa Senhora Aparecida
    - 02/11: Finados
    - 15/11: Proclamação da República
    - 20/11: Consciência Negra (nacional a partir de 2024, Lei 14.759/2023)
    - 25/12: Natal
    
    Feriados móveis (calculados pela Páscoa):
    - Carnaval: segunda e terça antes da Quarta de Cinzas (ponto facultativo)
    - Sexta-feira Santa: sexta antes da Páscoa
    - Corpus Christi: 60 dias após a Páscoa (ponto facultativo)
    
    Args:
        anos: Lista ou iterável de anos para gerar os feriados.
        
    Returns:
        DataFrame com colunas ``data`` (Timestamp) e ``feriado_nome`` (str).
    """
    linhas = []
    
    # Feriados fixos (mês, dia): nome
    fixos = {
        (1, 1): "Confraternização Universal",
        (4, 21): "Tiradentes",
        (5, 1): "Dia do Trabalho",
        (9, 7): "Independência",
        (10, 12): "Nossa Senhora Aparecida",
        (11, 2): "Finados",
        (11, 15): "Proclamação da República",
        (12, 25): "Natal",
    }
    
    for ano in anos:
        # Adiciona feriados fixos
        for (mes, dia), nome in fixos.items():
            linhas.append((pd.Timestamp(ano, mes, dia), nome))
        
        # Consciência Negra: nacional apenas a partir de 2024
        if ano >= 2024:
            linhas.append((pd.Timestamp(ano, 11, 20), "Consciência Negra"))
        
        # Feriados móveis baseados na Páscoa
        pascoa = pd.Timestamp(easter(ano))
        linhas += [
            (pascoa - pd.Timedelta(days=48), "Carnaval (segunda)"),
            (pascoa - pd.Timedelta(days=47), "Carnaval (terça)"),
            (pascoa - pd.Timedelta(days=2), "Sexta-feira Santa"),
            (pascoa + pd.Timedelta(days=60), "Corpus Christi"),
        ]
    
    return pd.DataFrame(linhas, columns=["data", "feriado_nome"]).sort_values("data", ignore_index=True)


def classificar_periodo(hora: int) -> str:
    """Classifica uma hora do dia em período para análise.
    
    Args:
        hora: Hora do dia (0-23).
        
    Returns:
        Uma das três categorias:
        - "horas_solares": 9h às 15h (maior incidência solar)
        - "ponta_noturna": 18h às 21h (pico de demanda)
        - "demais_horas": restante do dia
    """
    if 9 <= hora < 16:
        return "horas_solares"
    if 18 <= hora < 22:
        return "ponta_noturna"
    return "demais_horas"


# =============================================================================
# FUNÇÕES - Preparação da base horária
# =============================================================================

def adicionar_tempo(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta variáveis temporais derivadas do instante.
    
    Adiciona as seguintes colunas:
    - ano, mes, hora: extraídos do instante
    - data: data sem hora (para agregação diária)
    - dia_semana: 0=segunda a 6=domingo
    - fim_semana: True se sábado ou domingo
    - periodo_dia: classificação em horas_solares/ponta_noturna/demais
    - feriado: True se a data é feriado nacional
    - dia_util: True se não é fim de semana nem feriado
    - evento_operativo: True se houve evento operativo conhecido
    
    Args:
        df: DataFrame com coluna ``din_instante``.
        
    Returns:
        Cópia do DataFrame com as novas colunas.
    """
    out = df.copy()
    instante = out["din_instante"]
    
    # Extrai componentes temporais
    out["ano"] = instante.dt.year
    out["mes"] = instante.dt.month
    out["hora"] = instante.dt.hour
    out["data"] = instante.dt.normalize()  # Data sem hora
    out["dia_semana"] = instante.dt.dayofweek  # 0=seg, 6=dom
    out["fim_semana"] = out["dia_semana"] >= 5
    
    # Classifica período do dia
    out["periodo_dia"] = out["hora"].map(classificar_periodo)
    
    # Marca feriados
    feriados = feriados_nacionais(sorted(out["ano"].unique()))
    out["feriado"] = out["data"].isin(feriados["data"])
    
    # Dia útil = não é fim de semana nem feriado
    out["dia_util"] = ~out["fim_semana"] & ~out["feriado"]
    
    # Marca eventos operativos conhecidos
    out["evento_operativo"] = out["data"].isin(list(EVENTOS_OPERATIVOS))
    
    return out


def adicionar_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variáveis derivadas: geração centralizada, carga residual e participação renovável.
    
    Variáveis criadas:
    - geracao_centralizada: soma de hidráulica + térmica + eólica + solar
    - eolica_solar: soma de eólica + solar (fontes variáveis)
    - carga_residual: carga − eólica − solar (o que sobra para fontes controláveis)
    - participacao_eol_sol: fração da geração que é eólica + solar
    
    A carga residual é o indicador central desta análise: representa a demanda
    que precisa ser atendida por fontes controláveis (hidráulicas e térmicas).
    
    Args:
        df: DataFrame com colunas de geração e carga.
        
    Returns:
        Cópia do DataFrame com as novas colunas.
    """
    out = df.copy()
    
    # Soma toda a geração centralizada
    out["geracao_centralizada"] = out[COLUNAS_GERACAO].sum(axis=1)
    
    # Soma das fontes variáveis (eólica + solar)
    out["eolica_solar"] = out["val_gereolica"] + out["val_gersolar"]
    
    # Carga residual: o que sobra após descontar eólica e solar
    out["carga_residual"] = out["val_carga"] - out["eolica_solar"]
    
    # Participação das fontes variáveis na geração total
    out["participacao_eol_sol"] = out["eolica_solar"] / out["geracao_centralizada"]
    
    return out


# =============================================================================
# FUNÇÕES - Sinalização de valores suspeitos
# =============================================================================

def _tamanho_sequencia_igual(serie: pd.Series) -> pd.Series:
    """Calcula, para cada posição, o tamanho da sequência de valores idênticos consecutivos.
    
    Útil para detectar valores "congelados" (mesmo valor repetido por muitas horas).
    
    Args:
        serie: Série de valores numéricos.
        
    Returns:
        Série com o tamanho da sequência para cada posição.
    """
    # Cria grupos onde valores consecutivos iguais têm o mesmo número
    grupo = (serie != serie.shift()).cumsum()
    # Conta o tamanho de cada grupo
    return serie.groupby(grupo).transform("size")


def sinalizar_suspeitos(df: pd.DataFrame) -> pd.DataFrame:
    """Cria flags horárias para valores suspeitos na geração solar.
    
    IMPORTANTE: Esta função NÃO altera valores, apenas cria colunas de sinalização.
    
    Flags criadas:
    - cobertura_solar_parcial: antes de QUEBRA_SOLAR (29/04/2023), em todos os
      subsistemas. Não é incidente, é mudança de cobertura da série.
    
    - flag_solar_noturna_alta: solar > 2% do máximo anual entre 21h e 4h.
      Indica possível erro de registro (geração solar à noite).
    
    - flag_solar_zero_diurna: solar <= 1% do máximo anual entre 10h e 15h.
      Indica possível falha de registro em horário de plena geração.
    
    - flag_solar_congelada: mesmo valor por >= 4 horas seguidas, acima de 2%
      do máximo. Indica possível travamento do sistema de medição.
    
    - flag_suspeito: True se qualquer flag acima (exceto cobertura_parcial) for True.
    
    Args:
        df: DataFrame já processado com ``adicionar_tempo``, ordenado por
            subsistema e instante.
            
    Returns:
        Cópia do DataFrame com as colunas de flags.
    """
    out = df.sort_values(["id_subsistema", "din_instante"]).copy()
    
    # Calcula o máximo anual da solar por subsistema (referência para os limiares)
    maximo_anual = out.groupby(
        ["id_subsistema", "ano"], observed=True
    )["val_gersolar"].transform("max")
    
    solar = out["val_gersolar"]
    
    # Flag 1: Solar noturna acima do limiar
    out["flag_solar_noturna_alta"] = (
        out["hora"].isin(HORAS_NOTURNAS) & 
        (solar > LIMIAR_SOLAR_NOTURNA * maximo_anual)
    )
    
    # Flag 2: Cobertura solar parcial (antes da quebra de abril/2023)
    out["cobertura_solar_parcial"] = (
        out["id_subsistema"].astype(str).isin(SUBSISTEMAS_COM_QUEBRA) & 
        (out["din_instante"] < QUEBRA_SOLAR)
    )
    
    # Flag 3: Solar praticamente zero em horário de pico solar
    # (exclui período de cobertura parcial, onde zeros eram esperados)
    out["flag_solar_zero_diurna"] = (
        out["hora"].isin(JANELA_VALE) &
        (solar <= LIMIAR_SOLAR_DIURNA * maximo_anual) &
        ~out["cobertura_solar_parcial"]
    )
    
    # Flag 4: Solar "congelada" (mesmo valor por muitas horas)
    sequencia = out.groupby(
        "id_subsistema", observed=True
    )["val_gersolar"].transform(_tamanho_sequencia_igual)
    
    out["flag_solar_congelada"] = (
        (sequencia >= MIN_HORAS_CONGELADA) & 
        (solar > LIMIAR_SOLAR_NOTURNA * maximo_anual)
    )
    
    # Flag consolidada: qualquer suspeita (exceto cobertura parcial)
    out["flag_suspeito"] = out[
        ["flag_solar_noturna_alta", "flag_solar_zero_diurna", "flag_solar_congelada"]
    ].any(axis=1)
    
    return out


def resumo_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Conta horas sinalizadas por subsistema, ano e tipo de flag.
    
    Args:
        df: DataFrame com as colunas de flags.
        
    Returns:
        DataFrame agregado com contagem de cada tipo de flag.
    """
    colunas = [
        "flag_solar_noturna_alta",
        "flag_solar_zero_diurna", 
        "flag_solar_congelada",
        "flag_suspeito"
    ]
    return df.groupby(["id_subsistema", "ano"], observed=True)[colunas].sum()


# =============================================================================
# FUNÇÕES - Carga residual baixa
# =============================================================================

def limiares_residual_baixa(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o limiar de "carga residual baixa" por subsistema e mês.
    
    O limiar é o percentil 10 (P10) da carga residual no ANO_REFERENCIA (2024),
    calculado separadamente para cada subsistema e mês. Usamos 2024 como
    referência porque é o primeiro ano completo com cobertura solar consistente.
    
    Uma hora é considerada de "carga residual baixa" se estiver abaixo desse
    limiar. Isso indica condições propícias a excesso de oferta renovável.
    
    Args:
        df: DataFrame com colunas ``ano``, ``id_subsistema``, ``mes`` e ``carga_residual``.
        
    Returns:
        DataFrame com colunas ``id_subsistema``, ``mes`` e ``limiar_residual_baixa``.
    """
    # Filtra apenas o ano de referência
    base = df[df["ano"] == ANO_REFERENCIA]
    
    return (
        base.groupby(["id_subsistema", "mes"], observed=True)["carga_residual"]
        .quantile(PERCENTIL_RESIDUAL_BAIXA)
        .rename("limiar_residual_baixa")
        .reset_index()
    )


def adicionar_residual_baixa(df: pd.DataFrame, limiares: pd.DataFrame) -> pd.DataFrame:
    """Adiciona flag indicando se cada hora está abaixo do limiar de carga residual baixa.
    
    Args:
        df: DataFrame horário com colunas ``id_subsistema``, ``mes`` e ``carga_residual``.
        limiares: DataFrame com os limiares por subsistema e mês.
        
    Returns:
        DataFrame com colunas ``limiar_residual_baixa`` e ``residual_baixa`` (bool).
        
    Raises:
        ValueError: Se houver combinações subsistema × mês sem limiar definido.
    """
    out = df.merge(limiares, on=["id_subsistema", "mes"], how="left", validate="many_to_one")
    
    if out["limiar_residual_baixa"].isna().any():
        raise ValueError("há combinações subsistema × mês sem limiar no ano de referência")
    
    out["residual_baixa"] = out["carga_residual"] < out["limiar_residual_baixa"]
    return out


def preparar_base_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de preparação da base horária.
    
    Executa em sequência:
    1. adicionar_tempo: variáveis temporais
    2. adicionar_derivadas: carga residual e participação renovável
    3. sinalizar_suspeitos: flags de qualidade
    4. adicionar_residual_baixa: indicador de carga residual baixa
    
    Args:
        df: DataFrame bruto da base ONS.
        
    Returns:
        DataFrame completo com todas as variáveis derivadas e flags.
    """
    base = sinalizar_suspeitos(adicionar_derivadas(adicionar_tempo(df)))
    base["id_subsistema"] = base["id_subsistema"].astype(str)
    base = adicionar_residual_baixa(base, limiares_residual_baixa(base))
    return base.sort_values(["id_subsistema", "din_instante"], ignore_index=True)


# =============================================================================
# FUNÇÕES - Indicadores diários
# =============================================================================

def _maior_rampa(grupo: pd.DataFrame, janela_rampa: range = JANELA_RAMPA) -> float:
    """Calcula a maior rampa horária positiva dentro da janela especificada.
    
    A rampa é a diferença entre a carga residual de uma hora e a anterior.
    Retorna o maior valor positivo (maior subida) na janela.
    
    Args:
        grupo: DataFrame de um único dia com colunas ``carga_residual`` e ``hora``.
        janela_rampa: Range de horas a considerar.
        
    Returns:
        Maior diferença positiva hora a hora (MWmed/h).
    """
    diferencas = grupo["carga_residual"].diff()
    # Considera apenas as horas dentro da janela (exceto a primeira, que não tem diferença)
    return diferencas[grupo["hora"].isin(list(janela_rampa)[1:])].max()


def indicadores_diarios(
    base: pd.DataFrame,
    janela_vale: range = JANELA_VALE,
    janela_tarde: range = JANELA_TARDE,
    janela_rampa: range = JANELA_RAMPA,
) -> pd.DataFrame:
    """Calcula indicadores agregados por dia e subsistema.
    
    Requer que cada dia tenha exatamente 24 horas (sem lacunas).
    
    As janelas podem ser alteradas para análise de sensibilidade (DA-007, DA-008).
    
    Indicadores calculados:
    - Metadados: ano, mes, dia_semana, fim_semana, feriado, dia_util, evento_operativo
    - carga_media_dia: média da carga observada no dia
    - carga_residual_media_dia: média da carga residual no dia
    - horas_residual_baixa: quantas horas do dia ficaram abaixo do P10
    - dia_suspeito: True se alguma hora teve flag de suspeita
    - cobertura_solar_parcial: True se o dia está antes da quebra de abril/2023
    - participacao_eol_sol: fração eólica+solar na geração do dia
    - carga_media_vale: média da carga na janela do vale (10-15h)
    - indice_vale: razão carga_media_vale / carga_media_dia
    - carga_residual_min: mínimo da carga residual na janela do vale
    - hora_residual_min: hora em que ocorreu o mínimo
    - intercambio_medio_vale: média do intercâmbio na janela do vale
    - carga_residual_tarde: média da carga residual na janela da tarde (18-20h)
    - rampa_tarde: diferença entre carga_residual_tarde e carga_residual_min
    - maior_rampa_horaria: maior subida hora a hora na janela da rampa (16-21h)
    - mes_comparavel: True se o mês está entre maio e dezembro
    
    Args:
        base: DataFrame horário preparado por ``preparar_base_horaria``.
        janela_vale: Range de horas para o indicador de vale.
        janela_tarde: Range de horas para o indicador da tarde.
        janela_rampa: Range de horas para buscar a maior rampa.
        
    Returns:
        DataFrame com uma linha por dia × subsistema.
        
    Raises:
        ValueError: Se algum dia não tiver exatamente 24 horas.
    """
    base = base.sort_values(["id_subsistema", "din_instante"])
    chaves = ["id_subsistema", "data"]
    
    # Verifica que cada dia tem 24 horas
    horas = base.groupby(chaves).size()
    if not (horas == 24).all():
        raise ValueError("há dias sem 24 horas")

    # Agregações do dia inteiro
    dia = base.groupby(chaves).agg(
        ano=("ano", "first"),
        mes=("mes", "first"),
        dia_semana=("dia_semana", "first"),
        fim_semana=("fim_semana", "first"),
        feriado=("feriado", "first"),
        dia_util=("dia_util", "first"),
        evento_operativo=("evento_operativo", "first"),
        carga_media_dia=("val_carga", "mean"),
        carga_residual_media_dia=("carga_residual", "mean"),
        horas_residual_baixa=("residual_baixa", "sum"),
        dia_suspeito=("flag_suspeito", "any"),
        cobertura_solar_parcial=("cobertura_solar_parcial", "any"),
        eolica_solar_dia=("eolica_solar", "sum"),
        geracao_dia=("geracao_centralizada", "sum"),
    )
    
    # Participação renovável no dia
    dia["participacao_eol_sol"] = dia.pop("eolica_solar_dia") / dia.pop("geracao_dia")

    # Indicadores da janela do vale (10-15h)
    vale = base[base["hora"].isin(janela_vale)]
    dia["carga_media_vale"] = vale.groupby(chaves)["val_carga"].mean()
    dia["indice_vale"] = dia["carga_media_vale"] / dia["carga_media_dia"]
    
    # Mínimo da carga residual no vale e a hora em que ocorre
    idx_min = vale.groupby(chaves)["carga_residual"].idxmin()
    dia["carga_residual_min"] = vale.loc[idx_min.values, "carga_residual"].values
    dia["hora_residual_min"] = vale.loc[idx_min.values, "hora"].values
    dia["intercambio_medio_vale"] = vale.groupby(chaves)["val_intercambio"].mean()

    # Indicadores da janela da tarde (18-20h)
    tarde = base[base["hora"].isin(janela_tarde)]
    dia["carga_residual_tarde"] = tarde.groupby(chaves)["carga_residual"].mean()
    
    # Rampa da tarde: subida do mínimo até o nível da noite
    dia["rampa_tarde"] = dia["carga_residual_tarde"] - dia["carga_residual_min"]

    # Maior rampa horária na janela de rampa (16-21h)
    rampa = base[base["hora"].isin(janela_rampa)]
    dia["maior_rampa_horaria"] = rampa.groupby(chaves)[["carga_residual", "hora"]].apply(
        _maior_rampa, janela_rampa=janela_rampa
    )
    
    # Marca meses comparáveis entre anos (maio a dezembro)
    dia["mes_comparavel"] = dia["mes"].isin(MESES_COMPARAVEIS)
    
    return dia.reset_index()


def consolidar_dias_atipicos(diarios: pd.DataFrame) -> pd.DataFrame:
    """Propaga os incidentes de registro para todos os níveis e cria ``dia_atipico``.
    
    Um incidente na solar de um subsistema contamina a carga dele (porque a
    carga fecha pelo balanço) e, portanto, também afeta o SIN. Por cautela,
    a data inteira é tratada como suspeita em todos os níveis.
    
    O campo ``dia_atipico`` combina:
    - dia_suspeito: incidente de registro detectado
    - evento_operativo: evento real documentado (ex: apagão)
    
    Feriados NÃO entram em dia_atipico (são usados como controle no modelo).
    
    Args:
        diarios: DataFrame de indicadores diários.
        
    Returns:
        DataFrame com ``dia_suspeito`` propagado e ``dia_atipico`` calculado.
    """
    out = diarios.copy()
    
    # Propaga suspeita: se qualquer nível tem suspeita na data, todos têm
    datas_suspeitas = out.loc[out["dia_suspeito"], "data"].unique()
    out["dia_suspeito"] = out["data"].isin(datas_suspeitas)
    
    # Dia atípico = suspeito ou evento operativo (feriados ficam de fora)
    out["dia_atipico"] = out["dia_suspeito"] | out["evento_operativo"]
    
    return out
