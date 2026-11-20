"""Diagnóstico de qualidade da base Balanço de Energia nos Subsistemas.

Este módulo fornece funções para verificar a integridade e qualidade dos dados:
- Contagem de registros por arquivo e subsistema
- Verificação da grade temporal (horas faltantes, lacunas)
- Resumo de valores ausentes, negativos e zeros
- Detecção de duplicidades
- Validação da identidade do balanço (carga = geração - intercâmbio)
- Comparação entre SIN publicado e soma dos subsistemas

Cada função devolve tabelas (DataFrames) para que o notebook mostre e exporte
os resultados; nenhuma função altera a base recebida.

Exemplo de uso:
    >>> from src.qualidade_dados import resumo_qualidade, verificar_grade_temporal
    >>> qualidade = resumo_qualidade(base)
    >>> grade = verificar_grade_temporal(base, "2023-01-01", "2025-12-31")
"""

from __future__ import annotations

import pandas as pd

from src.carregar_dados import COLUNAS_GERACAO, COLUNAS_VALORES, ID_SIN

# Colunas que formam a chave única de cada registro
CHAVE = ["id_subsistema", "din_instante"]


def contar_por_arquivo(df: pd.DataFrame) -> pd.DataFrame:
    """Conta linhas por arquivo de origem e subsistema.
    
    Útil para verificar se cada arquivo anual tem o número esperado de registros
    e se todos os subsistemas estão presentes em cada ano.
    
    Args:
        df: DataFrame com coluna ``arquivo_origem``.
        
    Returns:
        Tabela pivotada com arquivos nas linhas e subsistemas nas colunas.
    """
    return (
        df.groupby(["arquivo_origem", "id_subsistema"], observed=True)
        .size()
        .unstack("id_subsistema")
    )


def horas_esperadas(inicio: str, fim: str) -> pd.DatetimeIndex:
    """Gera a grade horária completa entre ``inicio`` e ``fim`` (inclusive).
    
    Args:
        inicio: Data/hora inicial no formato "YYYY-MM-DD" ou "YYYY-MM-DD HH:MM:SS".
        fim: Data/hora final no mesmo formato.
        
    Returns:
        DatetimeIndex com todas as horas do período.
        
    Exemplo:
        >>> grade = horas_esperadas("2023-01-01", "2023-01-02")
        >>> len(grade)  # 25 horas (00h do dia 1 até 00h do dia 2)
    """
    return pd.date_range(inicio, fim, freq="h")


def verificar_grade_temporal(df: pd.DataFrame, inicio: str, fim: str) -> pd.DataFrame:
    """Verifica a completude da grade temporal por subsistema.
    
    Para cada subsistema, compara os instantes presentes com a grade esperada
    e identifica horas faltantes, registros fora da grade e passos irregulares.
    
    Args:
        df: DataFrame com colunas ``id_subsistema`` e ``din_instante``.
        inicio: Data/hora inicial esperada.
        fim: Data/hora final esperada.
        
    Returns:
        DataFrame com uma linha por subsistema contendo:
        - registros: número de linhas presentes
        - esperados: número de horas na grade
        - horas_faltantes: horas da grade sem registro
        - fora_da_grade: registros com horário fora da grade
        - passos_diferentes_de_1h: transições não-horárias
    """
    grade = horas_esperadas(inicio, fim)
    linhas = []
    
    for sub, grupo in df.groupby("id_subsistema", observed=True):
        instantes = pd.DatetimeIndex(grupo["din_instante"]).sort_values()
        
        # Calcula os passos entre instantes consecutivos
        passos = instantes.to_series().diff().dropna()
        
        linhas.append({
            "id_subsistema": sub,
            "registros": len(instantes),
            "esperados": len(grade),
            "horas_faltantes": len(grade.difference(instantes)),
            "fora_da_grade": int((~instantes.isin(grade)).sum()),
            "passos_diferentes_de_1h": int((passos != pd.Timedelta(hours=1)).sum()),
        })
    
    return pd.DataFrame(linhas).set_index("id_subsistema")


def resumo_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    """Gera tabela-resumo de qualidade das variáveis numéricas.
    
    Para cada variável numérica (val_*), conta valores ausentes, negativos
    e zeros, além de mostrar o mínimo e máximo observados.
    
    Args:
        df: DataFrame com as colunas de valores (val_*).
        
    Returns:
        DataFrame com uma linha por variável e colunas: ausentes, negativos,
        zeros, minimo, maximo.
    """
    return pd.DataFrame({
        "ausentes": df[COLUNAS_VALORES].isna().sum(),
        "negativos": (df[COLUNAS_VALORES] < 0).sum(),
        "zeros": (df[COLUNAS_VALORES] == 0).sum(),
        "minimo": df[COLUNAS_VALORES].min(),
        "maximo": df[COLUNAS_VALORES].max(),
    })


def contar_duplicidades(df: pd.DataFrame) -> dict[str, int]:
    """Conta duplicidades completas e da chave subsistema+instante.
    
    Args:
        df: DataFrame completo.
        
    Returns:
        Dicionário com:
        - linhas_duplicadas: linhas completamente idênticas (ignorando origem)
        - chave_duplicada: linhas com mesma chave (subsistema + instante)
    """
    sem_origem = df.drop(columns=["arquivo_origem"], errors="ignore")
    return {
        "linhas_duplicadas": int(sem_origem.duplicated().sum()),
        "chave_duplicada": int(df.duplicated(CHAVE).sum()),
    }


def mapa_id_nome(df: pd.DataFrame) -> pd.DataFrame:
    """Lista as combinações de id_subsistema e nom_subsistema.
    
    Útil para verificar se cada id tem um único nome associado,
    detectando possíveis inconsistências nos dados.
    
    Args:
        df: DataFrame com colunas ``id_subsistema`` e ``nom_subsistema``.
        
    Returns:
        DataFrame com colunas: id_subsistema, nom_subsistema, registros.
    """
    return (
        df.groupby(["id_subsistema", "nom_subsistema"], observed=True)
        .size()
        .rename("registros")
        .reset_index()
    )


def comparar_sin_com_soma(df: pd.DataFrame) -> pd.DataFrame:
    """Compara o SIN publicado com a soma dos quatro subsistemas.
    
    O SIN deveria ser exatamente igual à soma dos subsistemas. Diferenças
    indicam erros de arredondamento ou problemas nos dados.
    
    Args:
        df: DataFrame com todos os registros (SIN e subsistemas).
        
    Returns:
        DataFrame com diferença média e máxima para cada variável numérica.
    """
    # Separa os registros do SIN
    sin = df[df["id_subsistema"] == ID_SIN].set_index("din_instante")[COLUNAS_VALORES]
    
    # Soma os quatro subsistemas
    soma = (
        df[df["id_subsistema"] != ID_SIN]
        .groupby("din_instante")[COLUNAS_VALORES]
        .sum()
        .reindex(sin.index)
    )
    
    # Calcula a diferença absoluta
    diferenca = (sin - soma).abs()
    
    return pd.DataFrame({
        "diferenca_media": diferenca.mean(),
        "diferenca_maxima": diferenca.max()
    })


def testar_identidade_balanco(df: pd.DataFrame) -> pd.DataFrame:
    """Testa as duas convenções de sinal possíveis para o intercâmbio.

    O balanço de energia deve fechar: carga = geração ± intercâmbio.
    Existem duas convenções possíveis para o sinal do intercâmbio:
    
    - Convenção A: carga = geração − intercâmbio
      (intercâmbio positivo = exportação líquida, energia saindo do subsistema)
    
    - Convenção B: carga = geração + intercâmbio
      (intercâmbio positivo = importação líquida, energia entrando no subsistema)
    
    O resíduo próximo de zero indica qual convenção os dados seguem.
    
    Args:
        df: DataFrame com todos os registros.
        
    Returns:
        DataFrame com média e máximo do resíduo absoluto para cada convenção,
        por subsistema.
    """
    # Usa apenas os subsistemas (não o SIN, que é agregado)
    subs = df[df["id_subsistema"] != ID_SIN]
    
    # Soma toda a geração centralizada
    geracao = subs[COLUNAS_GERACAO].sum(axis=1)
    
    # Testa as duas convenções
    residuo_a = (geracao - subs["val_intercambio"] - subs["val_carga"]).abs()
    residuo_b = (geracao + subs["val_intercambio"] - subs["val_carga"]).abs()
    
    return (
        pd.DataFrame({
            "id_subsistema": subs["id_subsistema"],
            "residuo_abs_A_ger_menos_interc": residuo_a,
            "residuo_abs_B_ger_mais_interc": residuo_b,
        })
        .groupby("id_subsistema", observed=True)
        .agg(["mean", "max"])
        .round(3)
    )


def zeros_solares_por_hora(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o percentual de horas com geração solar zero, por hora e subsistema.
    
    Útil para identificar padrões de zeros noturnos (esperados) e diurnos
    (potencialmente problemáticos).
    
    Args:
        df: DataFrame com colunas ``din_instante``, ``id_subsistema`` e ``val_gersolar``.
        
    Returns:
        DataFrame pivotado com horas (0-23) nas linhas e subsistemas nas colunas,
        valores em percentual.
    """
    # Exclui o SIN (é agregado dos subsistemas)
    subs = df[df["id_subsistema"] != ID_SIN]
    
    return (
        subs.assign(
            hora=subs["din_instante"].dt.hour,
            zero=subs["val_gersolar"] == 0
        )
        .pivot_table(
            index="hora",
            columns="id_subsistema",
            values="zero",
            aggfunc="mean",
            observed=True
        )
        .mul(100)  # Converte para percentual
        .round(1)
    )
