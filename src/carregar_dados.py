"""Leitura da base comum Balanço de Energia nos Subsistemas (ONS, 2023-2025).

Este módulo é responsável por:
1. Localizar e ler os arquivos CSV anuais do ONS
2. Validar o esquema (colunas esperadas)
3. Concatenar os anos em uma única base
4. Separar o agregado SIN dos quatro subsistemas

Os arquivos de ``dados/brutos/`` são apenas lidos; nada é gravado nessa pasta.

Exemplo de uso:
    >>> from src.carregar_dados import carregar_base, separar_niveis
    >>> base = carregar_base()
    >>> sin, subsistemas = separar_niveis(base)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Caminho para a raiz do projeto (um nível acima de src/)
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_BRUTOS = RAIZ_PROJETO / "dados" / "brutos"
PADRAO_ARQUIVOS = "BALANCO_ENERGIA_SUBSISTEMA_*.csv"

# Colunas que devem estar presentes em cada arquivo CSV
COLUNAS_ESPERADAS = [
    "id_subsistema",      # Código do subsistema (N, NE, S, SE, SIN)
    "nom_subsistema",     # Nome por extenso do subsistema
    "din_instante",       # Data e hora (início do intervalo)
    "val_gerhidraulica",  # Geração hidráulica (MWmed)
    "val_gertermica",     # Geração térmica (MWmed)
    "val_gereolica",      # Geração eólica (MWmed)
    "val_gersolar",       # Geração solar fotovoltaica (MWmed)
    "val_carga",          # Carga verificada (MWmed)
    "val_intercambio",    # Intercâmbio líquido (MWmed, positivo = exportação)
]

# Subconjuntos úteis para operações posteriores
COLUNAS_VALORES = [c for c in COLUNAS_ESPERADAS if c.startswith("val_")]
COLUNAS_GERACAO = ["val_gerhidraulica", "val_gertermica", "val_gereolica", "val_gersolar"]

# Identificador do Sistema Interligado Nacional (agregado)
ID_SIN = "SIN"


def listar_arquivos(pasta: Path = PASTA_BRUTOS) -> list[Path]:
    """Retorna os CSVs anuais em ordem de nome (ano).
    
    Args:
        pasta: Diretório onde estão os arquivos. Padrão: dados/brutos/
        
    Returns:
        Lista de caminhos dos arquivos encontrados, ordenados alfabeticamente.
        
    Raises:
        FileNotFoundError: Se nenhum arquivo correspondente ao padrão for encontrado.
    """
    arquivos = sorted(Path(pasta).glob(PADRAO_ARQUIVOS))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo {PADRAO_ARQUIVOS} em {pasta}")
    return arquivos


def ler_arquivo(caminho: Path) -> pd.DataFrame:
    """Lê um CSV anual (UTF-8, ';') e valida o esquema.
    
    Args:
        caminho: Caminho completo do arquivo CSV.
        
    Returns:
        DataFrame com os dados do arquivo.
        
    Raises:
        ValueError: Se alguma coluna esperada estiver ausente.
    """
    df = pd.read_csv(caminho, sep=";", encoding="utf-8")
    
    # Verifica se todas as colunas esperadas estão presentes
    faltantes = set(COLUNAS_ESPERADAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"{caminho.name}: colunas ausentes {sorted(faltantes)}")
    return df


def carregar_base(pasta: Path = PASTA_BRUTOS) -> pd.DataFrame:
    """Concatena os arquivos anuais, converte ``din_instante`` e ordena.

    Esta é a função principal de entrada. Ela:
    1. Lista todos os arquivos CSV que seguem o padrão
    2. Lê e valida cada arquivo
    3. Concatena em uma única base
    4. Converte a coluna de data/hora para datetime
    5. Converte colunas categóricas para otimizar memória
    
    Args:
        pasta: Diretório onde estão os arquivos. Padrão: dados/brutos/
        
    Returns:
        DataFrame com todos os anos concatenados, ordenado por subsistema e instante.
        Inclui coluna ``arquivo_origem`` para rastreabilidade.
    """
    partes = []
    for caminho in listar_arquivos(pasta):
        parte = ler_arquivo(caminho)
        # Registra a origem de cada linha para rastreabilidade
        parte["arquivo_origem"] = caminho.name
        partes.append(parte)
    
    df = pd.concat(partes, ignore_index=True)
    
    # Converte a coluna de data/hora
    df["din_instante"] = pd.to_datetime(df["din_instante"], format="%Y-%m-%d %H:%M:%S")
    
    # Converte colunas categóricas para economizar memória
    for coluna in ("id_subsistema", "nom_subsistema", "arquivo_origem"):
        df[coluna] = df[coluna].astype("category")
    
    return df.sort_values(["id_subsistema", "din_instante"], ignore_index=True)


def separar_niveis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa o agregado SIN dos quatro subsistemas (evita dupla contagem).
    
    O SIN é a soma exata dos quatro subsistemas (N, NE, S, SE/CO). Por isso,
    análises que somam valores devem usar apenas um dos níveis, não ambos.
    
    Args:
        df: DataFrame com todos os registros (SIN e subsistemas).
        
    Returns:
        Tupla com dois DataFrames:
        - sin: Apenas os registros do agregado nacional
        - subsistemas: Apenas os registros dos 4 subsistemas regionais
    """
    eh_sin = df["id_subsistema"] == ID_SIN
    
    sin = df.loc[eh_sin].reset_index(drop=True)
    subsistemas = df.loc[~eh_sin].reset_index(drop=True)
    
    # Remove categorias não usadas para evitar confusão
    subsistemas["id_subsistema"] = subsistemas["id_subsistema"].cat.remove_unused_categories()
    subsistemas["nom_subsistema"] = subsistemas["nom_subsistema"].cat.remove_unused_categories()
    
    return sin, subsistemas
