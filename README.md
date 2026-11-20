# Vale diurno, rampa do fim da tarde e carga residual no SIN (2023–2025)

Projeto Integrador · Treinamento em Análise de Dados – Analytics (Pecege/ONS) · **Jeanderson Soares Mingorança** · entrega em 24/11/2026.

## Pergunta analítica

> Entre 2023 e 2025, como se modificaram o vale diurno e a rampa do fim da tarde da carga observada e da carga residual (carga − geração eólica − geração solar) no SIN e nos subsistemas, e em quais meses, horários e subsistemas essas mudanças foram mais intensas, para orientar a priorização de estudos de flexibilidade operativa?

**Público:** equipes de operação, planejamento e integração de renováveis. **Decisão apoiada:** quais subsistemas, meses e horários priorizar em estudos de flexibilidade (armazenamento, intercâmbio e resposta da demanda).

## Resposta em uma frase

No SIN, de 2023 para 2025, comparando dias do mesmo mês e tipo, **a carga residual do meio-dia caiu 8,6 GW e a rampa do fim da tarde cresceu 9,7 GW (+45%)**, e as horas diárias de carga residual baixa passaram de 1,8 para 4,9. A mudança se concentra no **Sudeste/Centro-Oeste e no Nordeste**, de **julho a outubro**, entre **10h–14h** e **17h–20h**. O padrão é compatível com a expansão da geração solar; a base não mede a MMGD nem os cortes de geração.

## Glossário

| Termo | Significado |
|-------|-------------|
| **SIN** | Sistema Interligado Nacional — a rede elétrica que conecta as usinas aos consumidores no Brasil |
| **Subsistema** | Divisão regional do SIN: Norte (N), Nordeste (NE), Sul (S) e Sudeste/Centro-Oeste (SE) |
| **Carga** | Demanda de energia elétrica medida em MWmed (megawatts médios por hora) |
| **Carga residual** | Carga total menos a geração eólica e solar (fontes variáveis); representa o que as fontes despacháveis (hidrelétricas, térmicas) precisam suprir. Também chamada de *net load* na literatura internacional (conceito popularizado pelo CAISO em 2012) |
| **Vale diurno** | Período do meio-dia (10h–15h) em que a carga residual atinge seu mínimo, devido à geração solar |
| **Rampa do fim da tarde** | Subida acentuada da carga residual entre 17h e 20h, quando o sol se põe e a demanda aumenta |
| **MMGD** | Micro e minigeração distribuída — painéis solares em telhados de casas e comércios |
| **Curtailment** | Corte de geração renovável quando há excesso de oferta (a base não mede isso) |
| **MQO** | Mínimos Quadrados Ordinários — método estatístico de regressão linear |
| **HAC** | Heteroskedasticity and Autocorrelation Consistent — correção para erros correlacionados no tempo |
| **P10** | Percentil 10 — valor abaixo do qual estão 10% das observações |

## Entregáveis

| Entregável | Arquivo |
|---|---|
| **Notebook principal** (executável do início ao fim) | `notebooks/JeandersonMingoranca_ProjetoIntegrador.ipynb` |
| **Documento do Projeto Integrador** | `docs/JeandersonMingoranca_ProjetoIntegrador_Revisado.docx` |
| Notebooks de desenvolvimento | `notebooks/01_*.ipynb` a `05_*.ipynb` |
| Figuras | `resultados/figuras/*.png` |
| Tabelas de resultados | `resultados/tabelas/*.csv` |
| Instruções e dependências | este `README.md` e `requirements.txt` |

## Início rápido

### Pré-requisitos

- **Python 3.13** ou superior instalado
- Acesso ao terminal (PowerShell no Windows, Terminal no macOS/Linux)
- Cerca de 500 MB de espaço em disco

### Passo a passo

**1. Abra o terminal na pasta do projeto**

No Windows, navegue até a pasta e clique com Shift + botão direito → "Abrir janela do PowerShell aqui".

**2. Crie um ambiente virtual**

O ambiente virtual isola as dependências do projeto, evitando conflitos com outros projetos Python.

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# Linux ou macOS
python -m venv .venv
source .venv/bin/activate
```

Após ativar, você verá `(.venv)` no início da linha de comando.

**3. Instale as dependências**

```bash
python -m pip install -r requirements.txt
```

Isso instala pandas, matplotlib, statsmodels e as outras bibliotecas necessárias (cerca de 2 minutos).

**4. Execute o notebook**

Abra o JupyterLab:

```bash
jupyter lab
```

Ou abra diretamente no VS Code (extensão Jupyter necessária).

Navegue até `notebooks/JeandersonMingoranca_ProjetoIntegrador.ipynb` e execute:
- **Kernel → Restart Kernel and Run All Cells**

O notebook leva menos de 1 minuto para executar e gera automaticamente:
- `dados/processados/` — bases intermediárias em formato Parquet
- `resultados/figuras/` — gráficos da análise
- `resultados/tabelas/` — tabelas com os resultados numéricos

**5. (Opcional) Verifique os resultados**

Após a execução, confira as pastas `resultados/figuras/` e `resultados/tabelas/` para ver os gráficos e tabelas gerados.

## Dados

- **Fonte:** Operador Nacional do Sistema Elétrico (ONS), *Balanço de Energia nos Subsistemas*, Portal de Dados Abertos, licença CC BY: https://dados.ons.org.br/dataset/balanco-energia-subsistema
- **Arquivos:** `dados/brutos/BALANCO_ENERGIA_SUBSISTEMA_2023.csv`, `_2024.csv`, `_2025.csv` (UTF-8, separador `;`) e o dicionário de dados. **Não são alterados** pelo código.
- **Unidade de observação:** uma hora × um subsistema (N, NE, S, SE/CO) ou o agregado SIN; 131.520 linhas, sem ausentes, duplicidades ou lacunas.
- **Derivados:** `dados/processados/*.parquet` (não versionados; recriados pelo notebook).

## Estrutura do repositório

```text
.
├── notebooks/
│   ├── JeandersonMingoranca_ProjetoIntegrador.ipynb   ← NOTEBOOK PRINCIPAL
│   └── 01_… a 05_….ipynb                              ← notebooks de desenvolvimento
├── src/                          ← código reutilizável (funções Python)
│   ├── carregar_dados.py         · leitura dos CSVs e separação SIN/subsistemas
│   ├── qualidade_dados.py        · diagnóstico de qualidade dos dados
│   ├── indicadores.py            · cálculo da carga residual e indicadores diários
│   ├── visualizacao.py           · geração das figuras
│   └── modelagem.py              · regressão estatística e sensibilidade
├── dados/
│   └── brutos/                   ← CSVs originais do ONS (baixar do portal)
├── resultados/
│   ├── figuras/                  ← gráficos PNG
│   └── tabelas/                  ← resultados numéricos em CSV
├── docs/                         ← documento do Projeto Integrador
└── requirements.txt              ← dependências com versões fixadas
```

## Método em resumo

1. **Diagnóstico:** verificação de estrutura, grade horária completa, ausência de duplicidades, validação de que SIN = soma dos subsistemas, e determinação do sinal do intercâmbio (positivo = exportação).

2. **Preparação:** cálculo da carga residual (carga − eólica − solar); marcação de incidentes de registro e do apagão de 15/08/2023 como dias atípicos; uso de feriados nacionais como variável de controle; tratamento da quebra na série solar em 29/04/2023 com recorte de maio a dezembro.

3. **Indicadores diários:** índice de vale (carga 10–15h ÷ carga média), carga residual mínima, rampa do fim da tarde (18h–20h), maior rampa horária (16h–21h) e horas de carga residual baixa (abaixo do P10 de 2024).

4. **Exploração visual:** Figuras 1–4 e 6 mostrando perfis horários, heatmaps mês×hora, distribuições da rampa e padrões de intercâmbio.

5. **Teste estatístico:** regressão MQO `indicador ~ ano + mês + dia útil` por nível, com erros HAC para autocorrelação; seis cenários de sensibilidade; comparação 2025×2024; correlação de Spearman (Figura 5).

As decisões metodológicas estão documentadas no notebook principal e no documento do Projeto Integrador.

## Limitações

- A base **não mede MMGD nem cortes de geração (curtailment)**: os padrões são compatíveis com a expansão solar, mas não a demonstram.
- Desenho observacional: temperatura, hidrologia e atividade econômica não foram controladas.
- Três anos de dados permitem comparações entre anos, mas não a estimativa de tendências de longo prazo.

## Checklist do Guia do Projeto Integrador

| Item | Situação |
|---|---|
| Pergunta clara e respondível com a base fornecida | ✅ |
| Notebook executa do início ao fim sem ajustes manuais | ✅ |
| Decisões de limpeza registradas e dados brutos preservados | ✅ |
| Gráficos com título, rótulos e interpretação ligada à pergunta | ✅ |
| Conclusões sustentadas pelos resultados, sem extrapolar | ✅ |
| Limitações, riscos e próximos passos explicados | ✅ |
| Fontes e códigos externos identificados | ✅ |
| Arquivos organizados e nomeados corretamente | ✅ |

## Licença

- **Dados do ONS:** licença CC BY. Credite o ONS e informe as transformações.
- **Código e textos:** de autoria própria.
