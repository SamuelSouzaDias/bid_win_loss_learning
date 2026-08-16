# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: clients (versão aula)
# MAGIC
# MAGIC Mesmo código do `01_bronze_clients_pt-BR` — mesma tabela, mesmo resultado.
# MAGIC Esta versão é o `00_bronze_bids_aula` explicado de novo, aplicado a um
# MAGIC arquivo diferente. Se você já leu o `00`, a estrutura inteira — catálogo/
# MAGIC schema/tabela, `dtype=str`, `withColumn`, `mergeSchema`, a checagem de
# MAGIC schema com sets — é idêntica. Aqui eu só explico de novo o que muda:
# MAGIC os dois "cacoetes" desta fonte específica.
# MAGIC
# MAGIC ## O que muda em relação ao `00_bronze_bids`
# MAGIC
# MAGIC A exportação de `clients.xlsx` tem duas peculiaridades que valem notar
# MAGIC agora — mas que **não são corrigidas neste notebook**. Corrigir aqui
# MAGIC quebraria a regra de ouro da Bronze (cópia fiel da fonte, sem julgamento).
# MAGIC Quem vai tratar essas duas coisas é o `02_silver_bids`:
# MAGIC
# MAGIC 1. **`2999-12-31`** — um valor sentinela. Contratos sem data de término
# MAGIC    definida (ainda ativos, prazo indeterminado) recebem essa data fixa em
# MAGIC    vez de um campo vazio. É uma convenção comum em sistemas legados
# MAGIC    pra evitar ter uma coluna "nullable" numa ferramenta que não lida bem
# MAGIC    com nulo — mas pra quem consome o dado, `2999-12-31` parece uma data
# MAGIC    de verdade até alguém saber que não é.
# MAGIC 2. **A string literal `'null'`** — onde falta um valor, a exportação grava
# MAGIC    o texto `null` (quatro caracteres), não um null de verdade. Pro Spark,
# MAGIC    isso é só uma string comum; um filtro tipo `.isNull()` não pega.
# MAGIC
# MAGIC Ambas sobrevivem intocadas até aqui — são só *anotadas* nesta introdução
# MAGIC pra você já saber que existem quando chegar no Silver e ver o motivo de
# MAGIC uma função chamada `denull`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 1 — configuração
# MAGIC
# MAGIC Igual ao `00`: catálogo, schema e nome da tabela final, montados com uma
# MAGIC f-string. A única diferença é o nome da tabela (`clients` em vez de
# MAGIC `bids`) e o caminho do arquivo de origem.

# COMMAND ----------

CATALOG = "bronze"
SCHEMA = "bid"
VOLUME_PATH = "/Volumes/raw/bid/bids/clients.xlsx"
TABLE = f"{CATALOG}.{SCHEMA}.clients"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 2 — instalar dependência
# MAGIC
# MAGIC `%pip install openpyxl` roda antes do resto do código nesta mesma célula —
# MAGIC é a biblioteca que `pandas.read_excel` usa por baixo pra abrir `.xlsx`.
# MAGIC Já expliquei isso no `00`; aqui é a mesma linha, sem novidade.

# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 3 — ler o Excel, marcar metadados, gravar
# MAGIC
# MAGIC Diferente do `00`, aqui as três etapas (ler, marcar `_ingested_at`/
# MAGIC `_source_file`, gravar) estão numa célula só, em vez de separadas. É só
# MAGIC uma escolha de organização — o efeito é idêntico.
# MAGIC
# MAGIC ### O que cada linha faz (relembrando)
# MAGIC
# MAGIC ```python
# MAGIC pdf_raw = pd.read_excel(VOLUME_PATH, sheet_name="Clients", dtype=str)
# MAGIC df_raw = spark.createDataFrame(pdf_raw)
# MAGIC ```
# MAGIC Lê a aba `"Clients"` do Excel com pandas, tudo como texto (`dtype=str`),
# MAGIC e converte pra um DataFrame Spark. Mesma lógica do `00`.
# MAGIC
# MAGIC ```python
# MAGIC df_bronze = (
# MAGIC     df_raw
# MAGIC     .withColumn("_ingested_at", F.current_timestamp())
# MAGIC     .withColumn("_source_file", F.lit(VOLUME_PATH))
# MAGIC )
# MAGIC ```
# MAGIC As mesmas duas colunas técnicas de sempre: quando a linha chegou, e de
# MAGIC onde. `.withColumn(...)` encadeado duas vezes porque cada chamada devolve
# MAGIC um DataFrame novo (Spark é imutável) e a próxima `.withColumn` parte desse
# MAGIC resultado.
# MAGIC
# MAGIC ```python
# MAGIC (
# MAGIC     df_bronze.write
# MAGIC     .format("delta")
# MAGIC     .mode("overwrite")
# MAGIC     .option("mergeSchema", "true")
# MAGIC     .saveAsTable(TABLE)
# MAGIC )
# MAGIC ```
# MAGIC Grava como tabela Delta, substituindo a tabela inteira (`overwrite`),
# MAGIC aceitando mudança de schema (`mergeSchema`) pelo mesmo motivo do `00`: a
# MAGIC fonte é uma planilha mantida manualmente, e recusar mudança de coluna
# MAGIC pararia o pipeline por uma alteração cosmética.
# MAGIC
# MAGIC ```python
# MAGIC print(f"linhas: {spark.table(TABLE).count()}")
# MAGIC ```
# MAGIC Relê a tabela persistida em disco (não o DataFrame em memória) e confirma
# MAGIC quantas linhas foram gravadas de fato.

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

pdf_raw = pd.read_excel(VOLUME_PATH, sheet_name="Clients", dtype=str)
df_raw = spark.createDataFrame(pdf_raw)

df_bronze = (
    df_raw
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.lit(VOLUME_PATH))
)

(
    df_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(TABLE)
)

print(f"linhas: {spark.table(TABLE).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 4 — checar o schema esperado
# MAGIC
# MAGIC Mesma lógica de sets do `00`: monta o conjunto de colunas que a gente
# MAGIC espera encontrar (`EXPECTED`), compara contra o que realmente veio
# MAGIC (`actual`), e usa diferença de conjuntos (`-`) pra achar o que sumiu
# MAGIC (`missing`, que interrompe o notebook com erro) e o que apareceu de novo
# MAGIC (`unexpected`, que só vira um aviso).
# MAGIC
# MAGIC A única diferença real é a lista de colunas esperadas — aqui é o schema
# MAGIC de `clients`, não de `bids`.

# COMMAND ----------

EXPECTED = {
    "client_id", "contract_name", "status", "start_date", "end_date",
    "state", "city", "segment", "account_executive", "director",
    "manager", "coordinator",
}

actual = set(df_bronze.columns) - {"_ingested_at", "_source_file"}
missing, unexpected = EXPECTED - actual, actual - EXPECTED

if missing:
    raise ValueError(f"Colunas ausentes na fonte: {sorted(missing)}")
if unexpected:
    print(f"AVISO — novas colunas absorvidas, revisar Silver: {sorted(unexpected)}")

print("Checagem de schema aprovada.")
