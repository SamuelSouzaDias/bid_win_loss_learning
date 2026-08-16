# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold: bid performance (versão aula)
# MAGIC
# MAGIC Mesmo código do `03_gold_bid_performance_pt-BR` — mesmas tabelas, mesmo
# MAGIC resultado. Aqui a Silver termina e o Gold começa: agora estamos
# MAGIC calculando números de negócio de verdade (taxa de vitória, segmentação,
# MAGIC etc.), não mais limpando dado.
# MAGIC
# MAGIC Duas regras moldam esta camada inteira, e valem a pena entender antes de
# MAGIC ler qualquer célula:
# MAGIC
# MAGIC **Toda taxa é reportada duas vezes** — incluindo e excluindo propostas
# MAGIC carregadas em lote (lembra do `is_bulk_load` do notebook anterior?). Um
# MAGIC único número aqui seria enganoso.
# MAGIC
# MAGIC **A taxa de vitória é reportada por contagem e por valor.** As duas
# MAGIC diferem em alguns pontos percentuais, e só a ponderada por valor reflete
# MAGIC a realidade comercial (uma proposta de R$2 milhões pesa mais que uma de
# MAGIC R$20 mil).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 1 — o join point-in-time
# MAGIC
# MAGIC `clients_clean` é uma dimensão SCD Type 2 (você viu como ela é construída
# MAGIC no `02_silver_bids_aula`) — um mesmo cliente pode ter várias linhas, uma
# MAGIC por versão de contrato. A pergunta que este join responde é: **qual
# MAGIC versão do cliente estava vigente no momento em que cada proposta foi
# MAGIC criada?** Não "qual é a versão atual do cliente hoje".
# MAGIC
# MAGIC Por que isso importa: um join simples por `client_id` faria *fan-out*
# MAGIC (a proposta se multiplica) — toda proposta de um cliente que renovou
# MAGIC contrato casaria com **as duas** versões daquele cliente, contando a
# MAGIC mesma proposta duas vezes em qualquer soma ou contagem adiante.
# MAGIC
# MAGIC ### O que tem dentro de `point_in_time_join` (em `transforms.py`)
# MAGIC
# MAGIC ```python
# MAGIC def point_in_time_join(bids, clients_scd2, client_cols=(...)):
# MAGIC     b = bids.alias("b")
# MAGIC     c = clients_scd2.alias("c")
# MAGIC
# MAGIC     joined = b.join(
# MAGIC         c,
# MAGIC         (F.col("b.client_id") == F.col("c.client_id"))
# MAGIC         & (F.col("b.created_at") >= F.col("c.valid_from").cast("timestamp"))
# MAGIC         & (
# MAGIC             F.col("c.valid_to").isNull()
# MAGIC             | (F.col("b.created_at") < F.col("c.valid_to").cast("timestamp"))
# MAGIC         ),
# MAGIC         "left",
# MAGIC     )
# MAGIC
# MAGIC     select_cols = ["b.*"] + [F.col(f"c.{col}") for col in client_cols]
# MAGIC     return joined.select(*select_cols)
# MAGIC ```
# MAGIC Essa é a mesma condição de join que você já viu no `02` (na checagem de
# MAGIC cobertura) — client_id bate, **e** o `created_at` cai dentro da janela
# MAGIC `[valid_from, valid_to)` daquela versão. A diferença é o tipo de join:
# MAGIC aqui é `"left"` (mantém todas as propostas, mesmo as sem versão
# MAGIC correspondente — que ficam com os atributos de cliente em NULL), enquanto
# MAGIC no `02` era `"left_anti"` (só pra contar as sem correspondência).
# MAGIC
# MAGIC A última linha, `["b.*"] + [F.col(f"c.{col}") ...]`, monta a lista de
# MAGIC colunas do resultado: todas as colunas de `bids` (`"b.*"`), mais só as
# MAGIC colunas específicas do cliente que a função recebeu em `client_cols` (não
# MAGIC *todas* as colunas de `clients`) — isso evita ambiguidade, já que
# MAGIC `client_id` existe nos dois lados.
# MAGIC
# MAGIC ### A célula
# MAGIC
# MAGIC ```python
# MAGIC fact = point_in_time_join(bids, clients).filter(F.col("outcome").isNotNull())
# MAGIC ```
# MAGIC `fact` (nome clássico de modelagem dimensional pra uma tabela de fatos —
# MAGIC eventos de negócio) fica só com propostas **fechadas** (`outcome` não
# MAGIC nulo — ganhou ou perdeu, não mais em aberto).
# MAGIC
# MAGIC ```python
# MAGIC unmatched = fact.filter(F.col("client_sk").isNull()).count()
# MAGIC if unmatched:
# MAGIC     print(f"AVISO — ...")
# MAGIC ```
# MAGIC Como o join foi `"left"`, uma proposta sem versão de cliente correspondente
# MAGIC aparece com `client_sk` NULL em vez de ser descartada. Essa checagem
# MAGIC reconta esse número aqui — mesmo já checado no Silver — porque é a
# MAGIC premissa da qual todo o resto deste notebook depende, e é mais barato
# MAGIC reverificar do que assumir silenciosamente.

# COMMAND ----------

from pyspark.sql import functions as F
import sys
sys.path.append("/Workspace/Users/samuelsouzadias@outlook.com/bid-win-loss-analytics/en")
from transforms import point_in_time_join

spark.sql("CREATE SCHEMA IF NOT EXISTS gold.bid")

bids = spark.table("silver.bid.bids_clean")
clients = spark.table("silver.bid.clients_clean")

# Join point-in-time: os atributos do cliente como eram quando a proposta
# foi feita, não como são hoje. Lista explícita de colunas no select — com
# client_id presente nos dois lados da condição de join, `b.*` mais colunas
# `c.*` nomeadas evita um "client_id" ambíguo no resultado.
fact = point_in_time_join(bids, clients).filter(F.col("outcome").isNotNull())      # somente propostas fechadas
# Propostas que não casaram com nenhuma versão de cliente — o Silver já
# checa isso (ver a célula "cobertura point-in-time" do 02), reverificado
# aqui já que é a premissa da qual todo este join depende.
unmatched = fact.filter(F.col("client_sk").isNull()).count()
if unmatched:
    print(f"AVISO — {unmatched} propostas fechadas não casaram com nenhuma versão de cliente")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 2 — métricas principais, com `groupBy` + `agg`
# MAGIC
# MAGIC Aqui aparece o padrão mais comum de agregação em Spark: `groupBy(...)`
# MAGIC agrupa linhas por uma ou mais colunas, e `.agg(...)` calcula um ou mais
# MAGIC resumos pra cada grupo — diferente das window functions do notebook
# MAGIC anterior, aqui as linhas *colapsam*: um grupo de 500 propostas vira **uma**
# MAGIC linha de resultado.
# MAGIC
# MAGIC ```python
# MAGIC def performance(df, *dims):
# MAGIC     return (
# MAGIC         df.groupBy(*dims)
# MAGIC           .agg(
# MAGIC               F.count("*").alias("bids_closed"),
# MAGIC               F.sum("outcome").alias("bids_won"),
# MAGIC               F.round(F.avg("outcome") * 100, 1).alias("win_rate_by_count"),
# MAGIC               F.round(
# MAGIC                   F.sum(F.when(F.col("outcome") == 1, F.col("contract_value_brl")).otherwise(0))
# MAGIC                   / F.sum("contract_value_brl") * 100, 1
# MAGIC               ).alias("win_rate_by_value"),
# MAGIC               F.round(F.sum("contract_value_brl"), 2).alias("value_bid_brl"),
# MAGIC           )
# MAGIC     )
# MAGIC ```
# MAGIC - `*dims` — o `*` aqui é "argumentos variádicos" do Python: a função
# MAGIC   aceita quantas dimensões você quiser passar (nenhuma, uma, várias), e
# MAGIC   elas chegam como uma tupla dentro da função. Isso permite reusar a
# MAGIC   mesma função `performance` tanto pra um resumo geral (zero dimensões)
# MAGIC   quanto por segmento, por estado, etc.
# MAGIC - `F.count("*")` conta linhas no grupo. `F.sum("outcome")` soma a coluna
# MAGIC   `outcome` (que é 1 pra ganhou, 0 pra perdeu) — ou seja, conta quantas
# MAGIC   foram ganhas. `F.avg("outcome") * 100` é a média de 1s e 0s vezes 100,
# MAGIC   que é exatamente a taxa de vitória em percentual.
# MAGIC - `win_rate_by_value` é a parte mais densa: `F.sum(F.when(outcome==1,
# MAGIC   valor).otherwise(0))` soma o valor só das propostas ganhas (zerando as
# MAGIC   perdidas dentro da soma), e divide pelo valor total do grupo — a taxa de
# MAGIC   vitória *ponderada por dinheiro*, não por quantidade de propostas.
# MAGIC - `.alias("nome")` dá um nome à coluna resultante — sem isso, o Spark
# MAGIC   geraria um nome automático feio, tipo `round((avg(outcome) * 100), 1)`.
# MAGIC
# MAGIC ```python
# MAGIC overall = performance(fact, F.lit(True).alias("_all")).drop("_all")
# MAGIC ```
# MAGIC Um truque pra reusar `performance` também pra um total geral sem
# MAGIC dimensão real: `F.lit(True)` cria uma coluna onde todo mundo tem o mesmo
# MAGIC valor, então agrupar por ela produz um grupo só (a empresa inteira). A
# MAGIC coluna auxiliar `_all` é descartada (`.drop`) depois, já que ela não
# MAGIC carrega informação nenhuma.

# COMMAND ----------

def performance(df, *dims):
    return (
        df.groupBy(*dims)
          .agg(
              F.count("*").alias("bids_closed"),
              F.sum("outcome").alias("bids_won"),
              F.round(F.avg("outcome") * 100, 1).alias("win_rate_by_count"),
              F.round(
                  F.sum(F.when(F.col("outcome") == 1, F.col("contract_value_brl")).otherwise(0))
                  / F.sum("contract_value_brl") * 100, 1
              ).alias("win_rate_by_value"),
              F.round(F.sum("contract_value_brl"), 2).alias("value_bid_brl"),
          )
    )


overall = performance(fact, F.lit(True).alias("_all")).drop("_all")
by_channel = performance(fact, "is_bulk_load")

overall.write.format("delta").mode("overwrite").saveAsTable("gold.bid.performance_overall")
by_channel.write.format("delta").mode("overwrite").saveAsTable("gold.bid.performance_by_channel")

display(by_channel)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 3 — segmentação, com o confound isolado
# MAGIC
# MAGIC Cada dimensão (segmento, estado, executivo) é reportada duas vezes: com
# MAGIC todas as propostas fechadas, e de novo só com as orgânicas. O gap entre
# MAGIC as duas colunas mostra o tamanho do artefato de migração naquela fatia
# MAGIC específica.
# MAGIC
# MAGIC ```python
# MAGIC organic = fact.filter(~F.col("is_bulk_load"))
# MAGIC ```
# MAGIC `~` é o operador de negação em expressões de coluna Spark (equivalente ao
# MAGIC `not` do Python) — aqui, filtra pra manter só onde `is_bulk_load` é falso.
# MAGIC
# MAGIC ```python
# MAGIC for dim in ["segment", "state", "account_executive"]:
# MAGIC     combined = (
# MAGIC         performance(fact, dim)
# MAGIC         .select(dim, "bids_closed", F.col("win_rate_by_count").alias("wr_all"))
# MAGIC         .join(
# MAGIC             performance(organic, dim)
# MAGIC               .select(dim, F.col("bids_closed").alias("bids_organic"), F.col("win_rate_by_count").alias("wr_organic")),
# MAGIC             dim, "left",
# MAGIC         )
# MAGIC         .withColumn("artefact_gap", F.round(F.col("wr_organic") - F.col("wr_all"), 1))
# MAGIC         .orderBy(F.col("bids_closed").desc())
# MAGIC     )
# MAGIC     combined.write.format("delta").mode("overwrite").saveAsTable(f"gold.bid.performance_by_{dim}")
# MAGIC     display(combined)
# MAGIC ```
# MAGIC Um loop Python comum, iterando sobre uma lista de nomes de dimensão — o
# MAGIC mesmo bloco de código roda três vezes, uma pra cada dimensão, gravando
# MAGIC três tabelas diferentes (`gold.bid.performance_by_segment`,
# MAGIC `..._by_state`, `..._by_account_executive`) usando uma f-string no nome
# MAGIC da tabela (`f"gold.bid.performance_by_{dim}"`).
# MAGIC
# MAGIC Dentro do loop: chama `performance(...)` duas vezes (uma com `fact`, todas
# MAGIC as propostas; outra com `organic`, só as não-lote), renomeia as colunas de
# MAGIC cada resultado pra não colidir (`wr_all` vs `wr_organic`), junta os dois
# MAGIC (`.join(..., dim, "left")` — junta pela própria dimensão sendo analisada,
# MAGIC ex.: `"segment"`), e calcula a diferença entre as duas taxas
# MAGIC (`artefact_gap`) — o tamanho do efeito da carga em lote naquele corte.

# COMMAND ----------

organic = fact.filter(~F.col("is_bulk_load"))

for dim in ["segment", "state", "account_executive"]:
    combined = (
        performance(fact, dim)
        .select(dim, "bids_closed", F.col("win_rate_by_count").alias("wr_all"))
        .join(
            performance(organic, dim)
              .select(dim,
                      F.col("bids_closed").alias("bids_organic"),
                      F.col("win_rate_by_count").alias("wr_organic")),
            dim, "left",
        )
        .withColumn("artefact_gap", F.round(F.col("wr_organic") - F.col("wr_all"), 1))
        .orderBy(F.col("bids_closed").desc())
    )
    combined.write.format("delta").mode("overwrite").saveAsTable(f"gold.bid.performance_by_{dim}")
    display(combined)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 4 — efeito do valor do contrato, com `approxQuantile`
# MAGIC
# MAGIC A pergunta: a taxa de vitória cai conforme o valor do contrato sobe? Pra
# MAGIC responder, dividimos as propostas em quatro faixas de valor (quartis) de
# MAGIC tamanho igual e comparamos a taxa de vitória entre elas.
# MAGIC
# MAGIC ### Por que `approxQuantile` em vez de `ntile()`
# MAGIC
# MAGIC `ntile(4)` é a função Spark mais óbvia pra dividir dado em quartis — mas
# MAGIC ela é uma window function que precisa **ordenar todo o DataFrame numa
# MAGIC janela sem partição**, ou seja, forçar tudo pra um único nó de
# MAGIC processamento pra conseguir dar ranks exatos. Com 1.600 linhas isso é
# MAGIC inofensivo, mas é exatamente o padrão por trás do aviso
# MAGIC `WindowExpression: No Partition Defined` que o Spark lança, e para de
# MAGIC escalar bem muito antes do resto deste pipeline.
# MAGIC
# MAGIC `approxQuantile` resolve isso com um algoritmo **distribuído e
# MAGIC aproximado** (baseado em sketch estatístico) que encontra os pontos de
# MAGIC corte sem precisar centralizar o dado. Depois, classificar cada linha em
# MAGIC uma faixa é só um `when/otherwise` simples, sem shuffle nenhum.
# MAGIC
# MAGIC ```python
# MAGIC q1, q2, q3 = fact.approxQuantile("contract_value_brl", [0.25, 0.5, 0.75], 0.01)
# MAGIC ```
# MAGIC Pede os pontos de corte pros percentis 25%, 50% e 75% da coluna
# MAGIC `contract_value_brl`. `relativeError=0.01` controla a precisão da
# MAGIC aproximação — 1% de erro é sobra pra bandas de relatório, e o algoritmo
# MAGIC fica barato independente do tamanho da tabela. `approxQuantile` devolve
# MAGIC uma lista com três valores, desempacotada direto em `q1, q2, q3`.
# MAGIC
# MAGIC ```python
# MAGIC value_bands = fact.withColumn(
# MAGIC     "value_quartile",
# MAGIC     F.when(F.col("contract_value_brl") <= q1, 1)
# MAGIC      .when(F.col("contract_value_brl") <= q2, 2)
# MAGIC      .when(F.col("contract_value_brl") <= q3, 3)
# MAGIC      .otherwise(4),
# MAGIC )
# MAGIC ```
# MAGIC O mesmo padrão de if/elif/else de sempre, agora classificando cada
# MAGIC proposta num quartil de 1 (mais barata) a 4 (mais cara) comparando contra
# MAGIC os três cortes calculados acima. Depois disso, é só reusar a mesma função
# MAGIC `performance(...)` de novo, agrupando por `"value_quartile"`.

# COMMAND ----------

# Cortes de quartil aproximados (relativeError=0.01 — precisão de sobra
# para bandas de relatório, e barato independente do tamanho da tabela).
q1, q2, q3 = fact.approxQuantile("contract_value_brl", [0.25, 0.5, 0.75], 0.01)

value_bands = fact.withColumn(
    "value_quartile",
    F.when(F.col("contract_value_brl") <= q1, 1)
     .when(F.col("contract_value_brl") <= q2, 2)
     .when(F.col("contract_value_brl") <= q3, 3)
     .otherwise(4),
)

by_value = (
    performance(value_bands, "value_quartile")
    .orderBy("value_quartile")
)

by_value.write.format("delta").mode("overwrite").saveAsTable("gold.bid.performance_by_value_band")
display(by_value)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 5 — motivos de perda, e o quanto pouco eles cobrem do quadro
# MAGIC
# MAGIC O número de cobertura (quantas perdas *têm* motivo registrado) é o ponto
# MAGIC principal desta tabela — uma distribuição de motivos construída sobre uma
# MAGIC fração pequena das perdas é um sinal, não uma estimativa confiável do
# MAGIC todo, e os dois números precisam ser publicados juntos, nunca só a
# MAGIC distribuição sozinha.
# MAGIC
# MAGIC ```python
# MAGIC losses = bids.filter(F.col("outcome") == 0)
# MAGIC
# MAGIC coverage = losses.agg(
# MAGIC     F.count("*").alias("losses_total"),
# MAGIC     F.sum(F.col("has_loss_reason").cast("int")).alias("losses_with_reason"),
# MAGIC     F.round(F.avg(F.col("has_loss_reason").cast("int")) * 100, 1).alias("coverage_pct"),
# MAGIC     F.sum(F.col("competitor_is_placeholder").cast("int")).alias("placeholder_attributed"),
# MAGIC )
# MAGIC ```
# MAGIC `.agg(...)` sem `.groupBy(...)` antes trata o DataFrame inteiro como um
# MAGIC único grupo — devolve uma linha só, com os quatro totais gerais.
# MAGIC `has_loss_reason` e `competitor_is_placeholder` foram criadas lá no
# MAGIC `02_silver_bids`, dentro de `flag_placeholder_competitor` — casting pra
# MAGIC `int` de novo pra poder somar booleanos.
# MAGIC
# MAGIC ```python
# MAGIC reasons_raw = (
# MAGIC     losses.filter(F.col("has_loss_reason"))
# MAGIC           .groupBy("loss_reason")
# MAGIC           .agg(F.count("*").alias("losses"))
# MAGIC )
# MAGIC
# MAGIC total_with_reason = reasons_raw.agg(F.sum("losses")).collect()[0][0]
# MAGIC ```
# MAGIC Agora sim um `groupBy` de verdade: conta quantas perdas existem por
# MAGIC motivo, só entre as que *têm* motivo registrado. `.collect()` traz o
# MAGIC resultado do Spark pra dentro do Python como uma lista comum —
# MAGIC `[0][0]` pega a primeira linha, primeira coluna, ou seja, o número puro.
# MAGIC Isso é deliberadamente diferente de uma window function `Window.partitionBy()`
# MAGIC vazia pra somar um total: com poucos motivos distintos, é mais barato e
# MAGIC mais claro calcular o total uma vez com `.collect()` e reusar como um
# MAGIC número Python simples do que forçar o Spark a fazer isso via window.
# MAGIC
# MAGIC ```python
# MAGIC reasons = (
# MAGIC     reasons_raw
# MAGIC     .withColumn("share_pct", F.round(F.col("losses") / F.lit(total_with_reason) * 100, 1))
# MAGIC     .orderBy(F.col("losses").desc())
# MAGIC )
# MAGIC ```
# MAGIC Cada motivo vira um percentual do total de perdas *com motivo* (não do
# MAGIC total geral de perdas). `F.lit(total_with_reason)` de novo — como visto no
# MAGIC `02`, sempre que um número Python simples precisa entrar numa expressão de
# MAGIC coluna Spark, ele precisa ser envolvido em `F.lit(...)`.

# COMMAND ----------

losses = bids.filter(F.col("outcome") == 0)

coverage = losses.agg(
    F.count("*").alias("losses_total"),
    F.sum(F.col("has_loss_reason").cast("int")).alias("losses_with_reason"),
    F.round(F.avg(F.col("has_loss_reason").cast("int")) * 100, 1).alias("coverage_pct"),
    F.sum(F.col("competitor_is_placeholder").cast("int")).alias("placeholder_attributed"),
)

reasons_raw = (
    losses.filter(F.col("has_loss_reason"))
          .groupBy("loss_reason")
          .agg(F.count("*").alias("losses"))
)

# Um número, calculado uma vez — mais barato e mais claro que uma soma via window.
total_with_reason = reasons_raw.agg(F.sum("losses")).collect()[0][0]

reasons = (
    reasons_raw
    .withColumn("share_pct", F.round(F.col("losses") / F.lit(total_with_reason) * 100, 1))
    .orderBy(F.col("losses").desc())
)

coverage.write.format("delta").mode("overwrite").saveAsTable("gold.bid.loss_reason_coverage")
reasons.write.format("delta").mode("overwrite").saveAsTable("gold.bid.loss_reasons")

display(coverage)
display(reasons)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 6 — métricas de qualidade de dados, tudo num só lugar
# MAGIC
# MAGIC Todo número desta tabela já existe em algum lugar deste notebook — a
# MAGIC fração de carga em lote está implícita em `performance_by_channel`, a
# MAGIC cobertura de motivo de perda está na tabela acima, a taxa de erro do join
# MAGIC point-in-time é um `print()` de aviso lá em cima. Nenhum deles mora num
# MAGIC lugar onde alguém consiga checar sem ler o notebook inteiro — esta seção
# MAGIC reúne tudo numa tabela só, uma linha por checagem.
# MAGIC
# MAGIC ```python
# MAGIC coverage_row = coverage.collect()[0]
# MAGIC ```
# MAGIC `coverage` (calculado no passo anterior) tem uma linha só; `.collect()[0]`
# MAGIC pega essa linha como um objeto Python (tipo `Row`) de onde dá pra ler cada
# MAGIC campo por nome, ex.: `coverage_row["coverage_pct"]`.
# MAGIC
# MAGIC ```python
# MAGIC all_matched = point_in_time_join(bids, clients)
# MAGIC unmatched_all = all_matched.filter(F.col("client_sk").isNull()).count()
# MAGIC pct_unmatched_all = round(unmatched_all / bids.count() * 100, 1)
# MAGIC ```
# MAGIC Este é o único cálculo novo desta célula: o `unmatched` calculado lá no
# MAGIC Passo 1 só cobria propostas **fechadas** (`fact`). Aqui refazemos o mesmo
# MAGIC join point-in-time sobre `bids` inteiro (abertas + fechadas) pra ter a
# MAGIC taxa de descasamento real sobre a base toda. Repare que `round(...)` aqui
# MAGIC é o `round` **built-in do Python**, não `F.round` — porque `unmatched_all /
# MAGIC bids.count()` já são dois números Python simples (resultado de
# MAGIC `.count()`), não uma expressão de coluna Spark.
# MAGIC
# MAGIC ```python
# MAGIC dq_metrics = spark.createDataFrame(
# MAGIC     [("total_bids", 1600.0, "..."), ("pct_bulk_loaded", 57.5, "..."), ...],
# MAGIC     ["metric", "value", "description"],
# MAGIC )
# MAGIC ```
# MAGIC Diferente do resto do notebook, aqui o DataFrame não vem de uma
# MAGIC transformação — é construído **direto de uma lista Python de tuplas**.
# MAGIC `spark.createDataFrame(dados, colunas)` aceita isso: uma lista onde cada
# MAGIC item é uma linha (uma tupla com um valor por coluna), e uma segunda lista
# MAGIC com os nomes das colunas. É a mesma função usada lá no `00_bronze_bids`
# MAGIC (`spark.createDataFrame(pdf_raw)`), só que ali convertendo um DataFrame
# MAGIC pandas, e aqui construindo os dados na mão.
# MAGIC
# MAGIC O resultado é um formato "longo" (uma linha por métrica, não uma coluna
# MAGIC por métrica) — mais fácil de consultar depois (`WHERE metric =
# MAGIC 'pct_bulk_loaded'`) e de adicionar uma métrica nova sem mudar o schema da
# MAGIC tabela.

# COMMAND ----------

coverage_row = coverage.collect()[0]

# Recalculado sobre todas as propostas, não só as fechadas que o `fact`
# cobre acima — esta é a única métrica aqui que precisa de uma passada nova.
all_matched = point_in_time_join(bids, clients)
unmatched_all = all_matched.filter(F.col("client_sk").isNull()).count()
pct_unmatched_all = round(unmatched_all / bids.count() * 100, 1)

dq_metrics = spark.createDataFrame(
    [
        (
            "total_bids",
            float(bids.count()),
            "Linhas em silver.bid.bids_clean",
        ),
        (
            "pct_bulk_loaded",
            round(bids.filter("is_bulk_load").count() / bids.count() * 100, 1),
            "Fração de propostas com created_at compartilhado por 10+ outras",
        ),
        (
            "pct_closed",
            round(bids.filter(F.col("outcome").isNotNull()).count() / bids.count() * 100, 1),
            "Fração de propostas com outcome registrado (ganhou/perdeu, vs ainda aberta)",
        ),
        (
            "loss_reason_coverage_pct",
            float(coverage_row["coverage_pct"]),
            "Fração de perdas com loss_reason não-nulo",
        ),
        (
            "pct_losses_placeholder_competitor",
            round(coverage_row["placeholder_attributed"] / coverage_row["losses_total"] * 100, 1),
            "Fração de perdas atribuídas ao valor padrão 'Competitor 1'",
        ),
        (
            "pct_bids_unmatched_point_in_time",
            float(pct_unmatched_all),
            "Fração de todas as propostas (abertas + fechadas) cujo created_at cai fora de toda janela de contrato de cliente conhecida",
        ),
    ],
    ["metric", "value", "description"],
)

dq_metrics.write.format("delta").mode("overwrite").saveAsTable("gold.bid.data_quality_metrics")
display(dq_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 7 — pipeline em aberto
# MAGIC
# MAGIC Propostas com `outcome` NULL — ainda não fechadas. Reportadas
# MAGIC separadamente, nunca somadas silenciosamente na coluna de perdas, o que
# MAGIC subestimaria a taxa de vitória em cerca de um terço.
# MAGIC
# MAGIC Mesmo join point-in-time de sempre — um join simples por `client_id`
# MAGIC faria fan-out de toda proposta aberta de um cliente renovado, contando em
# MAGIC dobro aqui também. A única linha nova é o `.groupBy("segment").agg(...)`
# MAGIC no final, resumindo quantas propostas e quanto valor está aberto por
# MAGIC segmento — o mesmo padrão `groupBy`/`agg` do Passo 2, só que escrito numa
# MAGIC linha só em vez de dentro de uma função nomeada.

# COMMAND ----------

open_bids = bids.filter(F.col("outcome").isNull())
pipeline = point_in_time_join(open_bids, clients).groupBy("segment").agg(F.count("*").alias("bids_open"), F.round(F.sum("contract_value_brl"), 2).alias("value_open_brl")).orderBy(F.col("value_open_brl").desc())

pipeline.write.format("delta").mode("overwrite").saveAsTable("gold.bid.open_pipeline")
display(pipeline)
