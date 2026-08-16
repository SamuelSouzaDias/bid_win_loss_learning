# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: tipado, limpo, deduplicado (versão aula)
# MAGIC
# MAGIC Mesmo código do `02_silver_bids_pt-BR` — mesmas tabelas, mesmo resultado.
# MAGIC Este é o notebook mais denso do pipeline em termos de conceito novo, então
# MAGIC a explicação também é a mais longa. Quatro coisas acontecem aqui:
# MAGIC
# MAGIC 1. Strings viram tipos de verdade; a string literal `'null'` vira NULL de
# MAGIC    verdade
# MAGIC 2. Datas sentinela (`2999-12-31`) são resolvidas
# MAGIC 3. `clients` é modelado como uma dimensão **SCD Type 2** (conceito novo —
# MAGIC    explico com calma mais abaixo)
# MAGIC 4. `is_bulk_load` é derivado usando uma **window function** (outro
# MAGIC    conceito novo)
# MAGIC
# MAGIC ## Uma peça importante antes de começar: `transforms.py`
# MAGIC
# MAGIC Repare que este notebook não escreve a lógica de transformação direto
# MAGIC aqui — ele **importa** funções prontas de um arquivo separado,
# MAGIC `transforms.py` (`from transforms import denull, add_is_bulk_load, ...`).
# MAGIC Por quê? Porque essas mesmas funções também são usadas pelo
# MAGIC `03_gold_bid_performance` e pelo `04_analysis` — escrevê-las uma vez num
# MAGIC lugar só evita ter três cópias ligeiramente diferentes da mesma lógica
# MAGIC espalhadas pelo projeto (e evita o bug clássico de corrigir uma cópia e
# MAGIC esquecer as outras). Cada função em `transforms.py` recebe um DataFrame
# MAGIC Spark e devolve outro — sem ler tabela nenhuma, sem gravar nada — o que
# MAGIC também é o que permite testar elas isoladamente, sem precisar de uma
# MAGIC conexão real com o Databricks.
# MAGIC
# MAGIC Ao longo deste notebook, cada vez que uma dessas funções for chamada, eu
# MAGIC explico o que tem *dentro* dela — não é uma caixa preta.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 1 — imports e leitura das tabelas Bronze
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.sql import functions as F, Window
# MAGIC ```
# MAGIC Além de `F` (que você já viu nos notebooks `00` e `01`), agora também
# MAGIC importamos `Window`. Isso é usado pra **window functions** — calcular algo
# MAGIC que depende de *outras linhas relacionadas*, não só da linha atual. Vou
# MAGIC explicar isso com um exemplo concreto na seção de `is_bulk_load` mais
# MAGIC abaixo.
# MAGIC
# MAGIC ```python
# MAGIC sys.path.append("/Workspace/.../en")
# MAGIC from transforms import denull, add_is_bulk_load, flag_placeholder_competitor, build_client_scd2
# MAGIC ```
# MAGIC `sys.path.append(...)` diz ao Python "procure módulos importáveis também
# MAGIC nesta pasta" — é isso que permite o `from transforms import ...` logo
# MAGIC depois encontrar o arquivo `transforms.py`, que vive numa pasta diferente
# MAGIC deste notebook.
# MAGIC
# MAGIC ```python
# MAGIC bronze_bids = spark.table("bronze.bid.bids")
# MAGIC bronze_clients = spark.table("bronze.bid.clients")
# MAGIC ```
# MAGIC `spark.table("catalogo.schema.tabela")` lê uma tabela já gravada no Unity
# MAGIC Catalog e devolve ela como um DataFrame Spark — é o "ler de volta" do que
# MAGIC os notebooks `00` e `01` gravaram.

# COMMAND ----------

from pyspark.sql import functions as F, Window
import sys
sys.path.append("/Workspace/Users/samuelsouzadias@outlook.com/bid-win-loss-analytics/en")
from transforms import denull, add_is_bulk_load, flag_placeholder_competitor, build_client_scd2

spark.sql("CREATE SCHEMA IF NOT EXISTS silver.bid")

bronze_bids = spark.table("bronze.bid.bids")
bronze_clients = spark.table("bronze.bid.clients")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 2 — tratamento de nulos com `denull`
# MAGIC
# MAGIC A exportação grava a string de quatro caracteres `'null'`, e `'-'` para
# MAGIC uma data de fechamento ausente. Nenhuma das duas é um NULL de verdade
# MAGIC pro Spark — pro Spark, `'null'` é só um texto qualquer, igual seria
# MAGIC `'abacate'`. Se você não trocar isso por um NULL real, qualquer filtro
# MAGIC do tipo `.isNull()` mais adiante simplesmente não vai pegar essas linhas,
# MAGIC e o dado ruim sobrevive silenciosamente pipeline afora.
# MAGIC
# MAGIC ### O que tem dentro de `denull` (em `transforms.py`)
# MAGIC
# MAGIC ```python
# MAGIC def denull(df, placeholders=("null", "-", "")):
# MAGIC     for c, t in df.dtypes:
# MAGIC         if t == "string":
# MAGIC             df = df.withColumn(
# MAGIC                 c,
# MAGIC                 F.when(F.trim(F.col(c)).isin(list(placeholders)), None).otherwise(F.col(c)),
# MAGIC             )
# MAGIC     return df
# MAGIC ```
# MAGIC - `df.dtypes` devolve uma lista de pares `(nome_da_coluna, tipo)` — por
# MAGIC   exemplo `[("bid_id", "string"), ("outcome", "string"), ...]`. O `for c, t
# MAGIC   in df.dtypes:` percorre essa lista, desempacotando cada par em `c`
# MAGIC   (coluna) e `t` (tipo).
# MAGIC - `if t == "string":` — só mexe em colunas de texto. Não faz sentido
# MAGIC   procurar a string `'null'` dentro de uma coluna que já é número.
# MAGIC - `F.when(condição, valor_se_verdadeiro).otherwise(valor_se_falso)` é o
# MAGIC   "if/else" do Spark, usado dentro de uma expressão de coluna — você vai
# MAGIC   ver esse padrão o projeto inteiro. Aqui: **se** o valor da coluna
# MAGIC   (depois de `F.trim`, que remove espaços em branco nas pontas) estiver
# MAGIC   na lista de placeholders, vira `None` (o `None` do Python, que o Spark
# MAGIC   entende como NULL de verdade) — **senão**, mantém o valor original.
# MAGIC - `F.col(c).isin(list(placeholders))` — `isin` testa se o valor da coluna
# MAGIC   está dentro de uma lista de opções. Equivalente a fazer várias
# MAGIC   comparações com `OR` encadeadas, só que mais legível.
# MAGIC - O loop reatribui `df = df.withColumn(...)` a cada volta — cada iteração
# MAGIC   parte do DataFrame que a iteração anterior devolveu, então no final
# MAGIC   `df` tem todas as colunas de texto tratadas.

# COMMAND ----------

bids = denull(bronze_bids)
clients = denull(bronze_clients)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 3 — tipagem
# MAGIC
# MAGIC Até aqui, toda coluna ainda é texto (lembra do `dtype=str` no Bronze?).
# MAGIC Agora é a hora de dar o tipo de verdade pra cada uma.
# MAGIC
# MAGIC `outcome` é deixado **nullable** de propósito: NULL significa que a
# MAGIC proposta ainda está aberta — um estado bem diferente de "perdida" — e não
# MAGIC pode virar `0` (o que faria a proposta parecer perdida). Cerca de 28% da
# MAGIC tabela está aberta.
# MAGIC
# MAGIC `to_timestamp` recebe um formato de data explícito (`TS_FORMAT`) em vez de
# MAGIC deixar o Spark adivinhar o formato sozinho. Por quê? O parser automático
# MAGIC do Spark, quando não reconhece o formato, devolve `NULL` silenciosamente
# MAGIC em vez de dar erro — exatamente o tipo de perda de dado silenciosa que
# MAGIC este pipeline inteiro tenta evitar. Passando o formato explícito, você
# MAGIC sabe exatamente o que está sendo esperado, e o `assert` logo depois pega
# MAGIC qualquer divergência.
# MAGIC
# MAGIC ### O código
# MAGIC
# MAGIC ```python
# MAGIC TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"
# MAGIC ```
# MAGIC Um padrão de data no formato do Spark (parecido, mas não idêntico, ao
# MAGIC formato de data do Python): ano-mês-dia, um `T` literal (entre aspas
# MAGIC simples porque é texto fixo, não parte do padrão), hora:minuto:segundo,
# MAGIC milissegundos, e o fuso horário (`XXX`).
# MAGIC
# MAGIC ```python
# MAGIC bids_typed = (
# MAGIC     bids
# MAGIC     .withColumn("bid_id", F.col("bid_id").cast("long"))
# MAGIC     .withColumn("client_id", F.col("client_id").cast("long"))
# MAGIC     .withColumn("created_at", F.to_timestamp("created_at", TS_FORMAT))
# MAGIC     ...
# MAGIC     .drop("created_at_str", "closed_at_str")
# MAGIC )
# MAGIC ```
# MAGIC - `.cast("tipo")` converte uma coluna pro tipo pedido (`"long"` = inteiro
# MAGIC   grande, `"int"` = inteiro, `"double"` = decimal).
# MAGIC - `F.to_timestamp(coluna, formato)` faz o parse do texto pra um timestamp
# MAGIC   de verdade, usando o formato que você passou.
# MAGIC - `.drop("col1", "col2")` remove colunas. Aqui removemos as versões em
# MAGIC   texto das datas (`created_at_str`, `closed_at_str`) — elas eram só uma
# MAGIC   cópia redundante mantida no Bronze; agora que temos as versões
# MAGIC   tipadas, não precisamos mais delas.
# MAGIC
# MAGIC ```python
# MAGIC bad_created = (
# MAGIC     bids.filter(F.col("created_at").isNotNull())
# MAGIC     .join(bids_typed.filter(F.col("created_at").isNull()), "bid_id", "inner")
# MAGIC     .count()
# MAGIC )
# MAGIC assert bad_created == 0, (...)
# MAGIC ```
# MAGIC Esta é uma **guarda de qualidade**: pega as linhas onde `created_at` *não*
# MAGIC era nulo antes do cast (em `bids`, ainda texto) e cruza (`.join(...,
# MAGIC "inner")`) com as linhas onde `created_at` *virou* nulo depois do cast
# MAGIC (em `bids_typed`). Se esse cruzamento achar alguma linha, significa que o
# MAGIC `to_timestamp` falhou silenciosamente em algo que tinha valor — ou seja, o
# MAGIC formato da fonte mudou e `TS_FORMAT` não bate mais. `assert condição,
# MAGIC mensagem` interrompe o notebook com erro (e mostra a mensagem) se a
# MAGIC condição for falsa. É o equivalente, em código, de um alarme: "pare tudo,
# MAGIC algo que eu esperava que fosse verdade não é".

# COMMAND ----------

TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"

bids_typed = (
    bids
    .withColumn("bid_id", F.col("bid_id").cast("long"))
    .withColumn("client_id", F.col("client_id").cast("long"))
    .withColumn("created_at", F.to_timestamp("created_at", TS_FORMAT))
    .withColumn("bid_date", F.to_timestamp("bid_date", TS_FORMAT))
    .withColumn("closed_at", F.to_timestamp("closed_at", TS_FORMAT))
    .withColumn("outcome", F.col("outcome").cast("int"))          # NULL = aberta
    .withColumn("is_confirmed_date", F.col("is_confirmed_date").cast("int"))
    .withColumn("contract_value_brl", F.col("contract_value_brl").cast("double"))
    .drop("created_at_str", "closed_at_str")
)

# Guarda: um created_at que era não-NULL antes do cast mas virou NULL depois
# significa que a format string parou de casar com a fonte — falhar
# ruidosamente, não silenciosamente.
bad_created = (
    bids.filter(F.col("created_at").isNotNull())
    .join(bids_typed.filter(F.col("created_at").isNull()), "bid_id", "inner")
    .count()
)
assert bad_created == 0, (
    f"{bad_created} valores de created_at falharam ao parsear com o formato {TS_FORMAT} "
    "— verifique se o formato de data da fonte mudou."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 4 — derivando `is_bulk_load` com uma window function
# MAGIC
# MAGIC ### O problema de negócio primeiro
# MAGIC
# MAGIC Uma grande fração das propostas foi importada em massa durante uma
# MAGIC migração de sistema, não registrada uma a uma conforme aconteciam. Elas
# MAGIC compartilham um `created_at` idêntico até o segundo — dezenas ou centenas
# MAGIC de propostas com o *mesmo exato timestamp*, o que uma pessoa cadastrando
# MAGIC manualmente nunca produziria. Essas propostas em lote se comportam
# MAGIC completamente diferente das orgânicas (taxa de vitória bem mais baixa,
# MAGIC concentrada em carteiras específicas), e sem marcá-las, elas contaminam
# MAGIC qualquer métrica segmentada — o executivo dono da carteira migrada parece
# MAGIC o pior desempenho da empresa, só por causa de como os registros dele
# MAGIC foram carregados.
# MAGIC
# MAGIC ### O que é uma window function
# MAGIC
# MAGIC Uma agregação normal (`groupBy(...).agg(...)`) pega várias linhas e
# MAGIC devolve **uma linha por grupo** — você perde o detalhe de linha. Uma
# MAGIC *window function* faz um cálculo que olha pra outras linhas relacionadas,
# MAGIC mas devolve **uma linha pra cada linha original** — ela "enriquece" cada
# MAGIC linha com informação do grupo dela, sem colapsar nada.
# MAGIC
# MAGIC Aqui, a pergunta é: "quantas outras propostas compartilham o mesmo
# MAGIC `created_at` que esta linha?" — isso é por definição uma pergunta que
# MAGIC olha pras outras linhas do mesmo grupo (mesmo `created_at`), mas a
# MAGIC resposta precisa ficar em cada linha individual, não resumida.
# MAGIC
# MAGIC ### O que tem dentro de `add_is_bulk_load` (em `transforms.py`)
# MAGIC
# MAGIC ```python
# MAGIC def add_is_bulk_load(df, threshold=10):
# MAGIC     batch = Window.partitionBy("created_at")
# MAGIC     return (
# MAGIC         df.withColumn("_batch_size", F.count("*").over(batch))
# MAGIC         .withColumn("is_bulk_load", (F.col("_batch_size") >= threshold).cast("boolean"))
# MAGIC         .drop("_batch_size")
# MAGIC     )
# MAGIC ```
# MAGIC - `Window.partitionBy("created_at")` define a "janela": agrupe as linhas
# MAGIC   por `created_at` (todas as linhas com o mesmo timestamp formam um
# MAGIC   grupo), mas — diferente de um `groupBy` normal — sem colapsar as
# MAGIC   linhas.
# MAGIC - `F.count("*").over(batch)` — o `.over(janela)` é o que transforma uma
# MAGIC   função de agregação comum (`F.count`) numa window function. `F.count("*")`
# MAGIC   sozinho, dentro de um `.agg(...)`, contaria linhas por grupo e devolveria
# MAGIC   uma linha por grupo. Com `.over(batch)`, ele conta quantas linhas
# MAGIC   existem no grupo (`created_at`) *daquela linha*, e escreve esse número
# MAGIC   em **cada** linha do grupo — por isso o resultado tem o mesmo número de
# MAGIC   linhas que o DataFrame original.
# MAGIC - A segunda coluna compara esse tamanho de lote contra o `threshold`
# MAGIC   (padrão 10) e vira um booleano.
# MAGIC - `_batch_size` é descartada no final — ela só existia como uma variável
# MAGIC   de trabalho pra chegar em `is_bulk_load`; o prefixo `_` sinaliza isso.
# MAGIC
# MAGIC O limiar de 10 é uma decisão de julgamento, documentada explicitamente:
# MAGIC duas propostas cadastradas no mesmo segundo é plausível (coincidência);
# MAGIC dez não é.

# COMMAND ----------

BULK_THRESHOLD = 10

bids_flagged = add_is_bulk_load(bids_typed, threshold=BULK_THRESHOLD)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 5 — cobertura de motivo de perda, e fechando `bids_clean`
# MAGIC
# MAGIC `competitor_name` carrega um valor padrão (`"Competitor 1"`) gravado
# MAGIC sempre que ninguém completou a apuração pós-proposta — ele não é um
# MAGIC concorrente real. Marcá-lo explicitamente evita que ele seja contado como
# MAGIC se fosse um concorrente de verdade em qualquer agregado mais adiante.
# MAGIC
# MAGIC ### O que tem dentro de `flag_placeholder_competitor`
# MAGIC
# MAGIC ```python
# MAGIC def flag_placeholder_competitor(df, placeholder="Competitor 1"):
# MAGIC     return df.withColumn(
# MAGIC         "competitor_is_placeholder",
# MAGIC         (F.col("competitor_name") == F.lit(placeholder)).cast("boolean"),
# MAGIC     ).withColumn("has_loss_reason", F.col("loss_reason").isNotNull())
# MAGIC ```
# MAGIC Duas colunas booleanas novas: uma comparando `competitor_name` contra o
# MAGIC valor padrão (`==` entre uma coluna e um literal), e outra checando se
# MAGIC `loss_reason` está preenchido (`.isNotNull()`).
# MAGIC
# MAGIC ### O resto da célula
# MAGIC
# MAGIC ```python
# MAGIC bids_clean = (
# MAGIC     flag_placeholder_competitor(bids_flagged, placeholder=PLACEHOLDER_COMPETITOR)
# MAGIC     .withColumn(
# MAGIC         "bid_status",
# MAGIC         F.when(F.col("outcome") == 1, "won")
# MAGIC          .when(F.col("outcome") == 0, "lost")
# MAGIC          .otherwise("open"),
# MAGIC     )
# MAGIC )
# MAGIC ```
# MAGIC `F.when(...).when(...).otherwise(...)` — o mesmo padrão de if/else de
# MAGIC antes, agora encadeado com múltiplas condições: é um "if / elif / elif /
# MAGIC else" dentro de uma expressão de coluna. Aqui traduz o número (`outcome`
# MAGIC 1, 0 ou NULL) pra um texto legível (`"won"`, `"lost"`, `"open"`).
# MAGIC
# MAGIC ```python
# MAGIC bids_clean.write.format("delta").mode("overwrite").saveAsTable("silver.bid.bids_clean")
# MAGIC ```
# MAGIC Grava o resultado final desta parte do notebook como a tabela
# MAGIC `silver.bid.bids_clean` — a tabela que `03_gold_bid_performance` vai ler.

# COMMAND ----------

PLACEHOLDER_COMPETITOR = "Competitor 1"

bids_clean = (
    flag_placeholder_competitor(bids_flagged, placeholder=PLACEHOLDER_COMPETITOR)
    .withColumn(
        "bid_status",
        F.when(F.col("outcome") == 1, "won")
         .when(F.col("outcome") == 0, "lost")
         .otherwise("open"),
    )
)

bids_clean.write.format("delta").mode("overwrite").saveAsTable("silver.bid.bids_clean")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 6 — clients: datas sentinela e SCD Type 2
# MAGIC
# MAGIC ### `2999-12-31`, resolvido
# MAGIC
# MAGIC Esse valor sentinela marca um contrato sem prazo definido. Mantido como
# MAGIC data, ele colocaria um contrato de quase mil anos em qualquer cálculo de
# MAGIC duração — vira `NULL`, com um flag explícito `is_open_ended` guardando a
# MAGIC informação de que era, sim, um contrato aberto (e não um dado faltando).
# MAGIC
# MAGIC ### O que é SCD Type 2 (o conceito novo desta seção)
# MAGIC
# MAGIC "SCD" = **Slowly Changing Dimension** ("dimensão de mudança lenta"), um
# MAGIC padrão clássico de modelagem dimensional. O problema que ele resolve:
# MAGIC um cliente pode renovar contrato, e cada renovação é um novo período de
# MAGIC vigência — não uma correção do registro anterior. Se você simplesmente
# MAGIC sobrescrevesse a linha do cliente a cada renovação ("Type 1"), perderia o
# MAGIC histórico: não teria como saber que atributos (segmento, executivo
# MAGIC responsável, etc.) valiam pra esse cliente há dois anos, quando uma
# MAGIC proposta específica foi feita.
# MAGIC
# MAGIC "Type 2" resolve isso mantendo **uma linha por versão**, cada uma com um
# MAGIC intervalo de validade explícito:
# MAGIC - `valid_from` — a partir de quando essa versão vale
# MAGIC - `valid_to` — até quando (exclusive); `NULL` significa "vale até hoje,
# MAGIC   é a versão atual" — não "eu não sei"
# MAGIC - `is_current` — `True` quando `valid_to IS NULL`
# MAGIC - `client_sk` — uma chave substituta (*surrogate key*) que identifica um
# MAGIC   par (cliente, versão) especificamente, diferente de `client_id`, que
# MAGIC   continua sendo a chave de negócio (identifica *quem* é o cliente,
# MAGIC   independente de qual versão)
# MAGIC
# MAGIC Com isso montado, um notebook mais adiante (`03_gold_bid_performance`)
# MAGIC pode escolher: quer a versão *atual* do cliente, ou a versão que estava
# MAGIC *vigente no momento em que uma proposta específica foi criada*? O
# MAGIC trabalho deste notebook termina em deixar as duas perguntas possíveis de
# MAGIC responder.
# MAGIC
# MAGIC ### O que tem dentro de `build_client_scd2` (em `transforms.py`)
# MAGIC
# MAGIC ```python
# MAGIC def build_client_scd2(df, sentinel="2999-12-31"):
# MAGIC     typed = (
# MAGIC         df.withColumn("client_id", F.col("client_id").cast("long"))
# MAGIC         .withColumn("is_open_ended", (F.col("end_date").startswith(sentinel)).cast("boolean"))
# MAGIC         .withColumn(
# MAGIC             "end_date",
# MAGIC             F.when(F.col("end_date").startswith(sentinel), None).otherwise(F.to_date("end_date")),
# MAGIC         )
# MAGIC         .withColumn("start_date", F.to_date("start_date"))
# MAGIC     )
# MAGIC ```
# MAGIC Primeiro bloco: tipa `client_id`, marca `is_open_ended` comparando o texto
# MAGIC bruto contra o sentinela (`.startswith(...)`, porque a data pode vir com
# MAGIC hora colada), e resolve `end_date` — vira `NULL` se for o sentinela,
# MAGIC senão faz o parse pra data de verdade com `F.to_date`.
# MAGIC
# MAGIC ```python
# MAGIC     version_order = Window.partitionBy("client_id").orderBy(
# MAGIC         F.col("start_date").asc_nulls_first(), F.col("end_date").asc_nulls_last()
# MAGIC     )
# MAGIC ```
# MAGIC Outra window function — mas agora com `.orderBy(...)` além do
# MAGIC `.partitionBy(...)`. `.partitionBy("client_id")` agrupa as versões do
# MAGIC mesmo cliente; `.orderBy(...)` define a ordem *dentro* de cada grupo:
# MAGIC primeiro por `start_date` (mais antiga primeiro; `asc_nulls_first` manda
# MAGIC nulos pro início, caso existam), depois por `end_date` como desempate.
# MAGIC Isso importa porque a próxima linha depende dessa ordem existir.
# MAGIC
# MAGIC ```python
# MAGIC     return (
# MAGIC         typed.withColumn("valid_from", F.col("start_date"))
# MAGIC         .withColumn("valid_to", F.lead("start_date").over(version_order))
# MAGIC         .withColumn("is_current", F.col("valid_to").isNull())
# MAGIC         .withColumn("client_sk", F.monotonically_increasing_id())
# MAGIC     )
# MAGIC ```
# MAGIC - `valid_from` é simplesmente o `start_date` daquela versão.
# MAGIC - `F.lead("start_date").over(version_order)` é o coração do SCD Type 2:
# MAGIC   `F.lead(coluna)` olha pra **a próxima linha**, na ordem definida pela
# MAGIC   window (`version_order`). Então `valid_to` de uma versão vira o
# MAGIC   `start_date` da *próxima* versão daquele mesmo cliente — exatamente a
# MAGIC   definição de "até quando essa versão valeu". Pra última versão de cada
# MAGIC   cliente, não existe "próxima linha", então `F.lead` devolve `NULL` —
# MAGIC   que é exatamente o que queremos: `NULL` = versão atual.
# MAGIC - `is_current` é só a checagem direta: `valid_to` é nulo?
# MAGIC - `F.monotonically_increasing_id()` gera um número único e crescente pra
# MAGIC   cada linha — não é sequencial sem buracos, mas garante unicidade, que é
# MAGIC   tudo que uma chave substituta precisa.

# COMMAND ----------

SENTINEL = "2999-12-31"

clients_clean = build_client_scd2(clients, sentinel=SENTINEL)

# Fazendo DROP antes de escrever, não só overwriteSchema=true: as colunas
# desta tabela mudaram (foram adicionadas valid_from/valid_to/is_current/
# client_sk), e uma definição de tabela do Unity Catalog desatualizada
# pode discordar do novo metadata do Delta mesmo com overwriteSchema
# ligado, lançando DELTA_METADATA_MISMATCH. Um drop + recriação limpa
# evita essa classe inteira de erro — consistente com o padrão de
# "overwrite completo, não incremental" deste pipeline em todo o resto.
spark.sql("DROP TABLE IF EXISTS silver.bid.clients_clean")
clients_clean.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable("silver.bid.clients_clean")

# Lê de volta em vez de reusar o DataFrame clients_clean em memória abaixo:
# se duas versões algum dia empatarem exatamente em (start_date, end_date)
# — o gerador antigo produzia duplicatas do mesmo dia antes de ser
# corrigido pra dar às renovações um gap real —, a window function do
# Spark não tem garantia de desempatar da mesma forma em toda
# reavaliação do mesmo plano lazy. A tabela recém-gravada em disco é a
# única coisa com garantia de bater com o que realmente foi persistido.
clients_clean = spark.table("silver.bid.clients_clean")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checando que a construção do SCD2 não quebrou em algum lugar
# MAGIC
# MAGIC Todo `client_id` deve ter **exatamente uma** versão vigente (`is_current
# MAGIC = True`) — isso é garantido pela própria lógica do `F.lead` acima (a
# MAGIC última versão de cada partição sempre tem `valid_to = NULL`), mas é
# MAGIC checado aqui de qualquer jeito, porque uma violação silenciosa quebraria
# MAGIC todo join point-in-time mais adiante de um jeito difícil de rastrear.
# MAGIC
# MAGIC ```python
# MAGIC bad_versioning = (
# MAGIC     clients_clean.groupBy("client_id")
# MAGIC     .agg(F.sum(F.col("is_current").cast("int")).alias("n_current"))
# MAGIC     .filter("n_current != 1")
# MAGIC     .count()
# MAGIC )
# MAGIC assert bad_versioning == 0, ...
# MAGIC ```
# MAGIC Isso é um `groupBy` de verdade (colapsando linhas, diferente das window
# MAGIC functions de antes): agrupa por `client_id`, soma quantas versões estão
# MAGIC marcadas como atuais (`.cast("int")` transforma `True`/`False` em `1`/`0`
# MAGIC pra poder somar), e filtra os casos onde essa soma não é exatamente 1. Se
# MAGIC essa contagem final não for zero, o `assert` para o notebook.

# COMMAND ----------

# Todo client_id deve ter exatamente uma versão vigente — garantido por
# construção (a última linha em cada partição sempre tem valid_to = NULL),
# checado explicitamente mesmo assim porque uma violação silenciosa aqui
# quebraria silenciosamente todo join point-in-time a jusante.
bad_versioning = (
    clients_clean.groupBy("client_id")
    .agg(F.sum(F.col("is_current").cast("int")).alias("n_current"))
    .filter("n_current != 1")
    .count()
)
assert bad_versioning == 0, f"{bad_versioning} client_ids não têm exatamente uma versão vigente"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 7 — integridade referencial
# MAGIC
# MAGIC Duas checagens diferentes aqui, e a diferença entre elas importa:
# MAGIC
# MAGIC 1. **O cliente existe, de alguma forma?** — Uma proposta apontando pra um
# MAGIC    `client_id` que não existe em `clients_clean` de jeito nenhum seria
# MAGIC    descartada silenciosamente por um inner join mais adiante. Checar
# MAGIC    aqui transforma isso num número visível, em vez de uma contagem de
# MAGIC    linhas que encolhe sem explicação.
# MAGIC 2. **O cliente existe, *na data certa*?** — Mesmo que o `client_id`
# MAGIC    exista, o `created_at` da proposta pode cair fora de toda janela
# MAGIC    `[valid_from, valid_to)` conhecida daquele cliente. Essa checagem
# MAGIC    *espera* achar uma contagem não-trivial (cerca de 6%): alguns
# MAGIC    clientes têm `start_date` desconhecido (o placeholder `'-'`, que virou
# MAGIC    NULL no `denull`), e algumas propostas foram criadas antes do início
# MAGIC    de contrato mais antigo conhecido daquele cliente — uma inconsistência
# MAGIC    real na fonte, não um bug deste join.
# MAGIC
# MAGIC ```python
# MAGIC orphans = (
# MAGIC     bids_clean.join(clients_clean.select("client_id").distinct(), "client_id", "left_anti").count()
# MAGIC )
# MAGIC ```
# MAGIC `"left_anti"` é um tipo de join que devolve **as linhas da esquerda que
# MAGIC não têm par na direita** — o oposto de um inner join. Aqui: propostas
# MAGIC cujo `client_id` não existe em nenhuma versão de `clients_clean`.
# MAGIC `.distinct()` remove duplicatas do lado dos clientes antes do join, já
# MAGIC que um mesmo `client_id` aparece em múltiplas versões SCD2 e a gente só
# MAGIC quer saber se ele existe, não contar versões.
# MAGIC
# MAGIC ```python
# MAGIC coverage_check = (
# MAGIC     bids_clean.alias("b")
# MAGIC     .join(
# MAGIC         clients_clean.alias("c"),
# MAGIC         (F.col("b.client_id") == F.col("c.client_id"))
# MAGIC         & (F.col("b.created_at") >= F.col("c.valid_from").cast("timestamp"))
# MAGIC         & (F.col("c.valid_to").isNull() | (F.col("b.created_at") < F.col("c.valid_to").cast("timestamp"))),
# MAGIC         "left_anti",
# MAGIC     )
# MAGIC )
# MAGIC ```
# MAGIC Este é o **join point-in-time** — o mesmo padrão que `03_gold_bid_performance`
# MAGIC usa pra associar cada proposta à versão certa do cliente. Peça por peça:
# MAGIC - `.alias("b")` / `.alias("c")` dão um apelido a cada DataFrame, necessário
# MAGIC   porque os dois têm uma coluna `client_id` — sem o apelido, `F.col("client_id")`
# MAGIC   seria ambíguo (o Spark não saberia de qual dos dois você está falando).
# MAGIC - A condição de join não é só `client_id == client_id` — tem três partes
# MAGIC   unidas por `&` (E lógico): o `client_id` bate, **e** o `created_at` da
# MAGIC   proposta é maior ou igual a `valid_from` daquela versão, **e** (o
# MAGIC   `valid_to` é nulo — versão atual, vale pra sempre — **ou** o
# MAGIC   `created_at` é menor que `valid_to`).
# MAGIC - Isso é literalmente a definição de "este `created_at` cai dentro da
# MAGIC   janela de validade desta versão do cliente".
# MAGIC - De novo `"left_anti"`: propostas que **não** casaram com nenhuma versão
# MAGIC   nessa condição — essas são as ~6% "sem cobertura point-in-time".

# COMMAND ----------

# Integridade referencial: o cliente existe, de alguma forma? (Qualquer
# versão — validade temporal é uma preocupação separada, checada a seguir.)
orphans = (
    bids_clean.join(clients_clean.select("client_id").distinct(), "client_id", "left_anti").count()
)
print(f"propostas órfãs: {orphans}")

# Cobertura point-in-time: o created_at de cada proposta realmente cai
# dentro da janela [valid_from, valid_to) de alguma versão do seu cliente?
# Um gap aqui significaria que o join point-in-time do Gold descarta os
# atributos de cliente daquela proposta silenciosamente.
coverage_check = (
    bids_clean.alias("b")
    .join(
        clients_clean.alias("c"),
        (F.col("b.client_id") == F.col("c.client_id"))
        & (F.col("b.created_at") >= F.col("c.valid_from").cast("timestamp"))
        & (F.col("c.valid_to").isNull() | (F.col("b.created_at") < F.col("c.valid_to").cast("timestamp"))),
        "left_anti",
    )
)
uncovered = coverage_check.count()
print(f"propostas sem versão de cliente correspondente no created_at: {uncovered}")
