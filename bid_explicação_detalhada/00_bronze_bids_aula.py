# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Bronze: bids (versão aula)
# MAGIC
# MAGIC Esta é a versão didática deste notebook. O código é **exatamente o mesmo**
# MAGIC do `00_bronze_bids_pt-BR` — mesmas tabelas, mesmo resultado. A diferença é
# MAGIC só o quanto se explica: aqui, cada método é explicado como se você nunca
# MAGIC tivesse visto PySpark antes.
# MAGIC
# MAGIC ## O que este notebook faz, em uma frase
# MAGIC
# MAGIC Pega o Excel bruto de propostas (`bids.xlsx`) e grava ele, sem mudar nada,
# MAGIC numa tabela Delta chamada `bronze.bid.bids`.
# MAGIC
# MAGIC ## Por que "sem mudar nada"?
# MAGIC
# MAGIC Essa é a regra da camada **Bronze** num pipeline medallion (Bronze → Silver
# MAGIC → Gold, que você vai ver nos próximos notebooks): Bronze é uma cópia fiel da
# MAGIC fonte. Nada de cast de tipo, nada de limpeza, nada de filtro. Se um dado
# MAGIC parece errado aqui, ele é preservado do mesmo jeito — quem decide o que
# MAGIC fazer com ele é o notebook seguinte (`02_silver_bids`), não este.
# MAGIC
# MAGIC Por quê guardar até o "lixo"? Porque se você limpa um dado na ingestão e
# MAGIC depois descobre que a limpeza estava errada, o dado original já se foi.
# MAGIC Guardando a cópia fiel na Bronze, você sempre pode reprocessar a partir
# MAGIC dela.
# MAGIC
# MAGIC ## Por que aceitar mudança de schema em vez de travar o notebook?
# MAGIC
# MAGIC A fonte aqui é um Excel exportado manualmente por alguém, não um sistema
# MAGIC automatizado — as colunas podem mudar de nome, sumir ou aparecer sem
# MAGIC ninguém avisar. Se este notebook falhasse toda vez que uma coluna nova
# MAGIC aparecesse, você teria o pipeline quebrando por causa de uma mudança
# MAGIC cosmética que nem afeta o que você precisa. Em vez disso, a Bronze aceita
# MAGIC a mudança e só *avisa*; quem exige rigidez de schema é o Silver, que valida
# MAGIC contra um contrato explícito.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 1 — configurar onde tudo vai morar
# MAGIC
# MAGIC No Databricks com Unity Catalog, toda tabela vive dentro de uma hierarquia
# MAGIC de três níveis: **catálogo → schema → tabela**. Pense nisso como pastas
# MAGIC dentro de pastas: `bronze.bid.bids` significa "a tabela `bids`, dentro do
# MAGIC schema `bid`, dentro do catálogo `bronze`".
# MAGIC
# MAGIC - `CATALOG` — o catálogo. Aqui separamos por camada do pipeline: existe um
# MAGIC   catálogo `bronze`, um `silver`, um `gold` (você vai ver os outros dois nos
# MAGIC   próximos notebooks).
# MAGIC - `SCHEMA` — dentro do catálogo, um agrupamento lógico. Aqui é `bid`,
# MAGIC   porque esse é o único domínio de negócio do projeto.
# MAGIC - `TABLE` — o nome final, montado com uma f-string (`f"{CATALOG}.{SCHEMA}.bids"`
# MAGIC   vira o texto `"bronze.bid.bids"`).
# MAGIC
# MAGIC `spark.sql(...)` roda comandos SQL puros dentro do Python — aqui, dois
# MAGIC comandos `CREATE ... IF NOT EXISTS`, que criam o catálogo e o schema caso
# MAGIC ainda não existam (e não fazem nada, sem erro, se já existirem — por isso
# MAGIC o `IF NOT EXISTS`).

# COMMAND ----------

CATALOG = "bronze"
SCHEMA = "bid"
VOLUME_PATH = "/Volumes/raw/bid/bids/bids.xlsx"
TABLE = f"{CATALOG}.{SCHEMA}.bids"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 2 — ler o Excel
# MAGIC
# MAGIC ### Por que pandas, se o resto do projeto é PySpark?
# MAGIC
# MAGIC Spark não sabe ler `.xlsx` nativamente — ele foi feito pra ler formatos como
# MAGIC CSV, Parquet, Delta, JSON. Pra Excel, a opção mais simples é deixar o
# MAGIC **pandas** ler o arquivo primeiro (pandas sabe ler Excel muito bem, com a
# MAGIC biblioteca `openpyxl` por baixo) e depois converter o resultado pra um
# MAGIC DataFrame do Spark.
# MAGIC
# MAGIC A alternativa seria instalar um pacote Java/Maven específico pra Excel
# MAGIC (`com.crealytics.spark.excel`), mas isso exige configurar o cluster —
# MAGIC pandas + openpyxl não exige nenhuma configuração de cluster, então este
# MAGIC notebook roda em qualquer lugar.
# MAGIC
# MAGIC ### O que cada linha faz
# MAGIC
# MAGIC ```python
# MAGIC pdf_raw = pd.read_excel(VOLUME_PATH, sheet_name="Bronze", dtype=str)
# MAGIC ```
# MAGIC - `pd.read_excel(...)` — lê o arquivo Excel e devolve um DataFrame do
# MAGIC   **pandas** (por isso o prefixo `pdf_` — "pandas DataFrame" — pra
# MAGIC   diferenciar de um DataFrame do Spark, que a convenção deste projeto
# MAGIC   prefixa com `df_`).
# MAGIC - `sheet_name="Bronze"` — lê pelo **nome** da aba, não por um intervalo fixo
# MAGIC   de células tipo `"Bronze!A1:K1583"`. Isso importa: se a fonte crescer uma
# MAGIC   linha amanhã, um intervalo fixo cortaria essa linha nova *sem erro
# MAGIC   nenhum* — você simplesmente perderia dado sem saber. Ler pelo nome da
# MAGIC   aba não tem esse problema.
# MAGIC - `dtype=str` — força toda coluna a ser lida como texto, mesmo que pareça
# MAGIC   número ou data. É o equivalente, em pandas, ao `inferSchema=false` do
# MAGIC   Spark. Por quê? Porque decidir o tipo certo de cada coluna (é uma data?
# MAGIC   um inteiro? um decimal?) é trabalho da camada Silver, não da Bronze — a
# MAGIC   Bronze só guarda o dado como uma cópia fiel do texto original.
# MAGIC
# MAGIC ```python
# MAGIC df_raw = spark.createDataFrame(pdf_raw)
# MAGIC ```
# MAGIC - Converte o DataFrame do pandas (que vive na memória de um único
# MAGIC   computador) para um DataFrame do **Spark** (que pode ser distribuído
# MAGIC   entre vários computadores). A partir daqui, tudo que fizermos usa a API
# MAGIC   do Spark, não do pandas.
# MAGIC
# MAGIC ```python
# MAGIC print(f"linhas: {df_raw.count()}   colunas: {len(df_raw.columns)}")
# MAGIC df_raw.printSchema()
# MAGIC ```
# MAGIC - `.count()` — conta quantas linhas o DataFrame tem. Em Spark isso
# MAGIC   dispara de fato uma execução (Spark é "preguiçoso" — só processa quando
# MAGIC   você pede um resultado concreto, tipo um count ou um write).
# MAGIC - `df_raw.columns` — lista com os nomes das colunas; `len(...)` conta
# MAGIC   quantas são.
# MAGIC - `.printSchema()` — imprime a estrutura do DataFrame: nome de cada coluna
# MAGIC   e o tipo que o Spark atribuiu a ela. Como lemos tudo com `dtype=str`,
# MAGIC   toda coluna aqui deve aparecer como `string`.
# MAGIC
# MAGIC ### Sobre o `%pip install`
# MAGIC
# MAGIC A primeira linha da célula abaixo, `%pip install openpyxl`, é um "comando
# MAGIC mágico" do Databricks (por isso o `%`) — instala a biblioteca `openpyxl`
# MAGIC no cluster antes de rodar o resto da célula. `pandas.read_excel` depende
# MAGIC dela por baixo dos panos pra conseguir ler arquivos `.xlsx`.

# COMMAND ----------

# MAGIC %pip install openpyxl
# MAGIC
# MAGIC import pandas as pd
# MAGIC
# MAGIC pdf_raw = pd.read_excel(VOLUME_PATH, sheet_name="Bronze", dtype=str)
# MAGIC df_raw = spark.createDataFrame(pdf_raw)
# MAGIC
# MAGIC print(f"linhas: {df_raw.count()}   colunas: {len(df_raw.columns)}")
# MAGIC df_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 3 — adicionar metadados de ingestão
# MAGIC
# MAGIC Antes de gravar, adicionamos duas colunas que **não existem na fonte** —
# MAGIC elas são geradas por este notebook, não vêm do Excel:
# MAGIC
# MAGIC - `_ingested_at` — quando essa linha foi carregada.
# MAGIC - `_source_file` — de qual arquivo ela veio.
# MAGIC
# MAGIC Por que isso importa? Imagine que, seis meses depois, alguém pergunta "essa
# MAGIC linha estranha apareceu quando, e veio de onde?" Sem essas duas colunas,
# MAGIC você não tem resposta — é adivinhação. Com elas, é uma consulta SQL.
# MAGIC O prefixo `_` (underscore) é só uma convenção pra sinalizar "coluna técnica,
# MAGIC não é dado de negócio".
# MAGIC
# MAGIC ### O que cada linha faz
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.sql import functions as F
# MAGIC ```
# MAGIC Importa o módulo de funções do Spark SQL, convencionalmente apelidado de
# MAGIC `F`. Praticamente toda transformação de coluna neste projeto (aqui e nos
# MAGIC próximos notebooks) usa alguma função de dentro de `F`.
# MAGIC
# MAGIC ```python
# MAGIC df_bronze = (
# MAGIC     df_raw
# MAGIC     .withColumn("_ingested_at", F.current_timestamp())
# MAGIC     .withColumn("_source_file", F.lit(VOLUME_PATH))
# MAGIC )
# MAGIC ```
# MAGIC - `.withColumn("nome", expressão)` é o método mais usado neste projeto
# MAGIC   inteiro — ele **adiciona uma coluna nova** (ou substitui uma existente, se
# MAGIC   o nome já existir) a um DataFrame. Importante: DataFrames em Spark são
# MAGIC   **imutáveis** — `.withColumn(...)` não modifica `df_raw`, ele devolve um
# MAGIC   DataFrame *novo* com a coluna adicionada. É por isso que encadeamos
# MAGIC   `.withColumn(...).withColumn(...)` — cada chamada recebe o resultado da
# MAGIC   anterior e devolve outro DataFrame novo.
# MAGIC - `F.current_timestamp()` — uma função do Spark que devolve o timestamp de
# MAGIC   agora, no momento em que o notebook roda.
# MAGIC - `F.lit(VOLUME_PATH)` — `lit` é abreviação de "literal". Diferente de
# MAGIC   `F.current_timestamp()`, que calcula um valor, `F.lit(...)` simplesmente
# MAGIC   pega o valor Python que você já tem (aqui, a string `VOLUME_PATH`) e o
# MAGIC   transforma numa coluna Spark com esse valor **repetido em toda linha**.
# MAGIC   Você precisa de `F.lit(...)` sempre que quer colocar um valor fixo do
# MAGIC   Python dentro de uma expressão de coluna Spark — sem ele, o Spark não
# MAGIC   entende `VOLUME_PATH` como uma coluna.
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
# MAGIC Isso grava o DataFrame como uma tabela de verdade no Unity Catalog:
# MAGIC - `.write` — inicia a operação de escrita.
# MAGIC - `.format("delta")` — o formato de arquivo usado é **Delta Lake**, o
# MAGIC   formato padrão do Databricks (Parquet por baixo, com um log de
# MAGIC   transações em cima — é o que permite versionamento, time travel, e
# MAGIC   updates/deletes eficientes).
# MAGIC - `.mode("overwrite")` — se a tabela já existir, ela é **substituída
# MAGIC   inteira**, não apendada. Cada vez que este notebook roda, ele recria a
# MAGIC   tabela do zero a partir do Excel mais recente.
# MAGIC - `.option("mergeSchema", "true")` — permite que o schema da tabela mude
# MAGIC   entre uma execução e outra (colunas novas, por exemplo) sem dar erro.
# MAGIC   Ver a explicação completa abaixo.
# MAGIC - `.saveAsTable(TABLE)` — o nome final, `"bronze.bid.bids"`, definido lá no
# MAGIC   Passo 1.

# COMMAND ----------

from pyspark.sql import functions as F

df_bronze = (
    df_raw
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.lit(VOLUME_PATH))
)

(
    df_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("mergeSchema", "true")   # deliberado: ver nota abaixo
    .saveAsTable(TABLE)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sobre o `mergeSchema` — por que essa opção existe
# MAGIC
# MAGIC Por padrão, o Delta Lake é rígido: se você tenta gravar um DataFrame cujas
# MAGIC colunas não batem exatamente com o schema já salvo da tabela, ele recusa a
# MAGIC escrita e lança erro. Isso é bom pra maioria dos casos — evita que um bug
# MAGIC silenciosamente mude o formato dos seus dados.
# MAGIC
# MAGIC Mas aqui a fonte é uma planilha Excel mantida manualmente por alguém, que
# MAGIC pode ganhar ou perder colunas sem aviso prévio. Se a Bronze recusasse
# MAGIC qualquer mudança de schema, o pipeline pararia de funcionar toda vez que
# MAGIC isso acontecesse — mesmo que a mudança fosse inofensiva (uma coluna extra
# MAGIC que ninguém usa ainda, por exemplo).
# MAGIC
# MAGIC `.option("mergeSchema", "true")` diz ao Delta: "aceite a mudança de schema,
# MAGIC não dê erro". O custo dessa flexibilidade é que uma coluna que só teve o
# MAGIC *nome* trocado na fonte (ex.: `client_id` virou `id_cliente`) chega aqui
# MAGIC como se fosse uma coluna nova — o Spark não tem como saber que é a "mesma"
# MAGIC coluna com outro nome. É por isso que a célula seguinte existe: ela checa
# MAGIC explicitamente se as colunas que a gente *espera* continuam lá.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 4 — checar o schema esperado
# MAGIC
# MAGIC Essa célula não transforma dado nenhum — ela só **valida** que as colunas
# MAGIC que o resto do pipeline depende ainda estão presentes, antes de seguir em
# MAGIC frente. Se uma coluna crítica sumiu, é melhor o notebook falhar aqui, com
# MAGIC uma mensagem clara, do que deixar o Silver quebrar de um jeito confuso
# MAGIC três notebooks depois.
# MAGIC
# MAGIC ### O que cada linha faz
# MAGIC
# MAGIC ```python
# MAGIC EXPECTED = {
# MAGIC     "bid_id", "created_at", ...
# MAGIC }
# MAGIC ```
# MAGIC Isso é um **set** (conjunto) do Python — como uma lista, mas sem ordem e
# MAGIC sem repetição. Sets são úteis aqui porque suportam operações matemáticas de
# MAGIC conjunto (união, interseção, diferença) direto com operadores.
# MAGIC
# MAGIC ```python
# MAGIC actual = set(df_bronze.columns) - {"_ingested_at", "_source_file"}
# MAGIC ```
# MAGIC - `df_bronze.columns` é uma lista com os nomes de todas as colunas do
# MAGIC   DataFrame (incluindo as duas que acabamos de adicionar).
# MAGIC - `set(...)` converte essa lista num set.
# MAGIC - O operador `-` entre dois sets é **diferença de conjuntos**: "tudo que
# MAGIC   está no primeiro, menos o que está no segundo". Aqui, tiramos as duas
# MAGIC   colunas técnicas (`_ingested_at`, `_source_file`) porque elas não vêm da
# MAGIC   fonte — não faz sentido comparar elas contra `EXPECTED`.
# MAGIC
# MAGIC ```python
# MAGIC missing, unexpected = EXPECTED - actual, actual - EXPECTED
# MAGIC ```
# MAGIC Duas diferenças de conjunto, calculadas de uma vez e atribuídas a duas
# MAGIC variáveis (Python permite isso — "desempacotamento" de tupla):
# MAGIC - `EXPECTED - actual` — o que a gente esperava que existisse mas **não**
# MAGIC   está no schema atual → colunas que sumiram.
# MAGIC - `actual - EXPECTED` — o que está no schema atual mas **não** estava na
# MAGIC   lista esperada → colunas novas, inesperadas.
# MAGIC
# MAGIC ```python
# MAGIC if missing:
# MAGIC     raise ValueError(f"Colunas ausentes na fonte: {sorted(missing)}")
# MAGIC if unexpected:
# MAGIC     print(f"AVISO — novas colunas absorvidas, revisar Silver: {sorted(unexpected)}")
# MAGIC ```
# MAGIC - Um set vazio (`set()`) é "falsy" em Python — `if missing:` só entra no
# MAGIC   bloco se o set tiver pelo menos um item.
# MAGIC - `raise ValueError(...)` **interrompe a execução do notebook com erro**.
# MAGIC   Isso é deliberado: se uma coluna que o pipeline precisa desapareceu, é
# MAGIC   melhor parar aqui, ruidosamente, do que deixar o problema se propagar em
# MAGIC   silêncio.
# MAGIC - Já uma coluna nova e inesperada não é motivo pra parar — ela só vira um
# MAGIC   `print` de aviso, porque não quebra nada que já existe; alguém só precisa
# MAGIC   decidir depois se o Silver deve usar essa coluna nova ou ignorá-la.
# MAGIC - `sorted(...)` transforma o set numa lista ordenada só pra a mensagem
# MAGIC   sair legível (sets não têm ordem garantida).
# MAGIC
# MAGIC ```python
# MAGIC print(f"Checagem de schema aprovada. {spark.table(TABLE).count()} linhas gravadas em {TABLE}.")
# MAGIC ```
# MAGIC Última linha, só roda se nada acima interrompeu o notebook: relê a tabela
# MAGIC que acabamos de gravar (`spark.table(TABLE)`, diferente de `df_bronze` —
# MAGIC isso é a tabela persistida em disco, não o DataFrame que ainda está na
# MAGIC memória) e confirma quantas linhas foram, de fato, gravadas.

# COMMAND ----------

EXPECTED = {
    "bid_id", "created_at", "created_at_str", "is_confirmed_date", "bid_date",
    "closed_at", "closed_at_str", "outcome", "loss_reason", "competitor_name",
    "client_id", "contract_value_brl",
}

actual = set(df_bronze.columns) - {"_ingested_at", "_source_file"}
missing, unexpected = EXPECTED - actual, actual - EXPECTED

if missing:
    raise ValueError(f"Colunas ausentes na fonte: {sorted(missing)}")
if unexpected:
    print(f"AVISO — novas colunas absorvidas, revisar Silver: {sorted(unexpected)}")

print(f"Checagem de schema aprovada. {spark.table(TABLE).count()} linhas gravadas em {TABLE}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conferindo o resultado
# MAGIC
# MAGIC `%sql` é outro comando mágico — muda a célula inteira pra rodar SQL puro em
# MAGIC vez de Python. Útil pra dar uma olhada rápida na tabela sem escrever
# MAGIC `spark.sql("...").display()`.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bronze.bid.bids
