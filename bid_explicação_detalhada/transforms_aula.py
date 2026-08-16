# transforms_aula.py — versão didática de transforms.py
#
# Mesmo código, mesma lógica, mesmo resultado do transforms.py de produção.
# A única diferença são os comentários: aqui cada função é explicada como se
# quem está lendo nunca tivesse visto PySpark. Este arquivo não é importado
# por nenhum notebook — ele existe só pra leitura e estudo. Quem os notebooks
# de produção (e os notebooks "aula" 02/03/04) importam é o transforms.py
# real.
#
# Por que essa lógica vive num arquivo separado, em vez de escrita direto
# dentro de cada notebook? Porque as mesmas quatro funções são usadas em
# três lugares diferentes do pipeline — 02_silver_bids, 03_gold_bid_performance,
# e a célula de regressão do 04_analysis. Escrever a lógica uma vez aqui e
# importar nos três evita ter cópias ligeiramente diferentes da mesma coisa
# espalhadas pelo projeto — o problema clássico de corrigir um bug numa
# cópia e esquecer as outras duas.
#
# Reparem também que toda função aqui segue a mesma regra: recebe um
# DataFrame Spark e devolve outro DataFrame Spark, sem nunca ler uma tabela
# (spark.table(...)) nem gravar nada (.write...) por dentro. Isso não é
# acidental — é o que torna essas funções testáveis isoladamente, com uma
# SparkSession local, sem precisar de uma conexão real com o Databricks: eu
# monto um DataFrame pequeno na mão, chamo a função, e comparo o resultado
# contra o que eu esperava. Uma função que lê ou grava tabela por conta
# própria não dá pra testar assim — ela sempre depende do ambiente real.

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def denull(df: DataFrame, placeholders=("null", "-", "")) -> DataFrame:
    """
    Troca strings-placeholder ('null', '-', '' por padrão) por NULL de
    verdade, em toda coluna de texto do DataFrame.

    ## Por que essa função existe

    A exportação Excel que alimenta este pipeline grava a string de quatro
    caracteres 'null' onde falta um valor — não um NULL de verdade. Pro
    Spark, 'null' (o texto) e NULL (a ausência de valor) são coisas
    completamente diferentes: um filtro `.isNull()` não pega a string
    'null'. Se você não corrigisse isso, esse "NULL disfarçado" sobreviveria
    silenciosamente a qualquer filtro daqui pra frente.

    ## Como ela funciona, linha por linha
    """
    # df.dtypes devolve uma lista de pares (nome_da_coluna, tipo) — por
    # exemplo [("bid_id", "string"), ("outcome", "string"), ...]. Todo
    # campo lido do Excel chega como "string" (lembra do dtype=str na
    # leitura?), então praticamente toda coluna passa pelo filtro abaixo.
    for c, t in df.dtypes:
        # Só mexe em colunas de texto — não faz sentido procurar a string
        # 'null' dentro de uma coluna que já é número ou timestamp.
        if t == "string":
            df = df.withColumn(
                c,
                # F.when(condição, valor_se_verdadeiro).otherwise(valor_se_falso)
                # é o "if/else" do Spark, usado dentro de uma expressão de
                # coluna. Aqui: SE o valor da coluna (depois de F.trim, que
                # remove espaços em branco nas pontas — protege contra
                # ' null ' com espaço, por exemplo) estiver na lista de
                # placeholders, vira None (que o Spark entende como NULL
                # real) — SENÃO, mantém o valor original.
                #
                # F.col(c).isin(list(placeholders)) testa se o valor está
                # dentro de uma lista de opções — equivalente a várias
                # comparações "OU" encadeadas, só que mais legível.
                F.when(F.trim(F.col(c)).isin(list(placeholders)), None).otherwise(
                    F.col(c)
                ),
            )
    # O loop reatribui df = df.withColumn(...) a cada volta. Como todo
    # DataFrame Spark é imutável, cada .withColumn(...) devolve um
    # DataFrame NOVO — reatribuir a mesma variável é o que permite a
    # próxima iteração continuar de onde a anterior parou. No final, df
    # já teve todas as suas colunas de texto tratadas, uma por uma.
    return df


def add_is_bulk_load(df: DataFrame, threshold: int = 10) -> DataFrame:
    """
    Marca is_bulk_load = True em toda linha cujo created_at é compartilhado
    por `threshold` ou mais outras linhas.

    ## Por que essa função existe

    Uma grande fração das propostas foi importada em massa durante uma
    migração de sistema, em vez de registrada uma a uma conforme
    aconteciam. Elas compartilham um created_at idêntico até o segundo —
    algo que uma pessoa cadastrando manualmente nunca produziria. Essas
    propostas em lote se comportam de um jeito completamente diferente das
    orgânicas (taxa de vitória bem mais baixa), e sem essa marcação, elas
    contaminariam qualquer métrica segmentada mais adiante.

    ## O conceito novo aqui: window function

    Uma agregação normal (groupBy().agg()) pega várias linhas e devolve
    UMA linha por grupo — você perde o detalhe da linha original. Uma
    *window function* calcula algo olhando pras outras linhas do mesmo
    grupo, mas devolve UMA linha PRA CADA linha original — ela "enriquece"
    cada linha com uma informação do grupo dela, sem colapsar nada. A
    pergunta que fazemos aqui — "quantas outras linhas têm o mesmo
    created_at que esta?" — é por definição uma pergunta sobre o grupo,
    mas a resposta precisa ficar em cada linha, não resumida numa linha só.
    """
    # Window.partitionBy("created_at") define a "janela": agrupe as linhas
    # por created_at (todo mundo com o mesmo timestamp forma um grupo),
    # mas — diferente de um groupBy comum — sem colapsar linha nenhuma.
    batch = Window.partitionBy("created_at")
    return (
        df
        # F.count("*").over(janela) é o que transforma uma função de
        # agregação comum (F.count) numa window function. Sozinho, dentro
        # de um .agg(...), F.count("*") contaria linhas por grupo e
        # devolveria uma linha por grupo. Com .over(batch), ele conta
        # quantas linhas existem no grupo (mesmo created_at) DAQUELA
        # linha, e escreve esse número em CADA linha do grupo — por isso
        # o resultado final tem o mesmo número de linhas do DataFrame
        # original.
        .withColumn("_batch_size", F.count("*").over(batch))
        # Compara o tamanho do lote contra o threshold (padrão 10) e vira
        # um booleano. O limiar de 10 é uma decisão de julgamento: duas
        # propostas no mesmo segundo é plausível (coincidência); dez não é.
        .withColumn("is_bulk_load", (F.col("_batch_size") >= threshold).cast("boolean"))
        # _batch_size era só uma variável de trabalho pra chegar em
        # is_bulk_load — o underscore no nome sinaliza isso, e ela é
        # descartada assim que cumpriu seu papel.
        .drop("_batch_size")
    )


def flag_placeholder_competitor(
    df: DataFrame, placeholder: str = "Competitor 1"
) -> DataFrame:
    """
    Adiciona duas colunas booleanas: competitor_is_placeholder (o
    concorrente vencedor é o valor padrão do sistema, não um concorrente
    real?) e has_loss_reason (existe um motivo de perda registrado?).

    ## Por que essa função existe

    `competitor_name` carrega um valor padrão ("Competitor 1") gravado
    sempre que ninguém completou a apuração pós-proposta — ele não é um
    concorrente de verdade. Sem marcar isso explicitamente, qualquer
    agregado por concorrente mais adiante contaria esse valor padrão como
    se fosse um concorrente real, produzindo uma manchete falsa do tipo
    "um concorrente leva a esmagadora maioria das perdas".
    """
    return df.withColumn(
        "competitor_is_placeholder",
        # Comparação simples entre uma coluna e um valor literal. F.lit(...)
        # transforma um valor Python comum (aqui, a string `placeholder`)
        # numa "coluna" Spark com esse valor fixo — necessário sempre que
        # você quer colocar um valor Python dentro de uma expressão de
        # coluna Spark.
        (F.col("competitor_name") == F.lit(placeholder)).cast("boolean"),
    ).withColumn(
        # .isNotNull() é o oposto de .isNull() — True quando a coluna TEM
        # um valor.
        "has_loss_reason", F.col("loss_reason").isNotNull()
    )


def build_client_scd2(df: DataFrame, sentinel: str = "2999-12-31") -> DataFrame:
    """
    Transforma um DataFrame bruto de clientes numa dimensão SCD Type 2.
    Espera client_id, start_date, end_date ainda como strings (como chegam
    do Bronze). Adiciona is_open_ended, faz o cast das datas, e deriva
    valid_from / valid_to (exclusivo; NULL = versão atual) / is_current /
    client_sk.

    ## O que é SCD Type 2, em uma frase

    "SCD" = Slowly Changing Dimension ("dimensão de mudança lenta"), um
    padrão clássico de modelagem dimensional pra lidar com atributos que
    mudam ao longo do tempo. Um cliente pode renovar contrato — cada
    renovação é um NOVO período de vigência, não uma correção do registro
    anterior. Se você simplesmente sobrescrevesse a linha do cliente a
    cada renovação ("Type 1"), perderia o histórico: não teria como saber
    quais atributos (segmento, executivo responsável, etc.) valiam pra
    esse cliente há dois anos, quando uma proposta específica foi feita.

    "Type 2" resolve isso mantendo uma LINHA POR VERSÃO, cada uma com um
    intervalo de validade explícito — é isso que esta função constrói.
    """
    typed = (
        df.withColumn("client_id", F.col("client_id").cast("long"))
        # is_open_ended marca se o contrato tem prazo indeterminado, ANTES
        # de apagar essa informação transformando a data sentinela em
        # NULL — sem esse flag, perderíamos a diferença entre "contrato
        # aberto" e "data desconhecida", já que os dois viram NULL na
        # coluna end_date. .startswith(sentinel) compara o começo do
        # texto, porque a data pode vir com hora colada.
        .withColumn(
            "is_open_ended", (F.col("end_date").startswith(sentinel)).cast("boolean")
        )
        .withColumn(
            "end_date",
            # Mesmo padrão if/else de sempre: se end_date é o valor
            # sentinela ("2999-12-31" — contrato sem data de término
            # definida), vira NULL de verdade. Senão, faz o parse do
            # texto pra uma data de verdade com F.to_date. Mantida como
            # data, a sentinela colocaria um contrato de quase mil anos em
            # qualquer cálculo de duração.
            F.when(F.col("end_date").startswith(sentinel), None).otherwise(
                F.to_date("end_date")
            ),
        )
        .withColumn("start_date", F.to_date("start_date"))
    )

    # Outra window function — mas agora com .orderBy(...) além do
    # .partitionBy(...). partitionBy("client_id") agrupa as versões do
    # MESMO cliente; orderBy(...) define a ordem DENTRO de cada grupo:
    # primeiro por start_date (mais antiga primeiro; asc_nulls_first
    # manda nulos pro início, caso existam), depois por end_date como
    # desempate. Essa ordem importa porque a próxima linha depende dela
    # existir.
    version_order = Window.partitionBy("client_id").orderBy(
        F.col("start_date").asc_nulls_first(), F.col("end_date").asc_nulls_last()
    )

    return (
        # valid_from é simplesmente o start_date daquela versão.
        typed.withColumn("valid_from", F.col("start_date"))
        # F.lead(coluna).over(janela) é o coração do SCD Type 2: F.lead
        # olha pra A PRÓXIMA LINHA, na ordem definida pela window
        # (version_order). Então valid_to de uma versão vira o start_date
        # da PRÓXIMA versão daquele mesmo cliente — exatamente a definição
        # de "até quando essa versão valeu". Pra última versão de cada
        # cliente, não existe "próxima linha" dentro daquela partição, e
        # F.lead devolve NULL — que é exatamente o que queremos: NULL
        # significa "vigente até hoje", não "eu não sei".
        .withColumn("valid_to", F.lead("start_date").over(version_order))
        # is_current é só a checagem direta: valid_to é nulo?
        .withColumn("is_current", F.col("valid_to").isNull())
        # F.monotonically_increasing_id() gera um número único e crescente
        # pra cada linha — não é sequencial sem buracos, mas garante
        # unicidade, que é tudo que uma chave substituta (surrogate key)
        # precisa ter. client_sk identifica um par (cliente, versão)
        # especificamente — diferente de client_id, que continua sendo a
        # chave de negócio, identificando *quem* é o cliente, não *qual*
        # versão.
        .withColumn("client_sk", F.monotonically_increasing_id())
    )


def point_in_time_join(
    bids: DataFrame,
    clients_scd2: DataFrame,
    client_cols=(
        "client_sk",
        "is_current",
        "segment",
        "state",
        "city",
        "account_executive",
        "director",
        "manager",
        "coordinator",
    ),
) -> DataFrame:
    """
    Junta propostas à versão do cliente que estava vigente no momento em
    que cada proposta foi criada. `bids` precisa ter client_id e created_at
    (timestamp). `clients_scd2` precisa ser o resultado de
    build_client_scd2. Propostas sem versão de cliente correspondente
    recebem NULL em toda coluna de client_cols, em vez de serem
    descartadas — espelha o LEFT join usado em 03_gold_bid_performance e
    na célula de regressão do 04_analysis.

    ## Por que esse join não é simplesmente "por client_id"

    clients_scd2 é uma dimensão SCD Type 2 — um mesmo cliente pode ter
    várias linhas, uma por versão de contrato. Um join comum, só por
    client_id, faria "fan-out": toda proposta de um cliente que renovou
    contrato casaria com AS DUAS versões daquele cliente, contando a mesma
    proposta duas vezes em qualquer soma ou contagem mais adiante. A
    pergunta certa não é "qual é a versão atual do cliente hoje", é "qual
    versão estava vigente quando ESSA proposta específica foi criada".
    """
    # .alias(...) dá um apelido a cada DataFrame — necessário porque os
    # dois têm uma coluna client_id. Sem o apelido, F.col("client_id")
    # seria ambíguo: o Spark não saberia de qual dos dois DataFrames você
    # está falando.
    b = bids.alias("b")
    c = clients_scd2.alias("c")

    joined = b.join(
        c,
        # A condição de join tem três partes unidas por & (E lógico) — não
        # é só "client_id bate":
        # 1) o client_id bate,
        (F.col("b.client_id") == F.col("c.client_id"))
        # 2) E o created_at da proposta é maior ou igual ao valid_from
        #    daquela versão,
        & (F.col("b.created_at") >= F.col("c.valid_from").cast("timestamp"))
        # 3) E (o valid_to é nulo — versão atual, vale pra sempre — OU o
        #    created_at é menor que o valid_to daquela versão).
        & (
            F.col("c.valid_to").isNull()
            | (F.col("b.created_at") < F.col("c.valid_to").cast("timestamp"))
        ),
        # "left" mantém TODAS as propostas, mesmo as sem versão
        # correspondente — que ficam com os atributos de cliente em NULL
        # em vez de sumirem do resultado. Isso é diferente do "left_anti"
        # usado no 02_silver_bids só pra CONTAR quantas propostas não
        # casam com nenhuma versão.
        "left",
    )

    # Monta a lista final de colunas: todas as colunas de bids ("b.*"),
    # mais só as colunas específicas do cliente pedidas em client_cols —
    # não TODAS as colunas de clients_scd2. Isso evita ambiguidade, já que
    # client_id existe nos dois lados e apareceria duplicado se a gente
    # pegasse "c.*" inteiro.
    select_cols = ["b.*"] + [F.col(f"c.{col}") for col in client_cols]
    return joined.select(*select_cols)
