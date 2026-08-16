# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Análise: do Gold aos dois achados principais (versão aula)
# MAGIC
# MAGIC Mesmo código do `04_analysis_pt-BR` — mesmos gráficos, mesmos números.
# MAGIC Este notebook **não calcula nada novo em Spark** — ele lê as tabelas
# MAGIC `gold.bid.*` que o `03_gold_bid_performance` já gravou, traz elas pra
# MAGIC pandas (bem menores agora, já agregadas) e faz dois tipos de coisa que
# MAGIC você ainda não viu nos notebooks anteriores: gráficos com matplotlib, e
# MAGIC uma regressão logística com statsmodels.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Passo 1 — trazer os agregados do Gold pra pandas
# MAGIC
# MAGIC ```python
# MAGIC overall = spark.table("gold.bid.performance_overall").toPandas()
# MAGIC ```
# MAGIC `.toPandas()` converte um DataFrame **Spark** (distribuído, preparado pra
# MAGIC processar dado grande em vários nós) pra um DataFrame **pandas** (vive na
# MAGIC memória de um computador só). Isso só é seguro fazer quando o resultado é
# MAGIC pequeno — cada tabela `gold.bid.*` aqui tem um punhado de linhas por
# MAGIC dimensão, porque já são agregados, não a base de propostas inteira. Trazer
# MAGIC a tabela `fact` (1.141 linhas de propostas fechadas) pra pandas também
# MAGIC seria tranquilo neste projeto pequeno, mas o princípio geral —
# MAGIC "`.toPandas()` só depois de agregar, nunca a tabela bruta grande" — é o
# MAGIC que evita estourar a memória de um nó quando o dado real é grande de
# MAGIC verdade.
# MAGIC
# MAGIC As oito tabelas lidas aqui são exatamente as oito que
# MAGIC `03_gold_bid_performance` gravou: `performance_overall`,
# MAGIC `performance_by_channel`, `performance_by_account_executive`,
# MAGIC `performance_by_segment`, `performance_by_value_band`,
# MAGIC `loss_reason_coverage`, `loss_reasons`, `open_pipeline`.
# MAGIC
# MAGIC ```python
# MAGIC plt.rcParams["figure.dpi"] = 110
# MAGIC plt.rcParams["axes.spines.top"] = False
# MAGIC plt.rcParams["axes.spines.right"] = False
# MAGIC ```
# MAGIC Configuração global do matplotlib, feita uma vez no topo: `dpi` controla
# MAGIC a resolução das imagens, e as duas linhas de `spines` removem a borda de
# MAGIC cima e da direita de todo gráfico deste notebook (puramente estético).

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Somente agregados — toda tabela gold aqui é pequena o bastante (um punhado
# de linhas por dimensão) para que um toPandas() de nó único seja a escolha
# certa. Trazer a fact table subjacente pro driver não seria.
overall            = spark.table("gold.bid.performance_overall").toPandas()
by_channel         = spark.table("gold.bid.performance_by_channel").toPandas()
by_executive       = spark.table("gold.bid.performance_by_account_executive").toPandas()
by_segment         = spark.table("gold.bid.performance_by_segment").toPandas()
by_value_band      = spark.table("gold.bid.performance_by_value_band").toPandas()
loss_coverage      = spark.table("gold.bid.loss_reason_coverage").toPandas()
loss_reasons       = spark.table("gold.bid.loss_reasons").toPandas()
open_pipeline      = spark.table("gold.bid.open_pipeline").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Achado 1 — o artefato de migração
# MAGIC
# MAGIC Propostas carregadas em lote (você viu `is_bulk_load` ser criado no
# MAGIC `02_silver_bids`) convertem a aproximadamente um terço da taxa das
# MAGIC registradas organicamente. O gap não é ruído — é 58% da base se
# MAGIC comportando como uma população diferente, concentrada na carteira de um
# MAGIC único executivo.
# MAGIC
# MAGIC ### Construindo o gráfico, painel por painel
# MAGIC
# MAGIC ```python
# MAGIC fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# MAGIC ```
# MAGIC `plt.subplots(linhas, colunas)` cria uma figura com uma grade de gráficos
# MAGIC — aqui, 1 linha e 2 colunas, ou seja, dois gráficos lado a lado. `fig` é a
# MAGIC figura inteira; `axes` é uma lista com os dois eixos individuais
# MAGIC (`axes[0]` = painel esquerdo, `axes[1]` = painel direito).
# MAGIC
# MAGIC ```python
# MAGIC channel_labels = by_channel["is_bulk_load"].map({True: "Carga em lote", False: "Orgânico"})
# MAGIC axes[0].bar(channel_labels, by_channel["win_rate_by_count"], color=["#c0c0c0", "#2b6cb0"])
# MAGIC ```
# MAGIC `by_channel["is_bulk_load"]` é uma **coluna de um DataFrame pandas** (uma
# MAGIC `Series`). `.map({...})` troca cada valor pelo correspondente no
# MAGIC dicionário — aqui, transforma `True`/`False` em textos legíveis pro
# MAGIC gráfico. `axes[0].bar(x, altura, color=[...])` desenha um gráfico de
# MAGIC barras no primeiro painel: um rótulo e uma altura por barra, com cores
# MAGIC diferentes (cinza pro lote, azul pro orgânico).
# MAGIC
# MAGIC ```python
# MAGIC axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
# MAGIC for i, v in enumerate(by_channel["win_rate_by_count"]):
# MAGIC     axes[0].text(i, v + 1, f"{v:.1f}%", ha="center")
# MAGIC ```
# MAGIC `PercentFormatter()` faz o eixo Y mostrar `%` automaticamente.
# MAGIC `enumerate(lista)` percorre uma sequência devolvendo `(índice, valor)` a
# MAGIC cada volta — aqui, usado pra escrever o valor exato (`f"{v:.1f}%"`, uma
# MAGIC casa decimal) em cima de cada barra, na posição `x=i` (a posição da
# MAGIC barra), `y=v+1` (um pouco acima do topo da barra).
# MAGIC
# MAGIC ```python
# MAGIC exec4 = by_executive[by_executive["account_executive"] == "Executive 4"].iloc[0]
# MAGIC ```
# MAGIC Filtragem de DataFrame pandas: `by_executive["account_executive"] ==
# MAGIC "Executive 4"` cria uma série de `True`/`False`; usar essa série entre
# MAGIC colchetes (`by_executive[...]`) filtra só as linhas onde é `True`.
# MAGIC `.iloc[0]` pega a primeira (e única) linha resultante como um objeto de
# MAGIC onde dá pra ler cada coluna (`exec4["wr_all"]`, `exec4["wr_organic"]`,
# MAGIC etc.).
# MAGIC
# MAGIC O segundo painel (`axes[1]`) segue exatamente o mesmo padrão do primeiro,
# MAGIC só que comparando três barras (todas as propostas do Executivo 4, só as
# MAGIC orgânicas dele, e a média da empresa) em vez de duas.
# MAGIC
# MAGIC ```python
# MAGIC plt.tight_layout()
# MAGIC plt.savefig("/tmp/achado1_artefato_migracao.png", bbox_inches="tight")
# MAGIC plt.show()
# MAGIC ```
# MAGIC `tight_layout()` ajusta espaçamento automaticamente pra nada ficar
# MAGIC cortado. `savefig(caminho)` salva a imagem em disco (esse PNG é o mesmo
# MAGIC que vai pro README, embutido como evidência). `show()` renderiza o
# MAGIC gráfico na saída da célula.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# --- Painel 1: lote vs orgânico, empresa toda ------------------------------
channel_labels = by_channel["is_bulk_load"].map({True: "Carga em lote", False: "Orgânico"})
axes[0].bar(channel_labels, by_channel["win_rate_by_count"],
            color=["#c0c0c0", "#2b6cb0"])
axes[0].set_title("Taxa de vitória: lote vs orgânico")
axes[0].set_ylabel("Taxa de vitória")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
for i, v in enumerate(by_channel["win_rate_by_count"]):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center")

# --- Painel 2: Executivo 4, todas as propostas vs somente orgânicas -------
exec4 = by_executive[by_executive["account_executive"] == "Executive 4"].iloc[0]
company_avg = overall["win_rate_by_count"].iloc[0]

bars = axes[1].bar(
    ["Todas as\npropostas", "Somente\norgânicas", "Média da\nempresa"],
    [exec4["wr_all"], exec4["wr_organic"], company_avg],
    color=["#c0392b", "#2b6cb0", "#888888"],
)
axes[1].set_title("Executivo 4: pior desempenho, ou artefato?")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
for bar, v in zip(bars, [exec4["wr_all"], exec4["wr_organic"], company_avg]):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")

plt.tight_layout()
plt.savefig("/tmp/achado1_artefato_migracao.png", bbox_inches="tight")
plt.show()

print(f"Executivo 4 — todas as propostas: {exec4['wr_all']:.1f}%   somente orgânicas: {exec4['wr_organic']:.1f}%   "
      f"gap: {exec4['artefact_gap']:.1f}pp")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ressalva que este gráfico não mostra
# MAGIC
# MAGIC O lote de carga em massa foi concentrado em Financial Services / Public
# MAGIC Sector — os dois segmentos de menor taxa de vitória de base. Remover a
# MAGIC carga em lote move o Executivo 4 de pior desempenho pra acima da média,
# MAGIC mas parte do que sobra ainda pode ser mix de segmento, não um sinal
# MAGIC orgânico limpo. A seção seguinte testa isso diretamente com uma regressão
# MAGIC que trava segmento, estado e tamanho do negócio como constantes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Controlando o confound: o efeito sobrevive?
# MAGIC
# MAGIC ### O que é um "confound", em uma frase
# MAGIC
# MAGIC Um confound é uma terceira variável que influencia tanto a coisa que você
# MAGIC está medindo quanto o resultado, criando uma correlação que parece causal
# MAGIC mas não é. Aqui: o gerador de dados concentrou os registros em lote em
# MAGIC Financial Services, Public Sector, e na carteira do Executivo 4 —
# MAGIC segmentos que convertem abaixo da média **independente** de como a
# MAGIC proposta foi registrada. Então o gap visto no gráfico acima poderia ser
# MAGIC parcialmente (ou inteiramente) mix de segmento vestido de artefato de
# MAGIC migração, não o efeito da carga em lote em si.
# MAGIC
# MAGIC ### O que é uma regressão logística, em uma frase
# MAGIC
# MAGIC É um modelo estatístico que estima a probabilidade de um resultado
# MAGIC binário (aqui: ganhar ou perder uma proposta) a partir de várias
# MAGIC variáveis ao mesmo tempo — e o que ele devolve pra cada variável é o
# MAGIC quanto ela move essa probabilidade **mantendo todas as outras
# MAGIC constantes**. Isso é exatamente o que "controlar por um confound"
# MAGIC significa na prática: perguntar "depois de levar em conta segmento,
# MAGIC estado e valor do negócio, o fato de ter sido carregado em lote ainda
# MAGIC muda a chance de vitória?"

# COMMAND ----------

# MAGIC %pip install statsmodels -q

# COMMAND ----------

# MAGIC %md
# MAGIC Se a importação abaixo falhar com um erro de módulo desatualizado, rode
# MAGIC `dbutils.library.restartPython()` numa nova célula e execute o notebook de
# MAGIC novo do início — esta é a primeira célula que importa `statsmodels`, então
# MAGIC normalmente não precisa de restart, mas o cache de ambiente do Databricks é
# MAGIC inconsistente sobre isso.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Montando o dado pro modelo
# MAGIC
# MAGIC ```python
# MAGIC reg_pdf = point_in_time_join(bids_clean, clients_clean, client_cols=("segment", "state")) \
# MAGIC     .filter("outcome IS NOT NULL") \
# MAGIC     .select("outcome", "is_bulk_load", "segment", "state", "contract_value_brl") \
# MAGIC     .toPandas()
# MAGIC ```
# MAGIC Diferente do resto deste notebook, que só lê tabelas Gold já prontas,
# MAGIC aqui é o único lugar que precisa de dado **no nível de linha** (uma
# MAGIC proposta por linha, não agregado) — porque uma regressão precisa de
# MAGIC observações individuais pra estimar os coeficientes, não de resumos. Por
# MAGIC isso ele volta pra `silver.bid.bids_clean` e refaz o mesmo join
# MAGIC point-in-time do `03_gold_bid_performance` — de novo, um join simples
# MAGIC por `client_id` faria fan-out das propostas de todo cliente renovado e
# MAGIC contaria em dobro no modelo, o que enviesaria o resultado.
# MAGIC `.filter("outcome IS NOT NULL")` — repare que aqui a condição é uma
# MAGIC **string SQL**, não uma expressão `F.col(...)`. Spark aceita as duas
# MAGIC formas em `.filter()`; são equivalentes, é só um jeito diferente de
# MAGIC escrever a mesma coisa.
# MAGIC
# MAGIC ```python
# MAGIC reg_pdf["log_value"] = np.log(reg_pdf["contract_value_brl"])
# MAGIC ```
# MAGIC Cria uma coluna nova num DataFrame pandas atribuindo direto
# MAGIC (`df["nome"] = ...`, diferente do `.withColumn(...)` do Spark). Por que
# MAGIC log? O valor de contrato tem distribuição bem assimétrica (a maioria dos
# MAGIC negócios é pequena, um punhado é gigante) — sem o log, esses poucos
# MAGIC valores enormes dominariam o ajuste do modelo. `np.log` comprime essa
# MAGIC escala. Repare também que aqui usamos o valor contínuo, não as faixas de
# MAGIC quartil do Gold — uma regressão não precisa (e não deveria) discretizar
# MAGIC um número que já é contínuo.

# COMMAND ----------

import numpy as np
import statsmodels.formula.api as smf
from pyspark.sql import functions as F; import sys; sys.path.append("/Workspace/Users/samuelsouzadias@outlook.com/bid-win-loss-analytics/en"); from transforms import point_in_time_join

# Dado no nível de linha, não os agregados do Gold — este é o único lugar
# do notebook que precisa disso. Mesmo join point-in-time do `fact` do
# 03_gold_bid_performance (clients_clean é SCD Type 2 — um join simples por
# client_id faria fan-out das propostas de todo cliente renovado e
# contaria em dobro no modelo).
bids_clean = spark.table("silver.bid.bids_clean")
clients_clean = spark.table("silver.bid.clients_clean")

reg_pdf = point_in_time_join(bids_clean, clients_clean, client_cols=("segment", "state")).filter("outcome IS NOT NULL").select("outcome", "is_bulk_load", "segment", "state", "contract_value_brl").toPandas()

# Valor contínuo em log em vez das faixas de quartil da camada Gold — uma
# regressão não precisa de binning, e o log evita que a distribuição de
# valor, assimétrica à direita, domine o ajuste.
reg_pdf["log_value"] = np.log(reg_pdf["contract_value_brl"])

print(f"linhas entrando no modelo: {len(reg_pdf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Por que segment e state estão no modelo, e account_executive não:**
# MAGIC segment e state são onde o lote de carga em massa foi concentrado — são o
# MAGIC confound que queremos controlar. `account_executive` é quase a mesma
# MAGIC variável que `is_bulk_load` aqui (os lotes ficam quase inteiramente na
# MAGIC carteira do Executivo 4 por construção), então incluí-lo absorveria o
# MAGIC próprio efeito que este modelo está tentando medir, em vez de controlar
# MAGIC por um confound separado — um erro comum em regressão chamado
# MAGIC "controlar por uma variável colinear ao tratamento".

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rodando o modelo
# MAGIC
# MAGIC ```python
# MAGIC model = smf.logit(
# MAGIC     "outcome ~ C(is_bulk_load) + C(segment) + C(state) + log_value",
# MAGIC     data=reg_pdf,
# MAGIC ).fit(disp=False)
# MAGIC ```
# MAGIC `smf.logit(fórmula, data=...)` usa a **API de fórmulas** do statsmodels —
# MAGIC parecida com a notação de fórmulas do R. `"outcome ~ ..."` significa
# MAGIC "modele `outcome` em função de...": `C(is_bulk_load)` e `C(segment)`
# MAGIC tratam essas colunas como **categóricas** (o `C(...)` gera automaticamente
# MAGIC uma variável binária pra cada categoria, em vez de tratar como número
# MAGIC contínuo — essencial pra `segment`, que é texto), enquanto `log_value`
# MAGIC entra como número contínuo direto. `.fit(disp=False)` roda o ajuste do
# MAGIC modelo (`disp=False` só silencia o log de progresso da otimização).
# MAGIC
# MAGIC ```python
# MAGIC BULK_TERM = "C(is_bulk_load)[T.True]"
# MAGIC coef = model.params[BULK_TERM]
# MAGIC ci_lo, ci_hi = model.conf_int().loc[BULK_TERM]
# MAGIC pval = model.pvalues[BULK_TERM]
# MAGIC ```
# MAGIC Quando você usa `C(is_bulk_load)`, o statsmodels cria automaticamente um
# MAGIC nome de coeficiente como `"C(is_bulk_load)[T.True]"` — o coeficiente que
# MAGIC representa "ser `True`, comparado à categoria de referência (`False`)".
# MAGIC `model.params[...]` pega o valor desse coeficiente; `model.conf_int()`
# MAGIC devolve o intervalo de confiança de 95% pra cada coeficiente (aqui
# MAGIC filtrado só pro nosso termo com `.loc[...]`); `model.pvalues[...]` pega o
# MAGIC p-valor — quão improvável seria ver esse efeito por puro acaso, se na
# MAGIC verdade não houvesse efeito nenhum.
# MAGIC
# MAGIC ```python
# MAGIC adjusted_or = np.exp(coef)
# MAGIC ```
# MAGIC O coeficiente de uma regressão logística vem numa escala chamada
# MAGIC "log-odds" (logaritmo da razão de chances), que não é diretamente
# MAGIC intuitiva. Aplicar `np.exp(...)` (o inverso do log) converte de volta pra
# MAGIC **odds ratio** (razão de chances): um OR de 0,20, por exemplo, significa
# MAGIC que as chances de vitória de uma proposta em lote são 20% das chances de
# MAGIC uma orgânica equivalente — mantendo segmento, estado e valor constantes.
# MAGIC Fazemos o mesmo `np.exp(...)` nos dois extremos do intervalo de
# MAGIC confiança pra reportar o OR com sua incerteza, não só o número central.
# MAGIC
# MAGIC ```python
# MAGIC p_bulk = reg_pdf.loc[reg_pdf.is_bulk_load, "outcome"].mean()
# MAGIC p_org = reg_pdf.loc[~reg_pdf.is_bulk_load, "outcome"].mean()
# MAGIC naive_or = (p_bulk / (1 - p_bulk)) / (p_org / (1 - p_org))
# MAGIC ```
# MAGIC Pra comparação, calculamos o odds ratio **sem nenhum controle** — direto
# MAGIC das taxas de vitória brutas de cada grupo. `reg_pdf.loc[condição,
# MAGIC "coluna"]` filtra linhas e já seleciona uma coluna de uma vez;
# MAGIC `.mean()` de uma coluna de 1s e 0s é a taxa de vitória daquele grupo. A
# MAGIC fórmula matemática de odds ratio é literalmente "chance de um grupo
# MAGIC dividida pela chance do outro", onde "chance" (odds) é `p / (1-p)`.

# COMMAND ----------

model = smf.logit(
    "outcome ~ C(is_bulk_load) + C(segment) + C(state) + log_value",
    data=reg_pdf,
).fit(disp=False)

BULK_TERM = "C(is_bulk_load)[T.True]"
coef = model.params[BULK_TERM]
ci_lo, ci_hi = model.conf_int().loc[BULK_TERM]
pval = model.pvalues[BULK_TERM]

adjusted_or = np.exp(coef)
print(f"Odds ratio ajustado para is_bulk_load: {adjusted_or:.2f}   "
      f"IC 95% [{np.exp(ci_lo):.2f}, {np.exp(ci_hi):.2f}]   p={pval:.1e}")

# Odds ratio ingênuo para comparação — o número que o gráfico do Achado 1
# sugere sem nenhum controle.
p_bulk = reg_pdf.loc[reg_pdf.is_bulk_load, "outcome"].mean()
p_org = reg_pdf.loc[~reg_pdf.is_bulk_load, "outcome"].mean()
naive_or = (p_bulk / (1 - p_bulk)) / (p_org / (1 - p_org))
print(f"Odds ratio sem controles:              {naive_or:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Como ler a comparação:** se o odds ratio ajustado tivesse se movido
# MAGIC fortemente em direção a 1 em relação ao ingênuo, isso significaria que o
# MAGIC mix de segment/state/valor estava fazendo a maior parte do trabalho, e a
# MAGIC história do "artefato de migração" estava superestimada. Se ele fica
# MAGIC próximo do número ingênuo (como fica: 0,20 ajustado vs 0,27 sem
# MAGIC controles), o efeito da carga em lote é real, não apenas um proxy de
# MAGIC quais segmentos calharam de ser carregados em lote — o que é o que
# MAGIC permite que o número principal do Achado 1 se sustente sem um asterisco
# MAGIC de mix de segmento.
# MAGIC
# MAGIC Isso ainda não é prova causal — é um ajuste observacional pelos
# MAGIC confounders que a base torna óbvios (segment, state, tamanho do negócio),
# MAGIC não por todo confounder que poderia existir. `model.summary()` tem a
# MAGIC tabela completa de coeficientes se você quiser ver como segment e state
# MAGIC individualmente movem a probabilidade de vitória.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Achado 2 — taxa de vitória por contagem vs por valor
# MAGIC
# MAGIC A empresa ganha contratos pequenos e perde os grandes. Contar propostas
# MAGIC favorece o desempenho; ponderar por receita não. Este gráfico segue
# MAGIC exatamente o mesmo padrão matplotlib do Achado 1 (`subplots`, `.bar(...)`,
# MAGIC `PercentFormatter`, texto em cima de cada barra), só que com um painel
# MAGIC único em vez de dois.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(by_value_band["value_quartile"].astype(str), by_value_band["win_rate_by_count"],
       color="#2b6cb0")
ax.set_title("Taxa de vitória por quartil de valor de contrato")
ax.set_xlabel("Quartil de valor (1 = menor, 4 = maior)")
ax.set_ylabel("Taxa de vitória")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
for i, v in enumerate(by_value_band["win_rate_by_count"]):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")

plt.tight_layout()
plt.savefig("/tmp/achado2_efeito_valor.png", bbox_inches="tight")
plt.show()

gap = overall["win_rate_by_count"].iloc[0] - overall["win_rate_by_value"].iloc[0]
print(f"Taxa de vitória por contagem: {overall['win_rate_by_count'].iloc[0]:.1f}%   "
      f"por valor: {overall['win_rate_by_value'].iloc[0]:.1f}%   gap: {gap:.1f}pp")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Por que acontece: motivos de perda, e o quanto pouco cobrem do quadro
# MAGIC
# MAGIC O campo de motivo resolveria se o Achado 2 é um problema de precificação.
# MAGIC Ele é preenchido pra uma fração pequena das perdas — reportado aqui como
# MAGIC um sinal, explicitamente não como uma estimativa confiável do todo.
# MAGIC
# MAGIC ```python
# MAGIC NON_PRICE_REASONS = {"Bid cancelled by client", ...}
# MAGIC price_related = loss_reasons[~loss_reasons["loss_reason"].isin(NON_PRICE_REASONS)]
# MAGIC price_share = price_related["share_pct"].sum()
# MAGIC ```
# MAGIC Em vez de tentar identificar motivos "de preço" com um regex ou busca por
# MAGIC palavra-chave (`"cost"`, `"price"`...), a lista faz o caminho inverso:
# MAGIC nomeia explicitamente os motivos que **não** são de preço. Isso é mais
# MAGIC confiável quando há só um punhado de motivos distintos — um motivo como
# MAGIC `"Financial proposal not competitive"` é claramente sobre preço, mas não
# MAGIC contém nenhuma das palavras-chave óbvias, e um regex perderia esse caso.
# MAGIC `loss_reasons["loss_reason"].isin(NON_PRICE_REASONS)` — o mesmo `.isin(...)`
# MAGIC que você viu dentro de `denull`, agora em pandas em vez de Spark (a
# MAGIC sintaxe é praticamente idêntica). `~` de novo é negação. Filtrando o
# MAGIC DataFrame pra manter só os motivos relacionados a preço, e somando a
# MAGIC coluna `share_pct` deles, chega no percentual final.

# COMMAND ----------

coverage_pct = loss_coverage["coverage_pct"].iloc[0]
print(f"Cobertura de motivo de perda: {coverage_pct:.1f}% "
      f"({loss_coverage['losses_with_reason'].iloc[0]} de {loss_coverage['losses_total'].iloc[0]} perdas)")
print(f"Atribuídas ao concorrente placeholder: {loss_coverage['placeholder_attributed'].iloc[0]}")

# Lista de exclusão explícita em vez de regex por palavra-chave: com apenas
# um punhado de motivos distintos, nomear os que não são de preço
# diretamente é mais confiável que um match por substring (ex.: "Financial
# proposal not competitive" é sobre preço mas não contém cost/price/commercial).
NON_PRICE_REASONS = {
    "Bid cancelled by client",
    "Scope did not match expectations",
    "Incumbent held established relationship",
    "Bid suspended / postponed",
    "Insufficient information from client",
    "Reason not recorded",
}
price_related = loss_reasons[~loss_reasons["loss_reason"].isin(NON_PRICE_REASONS)]
price_share = price_related["share_pct"].sum()
print(f"Fração dos motivos registrados relacionada a preço/custo: {price_share:.1f}%")

loss_reasons.sort_values("losses", ascending=False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline em aberto
# MAGIC
# MAGIC Reportado isoladamente — nunca somado na coluna de perdas, o que
# MAGIC subestimaria a taxa de vitória. `open_pipeline` já é o DataFrame pandas
# MAGIC lido no Passo 1; escrever o nome dele sozinho numa célula do Databricks
# MAGIC (sem `print` nem `display`) já é suficiente pra renderizar a tabela.

# COMMAND ----------

open_pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## O que este notebook não afirma
# MAGIC
# MAGIC Veja a seção "O que esta análise não permite afirmar" do README pra lista
# MAGIC completa — duração do ciclo de venda, a amostra não aleatória de motivo
# MAGIC de perda, valor mensal vs total do contrato, e a ausência de custo de
# MAGIC proposta. Repetir aqui seria só duplicação; o ponto é que essas
# MAGIC limitações se aplicam a todo gráfico acima, não somente ao texto ao lado
# MAGIC do qual estão.
