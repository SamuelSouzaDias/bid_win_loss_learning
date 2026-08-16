"""
generate_bid_data_aula.py — versão didática de generate_bid_data.py

Mesmo código, mesma lógica, mesmo resultado (com a mesma seed, os arquivos
gerados são idênticos byte a byte) do generate_bid_data.py de produção. A
diferença é só a quantidade de comentário: aqui cada bloco é explicado como
se quem está lendo nunca tivesse escrito um gerador de dados sintéticos
antes — o que é `numpy.random.Generator`, por que lognormal pra valor de
contrato, como os "cacoetes" de qualidade de dados (data sentinela, string
'null', carga em lote) são fabricados de propósito.

Este arquivo não é chamado por nenhum notebook — ele é só material de
leitura. Quem os notebooks realmente rodam é o generate_bid_data.py real.

---

Synthetic data generator for the Bid Performance Insights project.

Produces two source files that mimic a real B2B facilities-services bidding
system, including the data-quality pathologies commonly found in production
CRM exports:

  1. Bulk-loaded records sharing an identical creation timestamp
  2. A placeholder competitor value that masks missing post-mortem data
  3. A loss-reason field populated for only a small fraction of losses
  4. Redundant raw/string date columns
  5. Sentinel dates and literal 'null' strings

No real company data is used. Every value is generated.

Usage:  python generate_bid_data.py [--seed 42] [--outdir ./data/raw]
"""

import argparse
import random
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
#
# Todo "número mágico" que molda o resultado final mora aqui em cima, numa
# constante nomeada, em vez de espalhado dentro das funções lá embaixo. Isso
# não é só estilo — é o que torna possível ajustar "quero 60% de propostas
# em lote em vez de 58%" mudando uma linha, sem precisar caçar o número
# certo no meio de uma função de 80 linhas.

N_CLIENTS = 1_450
N_BIDS = 1_600

# Share of bids that arrive as bulk imports rather than organic entry.
BULK_SHARE = 0.58

# Probability a closed bid is won, by acquisition channel.
WIN_RATE_ORGANIC = 0.55
WIN_RATE_BULK = 0.255

# Share of all bids still awaiting an outcome.
OPEN_SHARE = 0.30

# Share of losses that received an actual post-mortem.
POST_MORTEM_SHARE = 0.085

PLACEHOLDER_COMPETITOR = "Competitor 1"

START = date(2024, 11, 1)
END = date(2026, 3, 31)

# Cada segmento é uma tupla de três posições: (peso relativo — usado como
# probabilidade de um cliente cair nesse segmento; multiplicador de taxa de
# vitória base — >1 converte melhor que a média, <1 converte pior; faixa de
# valor de contrato — "low"/"mid"/"high", usada mais abaixo pra sortear o
# valor). Financial Services e Public Sector têm multiplicador bem abaixo
# de 1 (0.27 e 0.60) de propósito — são os segmentos que vão parecer "ruins"
# antes da análise descobrir que boa parte disso é artefato de migração.
SEGMENTS = {
    # segment: (relative weight, baseline win-rate multiplier, value tier)
    "MANUFACTURING":        (0.13, 1.45, "high"),
    "HOSPITAL-CLINIC-LAB":  (0.10, 1.30, "high"),
    "SHOPPING CENTRE":      (0.08, 1.03, "mid"),
    "RETAIL":               (0.09, 1.03, "mid"),
    "CORPORATE PROPERTY":   (0.08, 1.03, "mid"),
    "ENERGY":               (0.04, 1.06, "high"),
    "TRANSPORT-LOGISTICS":  (0.05, 1.13, "mid"),
    "INSURANCE":            (0.04, 1.00, "mid"),
    "AUTOMOTIVE":           (0.04, 0.83, "high"),
    "PUBLIC SECTOR":        (0.13, 0.60, "low"),
    "FINANCIAL SERVICES":   (0.17, 0.27, "low"),
    "RESIDENTIAL PROPERTY": (0.05, 0.47, "low"),
}

STATES = ["SP", "PR", "RJ", "MG", "RS", "SC", "PA", "BA", "GO", "PE"]
# STATE_WEIGHTS anda lado a lado com STATES pelo índice — STATES[0]="SP" tem
# peso STATE_WEIGHTS[0]=0.34, e assim por diante. É um padrão comum neste
# arquivo: duas listas/dicionários paralelos em vez de uma estrutura mais
# complexa, porque é suficiente pro que esta função precisa fazer.
STATE_WEIGHTS = [0.34, 0.28, 0.09, 0.07, 0.06, 0.04, 0.04, 0.03, 0.03, 0.02]

CITIES = {
    "SP": ["SAO PAULO", "CAMPINAS", "GUARULHOS", "SANTO ANDRE", "SOROCABA"],
    "PR": ["CURITIBA", "LONDRINA", "MARINGA", "CASCAVEL"],
    "RJ": ["RIO DE JANEIRO", "NITEROI", "DUQUE DE CAXIAS"],
    "MG": ["BELO HORIZONTE", "UBERLANDIA", "CONTAGEM"],
    "RS": ["PORTO ALEGRE", "CAXIAS DO SUL"],
    "SC": ["FLORIANOPOLIS", "JOINVILLE"],
    "PA": ["BELEM", "ANANINDEUA"],
    "BA": ["SALVADOR", "FEIRA DE SANTANA"],
    "GO": ["GOIANIA", "ANAPOLIS"],
    "PE": ["RECIFE", "JABOATAO"],
}

# Loss reasons, weighted so that price-related causes dominate.
#
# Cada item é uma tupla (texto_do_motivo, peso). Os pesos somam
# aproximadamente 1.0 (não precisam somar exatamente, porque mais abaixo o
# código normaliza — divide cada peso pela soma de todos — antes de usar
# como probabilidade de verdade). Repare que os primeiros itens da lista já
# somam mais da metade do peso total: isso é o que garante que, quando o
# gerador sortear um motivo, a maioria das vezes vai sair algo relacionado
# a preço — a mesma característica que o README chama de "85% relacionado
# a preço/custo".
LOSS_REASONS = [
    ("Client prioritised cost reduction", 0.44),
    ("Cost structure incompatible with client budget", 0.11),
    ("Bid cancelled by client", 0.10),
    ("Competitor offered lower price", 0.08),
    ("Financial proposal not competitive", 0.08),
    ("Cost structure incompatible", 0.06),
    ("Unfavourable commercial terms", 0.05),
    ("Scope did not match expectations", 0.03),
    ("Incumbent held established relationship", 0.02),
    ("Bid suspended / postponed", 0.01),
    ("Insufficient information from client", 0.01),
    ("Reason not recorded", 0.01),
]

VALUE_TIERS = {
    # tier: (lognormal mean, sigma) for monthly contract value in BRL
    "low":  (10.95, 0.80),
    "mid":  (11.20, 0.82),
    "high": (11.45, 0.85),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def iso_utc(dt: datetime) -> str:
    """Match the raw export format: 2025-05-27T16:06:34.000+00:00"""
    # .strftime(formato) converte um objeto datetime do Python numa string,
    # seguindo os códigos do formato (%Y = ano com 4 dígitos, %m = mês com
    # 2 dígitos, etc.). O ".000+00:00" no final é texto fixo — este
    # gerador sempre grava milissegundos zerados e fuso UTC, imitando
    # exatamente o formato que o sistema de origem real produz.
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")


def excel_style_datetime(dt: datetime) -> str:
    """Match the raw export's string date column: '5/27/25 16:6' (no
    leading zeros on month/day/hour).

    strftime's '%-m' / '%-d' / '%-H' no-padding codes are a glibc/macOS
    extension and raise ValueError on Windows. Padding manually with
    strftime + str.lstrip keeps the same output cross-platform.
    """
    # .lstrip("0") remove zeros à esquerda do texto — "05" vira "5". O "or
    # '0'" depois cobre o caso em que .lstrip apaga TUDO (quando o valor
    # original era literalmente "00"), porque lstrip("0") de "00" devolve
    # uma string vazia, não "0".
    month = dt.strftime("%m").lstrip("0") or "0"
    day = dt.strftime("%d").lstrip("0") or "0"
    year = dt.strftime("%y")
    hour = dt.strftime("%H").lstrip("0") or "0"
    minute = dt.strftime("%M")
    return f"{month}/{day}/{year} {hour}:{minute}"


def random_datetime(rng, start: date, end: date) -> datetime:
    # `rng` é um gerador de números aleatórios do numpy
    # (np.random.default_rng(seed) — criado uma vez lá em main() e passado
    # adiante pra toda função que precisa sortear algo). Usar o MESMO
    # gerador, com a MESMA seed, em toda chamada é o que torna a saída
    # inteira do script determinística: rodar duas vezes com --seed 42
    # produz exatamente os mesmos arquivos, sempre.
    span = (end - start).days
    # rng.integers(baixo, alto) sorteia um inteiro aleatório no intervalo
    # [baixo, alto) — alto é exclusivo. int(...) converte o tipo numpy
    # (int64) pro int nativo do Python, mais previsível de usar com
    # timedelta.
    d = start + timedelta(days=int(rng.integers(0, span)))
    return datetime(d.year, d.month, d.day,
                    int(rng.integers(7, 20)),
                    int(rng.integers(0, 60)),
                    int(rng.integers(0, 60)))


def business_shift(dt: datetime, days: int) -> datetime:
    return dt + timedelta(days=days)


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------

def build_clients(rng) -> pd.DataFrame:
    # dict mantém a ordem de inserção em Python moderno, então
    # list(SEGMENTS) devolve os nomes de segmento na mesma ordem em que
    # foram declarados lá em cima — importante porque seg_weights (montado
    # logo abaixo) precisa estar alinhado item a item com essa lista.
    seg_names = list(SEGMENTS)
    # Pega só a primeira posição de cada tupla (o peso relativo) pra cada
    # segmento, e monta um array numpy com eles.
    seg_weights = np.array([SEGMENTS[s][0] for s in seg_names], dtype=float)
    # Normaliza: divide cada peso pela soma de todos, garantindo que a
    # soma final seja exatamente 1.0 — rng.choice(..., p=...) exige isso.
    seg_weights = seg_weights / seg_weights.sum()

    # Commercial hierarchy: directors -> managers -> coordinators, and a
    # separate account-executive population.
    #
    # List comprehensions — a forma compacta do Python de escrever "pra
    # cada i nesse intervalo, monte uma string com esse i dentro" sem um
    # loop `for` explícito de várias linhas. `f"Director {i}"` é uma
    # f-string, que insere o valor de `i` diretamente no texto.
    directors = [f"Director {i}" for i in range(1, 11)]
    managers = [f"Manager {i}" for i in range(1, 37)]
    coordinators = [f"Coordinator {i}" for i in range(1, 41)]
    executives = [f"Executive {i}" for i in range(1, 10)]

    # Executive 4 owns the portfolio that will later be bulk-imported.
    #
    # Pesos NÃO uniformes de propósito — Executive 4 (índice 3, quarto da
    # lista) tem peso bruto 0.30, bem acima dos outros — é essa
    # concentração que faz dele o "executivo com carteira migrada" mais
    # adiante, o artefato que a análise inteira do README existe pra
    # descobrir.
    exec_weights = np.array([0.09, 0.05, 0.04, 0.30, 0.07, 0.13, 0.06, 0.09, 0.17])
    exec_weights = exec_weights / exec_weights.sum()

    # rng.choice(população, size=N, replace=False) sorteia N valores
    # ÚNICOS (sem repetição, por causa de replace=False) de dentro do
    # intervalo np.arange(100, 9_999) — os IDs de cliente.
    ids = rng.choice(np.arange(100, 9_999), size=N_CLIENTS, replace=False)
    rows = []
    for cid in ids:
        # rng.choice(lista, p=pesos) sorteia UM item da lista, respeitando
        # as probabilidades em p (que precisam somar 1.0, por isso a
        # normalização lá em cima).
        segment = rng.choice(seg_names, p=seg_weights)
        state = rng.choice(STATES, p=STATE_WEIGHTS)

        # Financial-services and public-sector clients skew to Executive 4's
        # region, which is what makes the bulk-import confound believable.
        #
        # rng.random() sorteia um decimal entre 0 e 1 — comparar contra um
        # limiar (aqui, 0.62) é o jeito padrão de simular "isso acontece X%
        # das vezes" neste gerador inteiro. Aqui: clientes de Financial
        # Services ou Public Sector têm 62% de chance de cair no estado PR
        # e na carteira do Executivo 4 — é essa concentração deliberada
        # que faz o "artefato de migração" do README parecer,
        # inicialmente, mix de segmento em vez do que realmente é.
        if segment in ("FINANCIAL SERVICES", "PUBLIC SECTOR") and rng.random() < 0.62:
            state = "PR"
            executive = "Executive 4"
        else:
            executive = rng.choice(executives, p=exec_weights)

        city = rng.choice(CITIES[state])
        status = "Active" if rng.random() < 0.72 else "Inactive"

        start_dt = random_datetime(rng, date(2019, 1, 1), date(2026, 1, 1)).date()
        if status == "Active" and rng.random() < 0.28:
            end_dt = date(2999, 12, 31)          # open-ended sentinel
        else:
            end_dt = start_dt + timedelta(days=int(rng.integers(180, 2200)))

        rows.append({
            "client_id": int(cid),
            "contract_name": f"{segment.split('-')[0].strip().lower()} {rng.integers(1, 400)}",
            "status": status,
            # 4% de chance de start_date vir como o placeholder '-' em vez
            # de uma data real — um dos "cacoetes" de qualidade de dados
            # que o README documenta (é parte do porquê ~6% das propostas
            # não conseguem ser associadas a uma versão de cliente mais
            # adiante no pipeline).
            "start_date": "-" if rng.random() < 0.04 else start_dt,
            "end_date": end_dt,
            # Mesmo padrão pra state e city: 3% de chance de vir como a
            # string literal "null" em vez de um valor de verdade — é
            # exatamente o que a função denull() do transforms.py existe
            # pra corrigir no notebook 02_silver_bids.
            "state": "null" if rng.random() < 0.03 else state,
            "city": "null" if rng.random() < 0.03 else city,
            "segment": segment,
            "account_executive": executive,
            "director": rng.choice(directors),
            "manager": rng.choice(managers),
            "coordinator": rng.choice(coordinators),
        })

    df = pd.DataFrame(rows)

    # A handful of clients renew: a second contract period for the same
    # client_id, starting after the first one actually ends. Open-ended
    # contracts (the 2999-12-31 sentinel) have nothing to renew from, so
    # they're excluded from the sampling pool.
    #
    # The renewal's start_date is genuinely later than the original's
    # end_date — not a same-day duplicate — because Silver's SCD Type 2
    # dimension needs two non-overlapping validity windows to model, not
    # two identical timestamps with no ordering to break the tie on.
    #
    # df[condição] filtra linhas de um DataFrame pandas — igual você já
    # viu em transforms_aula.py, só que aqui é pandas em vez de Spark (a
    # sintaxe de filtro é praticamente a mesma ideia nas duas bibliotecas).
    renewable = df[df["end_date"] != date(2999, 12, 31)]
    # .sample(frac=0.06, random_state=...) sorteia aleatoriamente 6% das
    # linhas de renewable, sem reposição. random_state fixa a
    # aleatoriedade dessa amostragem específica — derivado do rng
    # principal (rng.integers(...)) pra continuar tudo determinístico a
    # partir de uma única --seed.
    renewals = renewable.sample(
        frac=0.06, random_state=int(rng.integers(0, 10_000))
    ).copy()

    # rng.integers(baixo, alto, size=N) sorteia um ARRAY inteiro de N
    # valores de uma vez, em vez de um valor por vez — mais eficiente que
    # chamar rng.integers(...) dentro de um loop Python.
    gap_days = rng.integers(1, 45, size=len(renewals))
    duration_days = rng.integers(180, 2200, size=len(renewals))
    # rng.random(N) < 0.28 sorteia N decimais e compara todos de uma vez
    # contra 0.28 — o resultado é um array de True/False, um por
    # renovação, decidindo se aquela renovação específica ficou em
    # aberto (contrato ainda vigente) ou fechada com uma duração definida.
    still_open = rng.random(len(renewals)) < 0.28

    # zip(lista1, lista2) percorre duas listas em paralelo, entregando um
    # par por vez — aqui, a data de fim do contrato original e o gap
    # sorteado pra aquela mesma renovação, juntos.
    new_start = [
        end + timedelta(days=int(gap))
        for end, gap in zip(renewals["end_date"], gap_days)
    ]
    renewals["start_date"] = new_start
    renewals["end_date"] = [
        date(2999, 12, 31) if open_ended else start + timedelta(days=int(dur))
        for start, dur, open_ended in zip(new_start, duration_days, still_open)
    ]
    renewals["contract_name"] = renewals["contract_name"] + " (renewal)"
    renewals["status"] = "Active"  # renewing implies still active

    # pd.concat([df1, df2]) empilha dois DataFrames um em cima do outro —
    # o resultado final de build_clients() tem as linhas originais MAIS as
    # linhas de renovação, cada uma com o mesmo client_id do original mas
    # um período de vigência diferente. ignore_index=True refaz o índice
    # do zero em vez de manter os índices originais duplicados.
    return pd.concat([df, renewals], ignore_index=True)


# --------------------------------------------------------------------------
# Bids
# --------------------------------------------------------------------------

def build_bids(rng, clients: pd.DataFrame) -> pd.DataFrame:
    # .drop_duplicates("client_id") mantém só a primeira linha de cada
    # client_id — útil aqui porque, pra sortear QUAL cliente uma proposta
    # pertence, não importa se esse cliente tem uma ou duas versões de
    # contrato (isso é problema do Silver, não deste gerador).
    # .set_index("client_id") transforma a coluna client_id no índice do
    # DataFrame, o que permite fazer unique_clients.loc[algum_id] pra
    # buscar os dados daquele cliente diretamente, sem filtrar.
    unique_clients = clients.drop_duplicates("client_id").set_index("client_id")

    n_bulk = int(N_BIDS * BULK_SHARE)
    n_organic = N_BIDS - n_bulk

    # ---- Bulk batches -----------------------------------------------------
    # Large portfolios ingested in one transaction. Sizes chosen so a few
    # dominate, mirroring what a migration actually looks like.
    #
    # Em vez de dividir n_bulk em lotes de tamanho aleatório uniforme, o
    # gerador usa uma lista de tamanhos FIXOS decrescentes (355, 220, 95...)
    # primeiro — imitando o padrão real de uma migração, onde poucos lotes
    # gigantes concentram a maior parte do volume, em vez de muitos lotes
    # médios e parecidos entre si.
    batch_sizes, remaining = [], n_bulk
    for size in (355, 220, 95, 50, 48, 44):
        if remaining <= 0:
            break
        take = min(size, remaining)
        batch_sizes.append(take)
        remaining -= take
    # Depois que os tamanhos fixos acabam, preenche o resto com lotes
    # menores e mais variados (10 a 30), até n_bulk ser totalmente
    # alocado.
    while remaining > 0:
        take = min(int(rng.integers(10, 30)), remaining)
        batch_sizes.append(take)
        remaining -= take

    # Bulk batches draw from concentrated portfolios (Executive 4 / PR /
    # financial services), which is precisely the confound to be discovered.
    #
    # `|` aqui é OU lógico entre duas condições booleanas de pandas —
    # equivalente ao `|` que você já viu em Spark, dentro de
    # point_in_time_join. unique_clients[condição].index.to_numpy() pega
    # os client_id (que virou o índice do DataFrame lá em cima) de todo
    # cliente que bate com a condição, como um array numpy.
    bulk_pool = unique_clients[
        (unique_clients["account_executive"] == "Executive 4")
        | (unique_clients["segment"].isin(["FINANCIAL SERVICES", "PUBLIC SECTOR"]))
    ].index.to_numpy()
    # Rede de segurança: se por acaso o pool concentrado for menor que
    # n_bulk (não deveria acontecer com N_CLIENTS=1450, mas é barato
    # garantir), completa com todo mundo.
    if len(bulk_pool) < n_bulk:
        bulk_pool = np.concatenate([bulk_pool, unique_clients.index.to_numpy()])

    organic_pool = unique_clients.index.to_numpy()

    records, bid_id = [], 100
    for size in batch_sizes:
        # Um timestamp por lote inteiro — TODAS as propostas deste lote
        # específico vão compartilhar exatamente este `stamp`. É essa
        # repetição de timestamp que o notebook 02_silver_bids detecta
        # depois com a window function de add_is_bulk_load().
        stamp = random_datetime(rng, START, END)
        # rng.choice(pool, size=N, replace=True) sorteia N clientes do
        # bulk_pool, PODENDO repetir (replace=True) — um mesmo cliente
        # pode aparecer várias vezes num lote, o que é realista pra uma
        # migração (o mesmo cliente tinha várias propostas históricas
        # importadas de uma vez).
        chosen = rng.choice(bulk_pool, size=size, replace=True)
        for cid in chosen:
            bid_id += 1
            records.append((bid_id, stamp, int(cid), "bulk"))

    for _ in range(n_organic):
        bid_id += 1
        # Diferente do lote, cada proposta orgânica sorteia SEU PRÓPRIO
        # timestamp individual — por isso elas não colidem em massa como
        # as de lote.
        stamp = random_datetime(rng, START, END)
        cid = int(rng.choice(organic_pool))
        records.append((bid_id, stamp, cid, "organic"))

    # Embaralha a lista de registros no lugar — sem isso, todas as
    # propostas em lote ficariam agrupadas no início do arquivo final, e
    # todas as orgânicas no final, o que não é realista (a ordem real de
    # bid_id não reflete a ordem de criação).
    rng.shuffle(records)

    # ---- Contract values --------------------------------------------------
    # Generated up front so the size effect can be expressed as a percentile.
    # Without this the segment effect swamps it, because the high-value
    # segments happen to be the easiest to win.
    values = []
    for _, _, client_id, _ in records:
        tier = SEGMENTS[unique_clients.loc[client_id, "segment"]][2]
        mu, sigma = VALUE_TIERS[tier]
        # np.exp(rng.normal(mu, sigma)) é como se sorteia um valor de uma
        # distribuição LOGNORMAL: sorteia de uma normal comum (sino,
        # simétrica) na escala logarítmica, depois desfaz o log com
        # np.exp. O resultado é uma distribuição de cauda longa à direita
        # — a maioria dos contratos é modesta, mas uns poucos são
        # gigantes — que é exatamente o formato que valor de contrato
        # tem no mundo real (e é por isso que 04_analysis usa np.log
        # antes de rodar a regressão, pra desfazer essa assimetria).
        # min(..., 4_000_000) trava um teto, pra não sortear um valor
        # absurdamente grande por acaso.
        values.append(round(float(min(np.exp(rng.normal(mu, sigma)), 4_000_000)), 2))
    # .rank(pct=True) transforma uma lista de valores nos PERCENTIS
    # relativos deles — o menor valor vira perto de 0.0, o maior perto de
    # 1.0. Isso é usado logo abaixo pra fazer contratos maiores serem
    # sistematicamente mais difíceis de ganhar, proporcionalmente ao
    # tamanho, não por um limiar fixo.
    value_pct = pd.Series(values).rank(pct=True).to_numpy()

    # ---- Outcomes ---------------------------------------------------------
    rows = []
    # enumerate(records) devolve (índice, item) a cada volta — o índice
    # `i` é usado logo abaixo pra buscar o value_pct correspondente a
    # esta mesma proposta, já que values e value_pct estão na mesma ordem
    # que records.
    for i, (bid_id, created_at, client_id, channel) in enumerate(records):
        client = unique_clients.loc[client_id]
        segment = client["segment"]
        seg_mult = SEGMENTS[segment][1]
        tier = SEGMENTS[segment][2]

        # is_confirmed_date: 0 means bid_date is a forecast, not a fact.
        confirmed = 1 if rng.random() > 0.057 else 0
        if confirmed:
            bid_date = datetime(created_at.year, created_at.month, created_at.day)
        else:
            bid_date = datetime(created_at.year, created_at.month, created_at.day) \
                       + timedelta(days=int(rng.integers(5, 45)))

        still_open = rng.random() < OPEN_SHARE
        base_rate = WIN_RATE_BULK if channel == "bulk" else WIN_RATE_ORGANIC
        # np.clip(valor, mínimo, máximo) trava um número dentro de um
        # intervalo — aqui garante que a probabilidade de vitória nunca
        # saia do intervalo [0.01, 0.95], mesmo depois de multiplicar por
        # seg_mult (que pode passar de 1, como em MANUFACTURING com 1.45).
        win_prob = float(np.clip(base_rate * seg_mult, 0.01, 0.95))

        # Contract value. Larger contracts are materially harder to win, so a
        # value-weighted win rate lands below the count-based one. That gap is
        # invisible unless the analyst thinks to weight by revenue.
        value = values[i]
        # Fórmula linear simples: quanto mais perto do topo do percentil
        # (value_pct perto de 1.0 — contrato caro), menor o size_factor
        # (perto de 1.55 - 1.10 = 0.45). Quanto mais barato (value_pct
        # perto de 0), maior o size_factor (perto de 1.55). Multiplicar
        # win_prob por esse fator é o que produz o "Achado 2" do README —
        # contratos grandes convertem sistematicamente pior.
        size_factor = 1.55 - 1.10 * float(value_pct[i])
        win_prob = float(np.clip(win_prob * size_factor, 0.01, 0.95))

        if still_open:
            outcome, closed_at = None, None
        else:
            outcome = 1 if rng.random() < win_prob else 0
            closed_at = business_shift(created_at, int(rng.integers(20, 300)))

        rows.append({
            "bid_id": bid_id,
            "_created_at": created_at,
            "_closed_at": closed_at,
            "is_confirmed_date": confirmed,
            "_bid_date": bid_date,
            "outcome": outcome,
            "client_id": client_id,
            "contract_value_brl": value,
            "_channel": channel,
        })

    df = pd.DataFrame(rows)

    # ---- Bulk closure artefact -------------------------------------------
    # Closures performed in a single sitting produce timestamps seconds apart.
    # Any cycle-time metric derived from these is meaningless.
    #
    # Esta seção fabrica a segunda pathologia mencionada no README: mais
    # da metade dos fechamentos foi registrada em sessões de encerramento
    # em massa, não no momento real em que a proposta fechou. É por isso
    # que 03_gold_bid_performance e 04_analysis nunca calculam duração de
    # ciclo de venda — o dado que permitiria isso é deliberadamente
    # fabricado como não confiável.
    closed = np.array(df[df["_closed_at"].notna()].index)
    rng.shuffle(closed)
    n_seq = int(len(closed) * 0.55)
    cursor = 0
    while cursor < n_seq:
        run = min(int(rng.integers(8, 60)), n_seq - cursor)
        batch_idx = closed[cursor:cursor + run]
        # The sitting has to postdate every bid it closes, otherwise the file
        # carries impossible records — a defect, not a realistic pathology.
        #
        # max(...) sobre uma expressão geradora — pega o created_at MAIS
        # RECENTE entre todas as propostas deste lote de fechamento, pra
        # garantir que a data de fechamento fabricada seja sempre depois
        # de toda proposta que ela está fechando (senão o dado ficaria
        # logicamente impossível: fechado antes de ter sido criado).
        latest = max(df.at[j, "_created_at"] for j in batch_idx)
        anchor = latest + timedelta(days=int(rng.integers(15, 240)),
                                    hours=int(rng.integers(0, 10)))
        for offset in range(run):
            idx = closed[cursor + offset]
            # Cada proposta deste lote de fechamento recebe o mesmo
            # "âncora" de tempo, deslocado só por alguns segundos
            # (offset * um número aleatório de 9 a 22) — exatamente o
            # padrão "fechado em sequência, segundos de intervalo" que o
            # README descreve.
            df.at[idx, "_closed_at"] = anchor + timedelta(seconds=offset * int(rng.integers(9, 22)))
        cursor += run

    # ---- Loss reasons and competitors ------------------------------------
    reasons = [r for r, _ in LOSS_REASONS]
    reason_p = np.array([p for _, p in LOSS_REASONS])
    reason_p = reason_p / reason_p.sum()

    competitors = [f"Competitor {i}" for i in range(2, 16)]
    comp_p = np.array([0.05, 0.03, 0.03, 0.10, 0.09, 0.06, 0.04,
                       0.03, 0.04, 0.05, 0.14, 0.05, 0.24, 0.05])
    comp_p = comp_p / comp_p.sum()

    loss_reason, competitor = [], []
    # df.iterrows() percorre um DataFrame pandas linha por linha,
    # devolvendo (índice, linha) — mais lento que operações vetorizadas,
    # mas aqui a lógica por linha (decidir motivo E concorrente juntos,
    # com dependência condicional entre eles) é mais simples de escrever
    # assim do que tentando vetorizar.
    for _, r in df.iterrows():
        if r["outcome"] == 0:
            if rng.random() < POST_MORTEM_SHARE:
                loss_reason.append(rng.choice(reasons, p=reason_p))
                competitor.append(rng.choice(competitors, p=comp_p))
            else:
                # No post-mortem: the system writes its default value.
                #
                # Este é o terceiro "cacoete" do README: quando ninguém
                # apurou o motivo da perda (91.5% das vezes, já que
                # POST_MORTEM_SHARE = 0.085), o sistema não deixa o campo
                # vazio — ele grava um valor PADRÃO de concorrente
                # (PLACEHOLDER_COMPETITOR = "Competitor 1"), fazendo
                # parecer que existe um concorrente específico levando a
                # maioria das perdas, quando na verdade é só "ninguém
                # apurou".
                loss_reason.append(None)
                competitor.append(PLACEHOLDER_COMPETITOR)
        else:
            loss_reason.append(None)
            competitor.append(None)

    df["loss_reason"] = loss_reason
    df["competitor_name"] = competitor

    # ---- Raw-export shaping ----------------------------------------------
    # Até aqui, df tinha colunas de trabalho internas (prefixadas com _,
    # como _created_at, _closed_at) — convenientes pra escrever o gerador,
    # mas não são o formato final que o Bronze espera. Este bloco monta o
    # DataFrame de SAÍDA de verdade, com exatamente os nomes de coluna e
    # formatos de texto que a exportação Excel real produziria.
    out = pd.DataFrame({
        "bid_id": df["bid_id"],
        # .map(funcao) aplica uma função a cada valor de uma coluna
        # pandas, um por um — o equivalente pandas do que .withColumn +
        # uma expressão faz em Spark.
        "created_at": df["_created_at"].map(iso_utc),
        "created_at_str": df["_created_at"].map(excel_style_datetime),
        "is_confirmed_date": df["is_confirmed_date"],
        "bid_date": df["_bid_date"].map(iso_utc),
        # pd.notna(x) — o inverso de "é nulo". Aqui: se _closed_at tem
        # valor, formata como data ISO; senão, grava o placeholder "-"
        # (proposta ainda aberta) — o mesmo "-" que denull() no
        # transforms.py sabe reconhecer e trocar por NULL de verdade.
        "closed_at": df["_closed_at"].map(lambda x: iso_utc(x) if pd.notna(x) else "-"),
        "closed_at_str": df["_closed_at"].map(
            lambda x: excel_style_datetime(x) if pd.notna(x) else "null"),
        # outcome vira a STRING "null" quando é NULL, e o inteiro (0 ou 1)
        # caso contrário — imitando fielmente como uma exportação Excel
        # real grava um campo numérico opcional.
        "outcome": df["outcome"].map(lambda x: "null" if pd.isna(x) else int(x)),
        # .fillna("null") substitui todo NULL de pandas pela string
        # literal "null" — o mesmo padrão de novo, o "cacoete" que o
        # Silver corrige.
        "loss_reason": df["loss_reason"].fillna("null"),
        "competitor_name": df["competitor_name"].fillna("null"),
        "client_id": df["client_id"],
        "contract_value_brl": df["contract_value_brl"],
    })
    # .sort_values("bid_id") ordena as linhas finais por bid_id (lembre
    # que records foi embaralhado lá em cima com rng.shuffle — este sort
    # devolve a ordem "natural" de um arquivo exportado, sem que isso
    # desfaça a aleatoriedade de QUAL cliente/timestamp cada bid_id
    # recebeu). .reset_index(drop=True) renumera o índice do DataFrame de
    # 0 em diante, descartando o índice antigo (bagunçado pelo sort).
    return out.sort_values("bid_id").reset_index(drop=True)


# --------------------------------------------------------------------------

def main():
    # argparse é a biblioteca padrão do Python pra ler argumentos de linha
    # de comando — o que permite rodar `python generate_bid_data.py --seed
    # 42 --outdir ./data/raw` no terminal.
    ap = argparse.ArgumentParser()
    # --seed, tipo inteiro, valor padrão 42 se a pessoa não passar nada.
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=".")
    # .parse_args() lê o que foi passado de fato na linha de comando e
    # devolve um objeto onde args.seed e args.outdir têm os valores
    # escolhidos (ou os padrões).
    args = ap.parse_args()

    # np.random.default_rng(seed) é a forma moderna e recomendada do
    # numpy de criar um gerador de números aleatórios — em vez da API
    # antiga np.random.seed(...) global, este `rng` é um OBJETO que você
    # passa explicitamente de função em função (repare que toda função
    # deste arquivo recebe `rng` como parâmetro). Isso evita um problema
    # clássico de estado global: duas partes diferentes do código
    # "brigando" pela mesma fonte de aleatoriedade sem saber uma da
    # outra.
    rng = np.random.default_rng(args.seed)
    # random.seed(...) fixa também o módulo `random` da biblioteca padrão
    # (diferente do numpy) — usado por pandas internamente em
    # .sample(random_state=...). Fixar os dois garante que TODA fonte de
    # aleatoriedade neste script seja determinística a partir de uma
    # única --seed.
    random.seed(args.seed)

    clients = build_clients(rng)
    bids = build_bids(rng, clients)

    # Grava os quatro arquivos de saída: Excel (o formato que o Bronze
    # real vai ler) e CSV (mais fácil de inspecionar rapidamente sem abrir
    # o Excel). index=False evita que pandas grave uma coluna extra com o
    # índice numérico interno do DataFrame.
    clients.to_excel(f"{args.outdir}/clients.xlsx", index=False, sheet_name="Clients")
    bids.to_excel(f"{args.outdir}/bids.xlsx", index=False, sheet_name="Bronze")
    clients.to_csv(f"{args.outdir}/clients.csv", index=False)
    bids.to_csv(f"{args.outdir}/bids.csv", index=False)

    print(f"clients: {clients.shape}   bids: {bids.shape}")


# Esse if é o padrão clássico do Python: o código dentro dele só roda
# quando o arquivo é executado DIRETAMENTE (`python generate_bid_data.py`),
# não quando ele é importado por outro script (`import generate_bid_data`).
# Sem isso, só importar o arquivo já dispararia a geração dos dados.
if __name__ == "__main__":
    main()
