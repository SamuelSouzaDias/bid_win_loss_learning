# Script de apresentação em vídeo — Bid Win/Loss Analytics
*Roteiro de fala, em português, pra gravação de um vídeo curto apresentando o projeto pra um recrutador técnico. Duração estimada: 5–6 minutos falando num ritmo natural, sem pressa.*

---

## Como usar este roteiro

- Texto normal = fala.
- `[entre colchetes]` = direção de cena — o que mostrar na tela naquele momento (não falar em voz alta).
- Cada bloco tem uma duração aproximada, só como referência pro ritmo. Não precisa decorar palavra por palavra — o importante é passar pelas mesmas ideias, na sua própria voz.

---

## 1. Abertura (0:00 – 0:25)

`[Tela: título do projeto, ou você de frente pra câmera]`

Oi, meu nome é Samuel Souza Dias. Eu trabalho com dados há cinco anos, principalmente com BI, Power BI e SQL Server — e nos últimos meses tenho me aprofundado em Data Engineering: PySpark, Databricks, arquitetura de dados em escala.

Esse vídeo é sobre um projeto de portfólio que eu construí do zero pra mostrar exatamente isso: como eu penso um pipeline de dados de ponta a ponta, e como eu lido com dado sujo de verdade — porque dado limpo não existe fora de tutorial.

---

## 2. O problema de negócio (0:25 – 1:10)

`[Tela: primeira linha do README, o número 32% vs 25%]`

O cenário é uma empresa B2B de serviços de facilities. A liderança comercial queria uma coisa simples: a taxa de vitória de propostas, aberta por segmento, região e executivo de contas, pra saber onde intervir.

Só que esse número simples acabou sendo a coisa menos útil da base inteira. E foi isso que guiou o projeto: não "que dashboard eu consigo montar", mas "o que os dados realmente estão dizendo, e o que eles parecem dizer mas não é verdade".

Achei dois problemas sérios escondidos atrás desse número — vou mostrar os dois.

---

## 3. Achado 1 — o artefato de migração (1:10 – 2:15)

`[Tela: gráfico bulk vs orgânico, e o gráfico do Executivo 4]`

O primeiro corte dos dados mostrava um executivo — vou chamar de Executivo 4 — fechando 26% das propostas dele, contra uma média da empresa de quase 32%. Numa leitura rápida, isso é o pior vendedor da empresa, numa amostra grande o suficiente pra parecer conclusivo.

Publicar isso teria sido um erro.

Quando eu investiguei mais fundo, descobri que 58% de todas as propostas da base compartilhavam o mesmo timestamp de criação com dezenas, às vezes centenas, de outras propostas. Isso não é gente cadastrando proposta — isso é migração de sistema, carga em lote.

E essas propostas em lote se comportam de um jeito completamente diferente das que foram registradas organicamente: convertem a menos da metade da taxa. Isolando esse efeito, o Executivo 4 sai de 26% e vai pra quase 46% — de pior desempenho da empresa pra bem acima da média.

`[Tela: número do odds ratio, 0,20]`

Mas eu não parei numa comparação simples. Rodei uma regressão logística controlando por segmento, estado e valor do contrato, porque a carga em lote também estava concentrada nos segmentos que já convertem pior — então parte do efeito podia ser só coincidência de mix. O resultado: mesmo controlando tudo isso, o odds ratio ajustado ficou em 0,20, muito próximo do valor sem nenhum controle. O efeito é real. Não é só um viés escondido de segmento.

---

## 4. Achado 2 — taxa por valor esconde um problema de preço (2:15 – 3:00)

`[Tela: gráfico de win rate por quartil de valor]`

O segundo achado: a empresa ganha os contratos pequenos e perde os grandes. Contando proposta por proposta, a taxa de vitória é quase 32%. Mas ponderando pelo valor de cada contrato, cai pra quase 25%.

Sete pontos percentuais de diferença — inteiramente por causa de negócios grandes sendo perdidos. E os poucos motivos de perda que existem registrados confirmam isso: 85% deles são relacionados a preço.

Isso é um achado que só aparece se você pensar em ponderar por receita, não só contar proposta. Publicar só a taxa por contagem estaria superestimando o desempenho comercial.

---

## 5. O problema por trás dos dois (3:00 – 3:30)

`[Tela: número de cobertura, 7,6%]`

E os dois achados têm uma raiz comum: só 7,6% das perdas têm motivo registrado. Os outros 92% estão atribuídos a um "concorrente" chamado Competitor 1 — que não é um concorrente de verdade, é o valor padrão que o sistema grava quando ninguém fez a apuração pós-proposta.

Essa é a minha recomendação principal pro negócio: motivo de perda devia ser obrigatório no fechamento. Hoje a empresa perde quase 780 propostas por ciclo sem saber por quê.

---

## 6. Como o pipeline foi construído (3:30 – 4:45)

`[Tela: diagrama da arquitetura medallion]`

Agora, a parte técnica — como eu cheguei nesses números.

O pipeline roda em Databricks, com PySpark e Delta Lake, seguindo arquitetura medallion: Bronze, Silver, Gold.

Bronze ingere o dado exatamente como ele chega — sem cast, sem limpeza. A fonte é uma exportação manual de Excel, então eu aceito mudança de schema em vez de derrubar o pipeline por uma coluna nova.

No Silver é onde entra o julgamento técnico. Trato string `null` literal como NULL de verdade, resolvo data sentinela, e — o ponto que eu mais quero destacar — modelo a dimensão de clientes como SCD Type 2.

`[Tela: valid_from / valid_to / is_current]`

Por quê isso importa: um cliente pode renovar contrato, e cada renovação é um novo período de vigência, não uma correção do registro anterior. Se eu simplesmente sobrescrevesse a linha do cliente a cada renovação, eu perderia histórico — e pior, eu associaria os dados atuais de um cliente a uma proposta de anos atrás, quando esses dados nem valiam ainda.

Então eu mantenho cada versão, com `valid_from` e `valid_to` explícitos, e no Gold eu junto cada proposta com a versão do cliente que estava vigente no momento exato em que aquela proposta foi criada — não a versão mais recente de hoje. Um join simples por client_id teria contado em dobro toda proposta de cliente que renovou.

No Gold eu também construí uma tabela própria de métricas de qualidade de dados — não deixando esses números espalhados em gráficos que foram feitos pra outra coisa, mas centralizados, consultáveis, prontos pra virar um alerta de monitoramento.

---

## 7. Por que isso importa pra quem está contratando (4:45 – 5:20)

`[Tela: você de novo, ou o repositório no GitHub]`

Eu fiz esse projeto pensando em três coisas que eu acho que fazem a diferença entre um portfólio júnior e um portfólio sênior: primeiro, tratamento de erro e validação de dado, não só o caminho feliz — tem assert, tem checagem de integridade referencial, tem guarda contra parsing silencioso de data. Segundo, decisão técnica documentada e justificada, não só o resultado final — por que SCD Type 2, por que `approxQuantile` em vez de `ntile`, por que controlar a regressão pelas variáveis certas. E terceiro, honestidade sobre o que os dados não permitem afirmar — eu documento explicitamente as limitações, tipo por que duração de ciclo de venda não é uma métrica confiável nessa base.

Todo o dado aqui é sintético, gerado por um script com seed fixa — então qualquer pessoa que clonar o repositório reproduz exatamente esses mesmos números.

---

## 8. Fechamento (5:20 – 5:45)

`[Tela: link do GitHub]`

O código completo, os notebooks, e o README com todos os gráficos estão no meu GitHub — o link está na descrição. O projeto também está documentado em inglês e português, notebook por notebook.

Se você chegou até aqui, obrigado por assistir — e eu adoraria conversar mais sobre como eu penso dado, arquitetura e qualidade em qualquer entrevista.

`[Fim]`

---

## Notas de produção

- **Tom:** confiante e direto, sem soar decorado. Pausas curtas depois de cada número (26%, 45%, 0,20, 7,6%) ajudam o espectador a processar antes de você seguir.
- **Ritmo:** as seções 3 e 4 (os dois achados) são o coração do vídeo — não corra nelas. As seções 6 e 7 podem ser cortadas ou resumidas se o vídeo precisar ficar mais curto (ex.: pra um post de LinkedIn de 90 segundos, use só as seções 1, 3 resumida, e 8).
- **Corte para vídeo curto (60–90s):** abertura (encurtada) → Achado 1 (só o gráfico e o número final, sem a parte da regressão) → uma frase de fechamento com o link.
