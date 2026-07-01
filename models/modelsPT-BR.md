# Possibilidades de Modelagem e Análise de Desempenho
## As simulações de Regressão Linear Simples e comparações com Exponencial e afins foram feitas a partir do modelo Claude Sonnet 4.6 em 29/06/2026


### Modelo B (4.2) — A curva SoH(EFC) — target real, dado de laboratório

Esse é o modelo com target **real e medido** (a degradação das células). O que já temos (reta `SoH = 1 + k·EFC`) é o baseline. As opções para melhorar:

**Opção B1 — Linear (padrão)**

`SoH = 1 + k·EFC`

Simples, interpretável, mas assume degradação constante para sempre — na prática baterias degradam mais rápido no início e no fim (curva em "S" invertido).

**Opção B2 — Exponencial**

`SoH = e^(k·EFC)`

Naturalmente limitada entre 0 e 1, matematicamente mais coerente com processos de degradação química. Não muda muito nos EFC baixos, mas diverge menos no longo prazo. Ainda uma curva por parâmetro só.

**Opção B3 — Regressão com charge_rate como feature**

`SoH = f(EFC, charge_rate)`

Adiciona o efeito da taxa de carga como covariável, em vez de ajustar curvas separadas por grupo. Com ~94 pontos, isso é viável — mas com 4 grupos de taxa e poucos pontos cada (19–30), o ganho sobre os modelos por grupo pode ser marginal.

**Opção B4 — GPR (Gaussian Process Regression)**

Não assume forma funcional — aprende a curva dos dados. Principal vantagem: gera **intervalos de incerteza naturalmente**, o que é muito útil aqui porque nossos pontos extrapolados (cycle_count_estimated=True) têm incerteza diferente dos pontos reais. Desvantagem: com ~94 pontos e uma relação que parece razoavelmente linear/exponencial, o ganho em ajuste pode não justificar a complexidade.

---

### Modelo A (4.1) — Features de rota → ? — precisamos redefinir o target

O plano original diz "regressão de EFC em função das features de rota". Mas `EFC = DoD_equivalente` (por definição) — então prever EFC a partir de features que incluem `DoD_equivalente` seria circular. Precisamos decidir o que esse modelo realmente tenta responder. Há duas interpretações úteis:

**Interpretação A1 — Diagnóstico de uso por veículo/rota**

Não é um modelo preditivo — é uma análise descritiva: quais features de condução (`C_rate_medio`, `aggressive_events_per_km`, `pct_uphill`, etc.) mais correlacionam com `DoD_equivalente` alto? Útil para entender *por que* certos veículos/rotas consomem mais EFC por viagem. Isso seria um EDA com correlações e decomposição de variância, não um modelo ML propriamente dito.

**Interpretação A2 — Prever DoD_equivalente de uma rota nova**

Dado um conjunto de features de rota (topografia, estilo de direção, auxiliaries, vento), prever quanto EFC aquela rota vai consumir — sem precisar simular o veículo. Isso teria utilidade real: planejar impacto de degradação antes de rodar a simulação. Com 105 amostras e ~8 features, cabe uma regressão linear múltipla ou uma árvore de decisão simples.


## Etapas de simulação 
Essas serão algumas das etapas até chegar a um modelo concreto e de produção. 
```
1. Modelo B2 (exponencial) — comparar com B1 (linear já feito)
↓ se R² melhorar pouco
2. Decidir se B4 (GPR) vale pelo intervalo de incerteza
↓ em paralelo
3. Interpretação A1 (análise de correlação) — custo baixo, gera insight
↓ se os resultados forem interessantes
4. Interpretação A2 (regressão features → DoD_equivalente) — modelo preditivo real
```

### B2 x B1
Plot disponível em: /data-analysis/images/b2Xb1.png

|  | B1 linear | B2 exponencial |
| --- | --- | --- |
| k | -0.000462 | -0.000481 |
| R² pooled | 0.848 | 0.860 |
| Δ R² | — | +0.0115 |
| Parâmetros | 1 (k) | 1 (k) |
| Garante SoH≥0 sempre | Não (pode ficar negativo em EFC muito alto) | Sim (assíntota em 0) |

Resultado: B2 escolhido para segunda etapa

### B2 x B4
O argumento a favor do GPR não é "ajusta melhor" (já vimos que ganha pouco), é "captura onde a incerteza é maior". O objetivo é comparar a banda de confiança do GPR nos pontos reais vs. nos pontos que extrapolamos na Fase 1 (cycle_count_estimated=True).

O desvio padrão do GPR varia de 0.0075 a 0.0083 — uma faixa de variação de apenas ~10% relativa, e essa variação só aparece de fato perto de EFC=210-230 (a borda extrema dos dados, fora até do range observado de 221). Em quase toda a faixa útil (EFC 0–200), a incerteza é praticamente constante.

Resultado: A causa fica clara no kernel otimizado: length_scale=500 — o GPR "decidiu" que a curva é tão suave que ela se comporta quase como ruído homogêneo ao longo de tudo. Na prática, ele convergiu para algo parecido com "uma curva suave + ruído constante", que é quase a mesma informação que já tínhamos com o ajuste exponencial simples (B2).

###  Análise de Correlação
Disponível em: /data/processed/analysis/

1. completedDistance_km (r=0.86) e duracao_h (r=0.81) são, de longe, as correlações mais fortes — e olhando o heatmap, elas correlacionam 0.98 entre si. Isso significa que não são dois sinais independentes, são a mesma informação medida de duas formas (porque a velocidade média não varia muito entre rotas). A conclusão direta: o principal determinante de quanto EFC uma viagem consome é simplesmente o tamanho da viagem 

2. analisado posteriormente: C_rate_medio = net_energy_Wh / nominal_capacity_Wh / duracao_h, e isso é exatamente DoD_equivalente / duracao_h. Ou seja, o r=0.73 que ele mostra com o target não é um insight comportamental, é quase uma tautologia matemática

3. demais redundantes: DoD_pct, net_energy_Wh, nominal_capacity_Wh, e agora C_rate_medio

4. pct_uphill (r=0.48) é o sinal mais limpo: rotas com mais subida consomem proporcionalmente mais bateria — fisicamente esperado e consistente com a correlação negativa entre pct_uphill e efficiency_mWh (-0.56) no heatmap

5. metadados: traffic_factor, occupancy, auxiliaries e wind, que são os parâmetros que a simulação varia de propósito (tráfego, ocupação, ar-condicionado, vento), têm correlação perto de zero com DoD_equivalente (entre -0.11 e 0.01)

No geral, tamanho da rota explica quase tudo (~74% da variância via r²≈0.86²); topografia (subida) adiciona um efeito real, mas menor; estilo de condução agressivo parece ter efeito, mas está confundido pela normalização por km; e os parâmetros de configuração da simulação (tráfego, ocupação, AC, vento) não mostram efeito detectável nessa análise simples

### Modelos de Aprendizado
Modelo escolhido: XGBoost 
Em especial pela profundidade de árvores e generalização proporcionadas.

A ideia principal é a de testar o modelo sem e com vehID como um one-hot encoding, visando um possível aumento de desempenho. Na ideia inicial, de interpretação de comportamento de condução, o vehID não teria tanto sentido, mas visando um aumento no ganho de R^2 no valor das predições, ele foi adotado como uma dummy feature. 

Treinamos com o LOO-CV (Leave-One-Out Cross Validation):
Para cada uma das 105 observações:
    Treina com as outras 104  (≈99% dos dados)
    Prevê a 1 que ficou fora
    Registra o erro

R² final = calculado sobre as 105 previsões "fora da amostra"

Com n=105, um split 80/20 deixaria só 21 amostras para teste — e com 5 veículos × 21 rotas, um split aleatório poderia facilmente deixar um veículo inteiro sub-representado no teste, tornando as métricas muito instáveis dependendo do seed.

O teste com vehID apresentou um aumento significativo, de cerca de 0.85 de R^2 para 0.95. Os resíduos continuam seguindo a mesma forma de aleatóridade, o que é um bom sinal, sem aparente vazamento de dados. 

O que aconteceu:
| vehID | DoD real médio | Sem dummy (pred médio) | Com dummy (pred médio) |
| --- | --- | --- | --- |
| EV10 | 0.345 | 0.275 ← subestima | 0.321 ← muito melhor |
| EV1 | 0.310 | 0.272 ← subestima | 0.285 ← melhor |
| EV4 | 0.218 | 0.266 ← superestima | 0.231 ← muito melhor |
| EV7 | 0.194 | 0.235 ← superestima | 0.203 ← muito melhor |

Sem o dummy, o XGBoost aprende uma relação única entre distância/topografia e DoD para todos os veículos misturados. Como o EV10 (eUp, pack menor) e EV4 (ID3, pack maior) fazem as mesmas rotas mas consomem frações de bateria muito diferentes, o modelo sem dummy tenta achar um "meio-termo" e erra sistematicamente dos dois lados.

Com o dummy, o XGBoost pode aprender: "para esta rota de 80km, um EV10 consome ~0.55 de fração enquanto um EV7 consome ~0.30" — ou seja, a relação distância→DoD tem uma inclinação diferente por veículo, e o dummy permite que o boosting capture isso.

Disponível em: /data/processed/models/

## Fase 5 e 6
Os outputs são o resultado final do projeto:

| Figura | O que é |
| --- | --- |
| `fig1_soh_efc_projection` | Curva exp. (Fase 1) + DoD_equivalente **real** do d-EVD (Fase 2) |
| `fig2_dod_distribution` | DoD_equivalente **real** do d-EVD (Fase 2) |
| `fig3_xgb_predicted_vs_real` | Previsões do **XGBoost** vs. DoD_equivalente real |
| `fig4_degradation_by_vehicle` | DoD_equivalente **real** acumulado + curva exp. (Fase 1) |

Disponível em: /data/processed/output/


# Conclusão
Durante o projeto, de sua concepção, conseguimos conectar o conhecimento da degradação e ciclo de vida de baterias de Ion-Litío e expandir sua escala para aplicação em baterias similares mas de maior magnitude em veículos elétricos. A chave comum foi a variável adimensional (DoD_equivalente / EFC) como conversação entre a célula de laboratório e a escala de pack de veículo real. 

A degradação é previsível a partir do perfil de condução. A curva exponencial SoH = exp(-0.000481 × EFC) ajustada nos dados de laboratório, aplicada sobre o EFC acumulado das rotas simuladas, gera previsões fisicamente coerentes: veículos com pack menor (eUp, i3) acumulam EFC mais rápido e atingem o fim de vida antes que veículos com pack maior (ID4, SUV).

O XGBoost consegue prever o consumo de EFC de uma rota sem simular o veículo completo. R²=0.955 com LOO-CV para os 5 veículos conhecidos — o que significa que, dado o perfil de uma rota (distância, topografia, metadados) e o modelo do veículo, conseguimos estimar quanto aquela rota vai degradar a bateria sem precisar rodar a simulação SUMO.

Em vista de tudo isso, não validamos contra degradação real medida, pela própria natureza dos dados disponíveis. A curva SoH(EFC) vem de células em laboratório sob um protocolo específico (60% DoD, padrão UDDS). Nunca tivemos um veículo real com histórico de uso E medição de capacidade ao longo do tempo para comparar. O pipeline é metodologicamente sólido, mas não tem ground truth de campo.