Archive for informations and questions

X: Colunas do dataset
Y: Se o piloto vai estar no pódio ou não. Duas classes, 0 ou 1

Dados 'train': 2021 até 2024
Dados 'test': 2024 até 2025

O modelo é cego para o nome dos pilotos, ele só vê a performance
Não sabemos como resolver a questão de 'só ter 3 no pódio'
Não devemos usar os tempos das corridas, mas talvez da qualificação
Poderíamos olhar o tempo que o piloto fez no ano passado,
mas a posição dele naquela corrida no ano passado já ajuda muito

Tem como corrigir o 'race-experience'?
Finish rate não considera +1Lap como finalizado

----------------------------------------

Adicionados no CLF.ipynb:

-Um gráfico da correlação por classe
-Uma análise sobre o Naive Bayes e como ele se encontra no nosso código, acho que finalizamos o NB desse jeito
-Histogramas das variáveis por classe e são bem diferentes, gráficos bons pro banner
-Uma visão sobre qual o melhor número de vizinhos (k) pro KNN (sem Cross Validation)
-Pesos do random forest alterados (deu overfit no test mas train melhorou muito, bizarro)

Questões:
Usamos F1-macro como parâmetro principal?
QDA não precisa fazer oversampling ou undersampling pq n1 = 18*p
Pra reduzir as variáveis totais vamos usar só PCA?

----------------------------------------

1. Proficiência na Pista ("Fator Interlagos")
Acho bom colocar isso, mas com um fato simples de multiplicação.

2. Consistência do Piloto (Desvio Padrão)
Essa já não incluiria por agora. Fui justamente contra uma variável de desvio padrão da consistência que o prof. britânico disse pra rodar o modelo mais cru.

3. Qualidade Geral do Piloto (Longo Prazo)
Pensando na vida real é uma boa. É o efeito Verstappen... sempre estaria cotado pro pódio mesmo não indo tão bem no ano.

4. O Alvo (Target Binário)
Pódio como binário já fazemos.

O que colocaria:
5. Finish rate
Adicionar " +x Lap(s) " junto de Finished como 'terminou a corrida'. X pode ir de 1 a qts for e tem o 's' no fim

6. Race experience & circuit experience:
Poderíamos rodar um data ingestion só pra pegar as experiences certas?

7. Tempo da qualificação
Só comentando, não acho que precisa, o grid position com o multiplicador do 1. já arruma isso

8. Pontos do piloto e do construtor até o momento
Acho que esse dado vai ajudar bastante. Pra featura envolveria fazer uma conta das passadas até o momento

9. weighted_avg_position & weighted_avg_points
Os weights dão um peso exponencialmente maior pra corridas mais recentes, não sei se isso não entraria no bias. Acho que não mas poderia entrar

10. Add pontos dos construtores
Pra levar em conta que o carro pode ser bom e o piloto nem tanto (acontece bastante!). Pode ser os pontos nas últimas 3 corridas pra uma equipe, daí os dois piliotos da equipe têm o msm número