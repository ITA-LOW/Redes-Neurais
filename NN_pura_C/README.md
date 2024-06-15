# Rede Neural pura em C
- Definição das Estruturas: As estruturas layer e network definem os componentes básicos de uma rede neural, como camadas, neurônios, funções de ativação, pesos, etc.
- Função randr: Gera um número aleatório dentro de um intervalo específico.
- Função layer_new: Cria e inicializa uma nova camada com pesos aleatórios usando a inicialização de Xavier.
- Função layer_free: Libera a memória alocada para uma camada.
- Função layer_processing: Processa a entrada através de uma camada aplicando a função de ativação.
- Função network_free: Libera a memória alocada para a rede neural.
- Função network_new: Cria e inicializa uma nova rede neural.
- Função network_predict: Faz a predição com a rede neural processando a entrada através das camadas.
- Função network_backpropagation: Realiza o backpropagation para ajustar os pesos da rede neural.
- Função network_train: Treina a rede neural ajustando os pesos através de múltiplas iterações (épocas) até que o erro mínimo seja alcançado.
- Funções de Ativação e Custo: sigmoid, sigmoid_derivative, e mse são funções auxiliares para a ativação dos neurônios e cálculo do erro.
- Função main: Inicializa os dados de treinamento, cria e treina a rede neural, faz predições, e libera a memória alocada. 