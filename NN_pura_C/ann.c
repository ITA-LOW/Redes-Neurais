#include <math.h>    // Biblioteca para funções matemáticas
#include <stdlib.h>  // Biblioteca para alocação de memória e outras funções de utilidade
#include <stdio.h>   // Biblioteca para entrada e saída padrão
#include <unistd.h>  // Biblioteca para manipulação de operações de sistema
#include <time.h>    // Biblioteca para funções relacionadas ao tempo

/*
    Rede neural pura em C com back propagation
*/

// Definição da estrutura de uma camada na rede neural
typedef struct __layer {
    double **W;                      // Matriz de pesos
    double *delta;                   // Gradiente de erro
    double *output;                  // Saída da camada

    int n_neurons_curr;              // Número de neurônios na camada atual
    int n_neurons_prev;              // Número de neurônios na camada anterior

    double (*activation)(double);    // Função de ativação
    double (*activation_derivative)(double); // Derivada da função de ativação
} layer;

// Definição da estrutura da rede neural
typedef struct __network {
    layer **layers;                  // Array de camadas
    int n_layers;                    // Número de camadas

    double lrate;                    // Taxa de aprendizado
    double (*cost_loss)(double *, double *, int); // Função de custo
} network;

// Gera um número aleatório entre a e b
double randr(double a, double b) {
    return ((rand() % 10000000) / 10000000.0) * (b - a) + a;
}

// Cria uma nova camada na rede neural
layer *layer_new(int n_neurons_curr, int n_neurons_prev, double (activation)(double), double (activation_derivative)(double)) {
    layer *L = (layer *)malloc(sizeof(layer));

    L->n_neurons_curr = n_neurons_curr; // Define o número de neurônios na camada atual
    L->n_neurons_prev = n_neurons_prev; // Define o número de neurônios na camada anterior
    L->activation = activation; // Define a função de ativação
    L->activation_derivative = activation_derivative; // Define a derivada da função de ativação
    L->delta = (double *)malloc(sizeof(double) * n_neurons_curr); // Aloca memória para o gradiente de erro
    L->output = (double *)malloc(sizeof(double) * n_neurons_curr); // Aloca memória para a saída

    L->W = (double **)malloc(sizeof(double *) * n_neurons_curr); // Aloca memória para a matriz de pesos

    for (int i = 0; i < n_neurons_curr; i++) {
        L->W[i] = (double *)malloc(sizeof(double) * (n_neurons_prev + 1)); // +1 para o bias
    }

    // Inicialização dos pesos usando a inicialização de Xavier
    double limit = sqrt(6.0 / (n_neurons_prev + n_neurons_curr));
    for (int i = 0; i < n_neurons_curr; i++) {
        for (int j = 0; j < n_neurons_prev + 1; j++) {
            L->W[i][j] = randr(-limit, limit);
        }
    }
    return L;
}

// Libera a memória alocada para uma camada
void layer_free(layer *L) {
    for (int i = 0; i < L->n_neurons_curr; i++) {
        free(L->W[i]);
    }

    free(L->W);
    free(L->output);
    free(L->delta);
    free(L);
}

// Processa a entrada através de uma camada
void layer_processing(double *X, layer *L) {
    for (int i = 0; i < L->n_neurons_curr; i++) {
        double s = 0;
        for (int j = 0; j < L->n_neurons_prev; j++) {
            s += L->W[i][j] * X[j];
        }
        s += L->W[i][L->n_neurons_prev]; // Soma do bias
        L->output[i] = L->activation(s); // Aplica a função de ativação
    }
}

// Libera a memória alocada para a rede neural
void network_free(network *n) {
    for (int i = 0; i < n->n_layers - 1; i++) {
        layer_free(n->layers[i]);
    }
    free(n->layers);
    free(n);
}

// Cria uma nova rede neural
network *network_new(int n_neurons_input, int n_neurons_hide, int n_neurons_output, double lrate, double (activation)(double), double (activation_derivative)(double), double (cost_loss)(double *, double *, int)) {
    network *n = (network *)malloc(sizeof(network));
    n->lrate = lrate; // Define a taxa de aprendizado
    n->n_layers = 3; // Define o número de camadas (entrada, oculta, saída)
    n->layers = (layer **)malloc(sizeof(layer *) * (n->n_layers - 1)); // Aloca memória para as camadas
    n->layers[0] = layer_new(n_neurons_hide, n_neurons_input, activation, activation_derivative); // Cria a camada oculta
    n->layers[1] = layer_new(n_neurons_output, n_neurons_hide, activation, activation_derivative); // Cria a camada de saída
    n->cost_loss = cost_loss; // Define a função de custo
    return n;
}

// Faz a predição com a rede neural
void network_predict(double *input, network *n) {
    layer_processing(input, n->layers[0]); // Processa a entrada pela primeira camada (oculta)
    layer_processing(n->layers[0]->output, n->layers[1]); // Processa a saída da primeira camada pela segunda camada (saída)
}

// Realiza o backpropagation e atualiza os pesos
double network_backpropagation(double **X, double **Y, int train_size, network *n) {
    double total_loss = 0;
    for (int i = 0; i < train_size; ++i) {
        network_predict(X[i], n); // Forward propagation

        // Calcula o MSE
        total_loss += n->cost_loss(Y[i], n->layers[1]->output, n->layers[1]->n_neurons_curr);

        // Backpropagation
        int n_layer = (n->n_layers - 2);
        for (int j = 0; j < n->layers[n_layer]->n_neurons_curr; j++) {
            n->layers[n_layer]->delta[j] = (n->layers[n_layer]->output[j] - Y[i][j]) * n->layers[n_layer]->activation_derivative(n->layers[n_layer]->output[j]);
        }

        for (int l = (n->n_layers - 2); l >= 1; l--) {
            for (int j = 0; j < n->layers[l]->n_neurons_prev; j++) {
                double error = 0.0;
                for (int k = 0; k < n->layers[l]->n_neurons_curr; k++) {
                    error += n->layers[l]->W[k][j] * n->layers[l]->delta[k];
                }
                n->layers[l - 1]->delta[j] = error * n->layers[l - 1]->activation_derivative(n->layers[l - 1]->output[j]);
            }
        }

        for (int l = (n->n_layers - 2); l >= 0; l--) {
            double *X_out_prev = l == 0 ? X[i] : n->layers[l - 1]->output;
            for (int j = 0; j < n->layers[l]->n_neurons_curr; j++) {
                double lrate_times = n->lrate * n->layers[l]->delta[j];
                for (int k = 0; k < n->layers[l]->n_neurons_prev; k++) {
                    n->layers[l]->W[j][k] -= lrate_times * X_out_prev[k];
                }
                n->layers[l]->W[j][n->layers[l]->n_neurons_prev] -= lrate_times; // Atualização do bias
            }
        }
    }
    return total_loss / train_size; // MSE por amostra
}

// Treina a rede neural
void network_train(double **X, double **Y, int train_size, network *n, int max_epochs, double min_error) {
    double error = min_error + 1;
    int i = 0;
    while (i < max_epochs && error > min_error) {
        error = network_backpropagation(X, Y, train_size, n); // Executa o backpropagation
        if (i % 1000 == 0) { // Relatório de erro a cada 1000 épocas
            printf("[%d] MSE %lf\n", i, error);
        }
        i++;
    }
}

// Função de ativação sigmoide
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivada da função de ativação sigmoide (usando a saída do neurônio)
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Função de custo MSE (Mean Squared Error)
double mse(double *x, double *y, int size) {
    double s = 0.0;
    for (int i = 0; i < size; i++) {
        s += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return s / size;
}

int main() {
    int ni = 3; // Neurônios na camada de entrada
    int nh = 3; // Neurônios na camada oculta
    int no = 1; // Neurônios na camada de saída

    double lrate = 0.1; // Taxa de aprendizado ajustada
    int train_size = 8; // Tamanho do conjunto de treinamento

    double **X = (double **)malloc(sizeof(double *) * train_size); // Alocação de memória para o conjunto de treinamento
    double **Y = (double **)malloc(sizeof(double *) * train_size); // Alocação de memória para as saídas esperadas

    for (int i = 0; i < train_size; i++) {
        X[i] = (double *)malloc(sizeof(double) * ni);
        Y[i] = (double *)malloc(sizeof(double) * no);
    }

    // Inicialização do conjunto de treinamento (XOR)
    X[0][0] = 0; X[0][1] = 0; X[0][2] = 0; Y[0][0] = 0;
    X[1][0] = 0; X[1][1] = 0; X[1][2] = 1; Y[1][0] = 1;
    X[2][0] = 0; X[2][1] = 1; X[2][2] = 0; Y[2][0] = 1;
    X[3][0] = 0; X[3][1] = 1; X[3][2] = 1; Y[3][0] = 0;
    X[4][0] = 1; X[4][1] = 0; X[4][2] = 0; Y[4][0] = 1;
    X[5][0] = 1; X[5][1] = 0; X[5][2] = 1; Y[5][0] = 0;
    X[6][0] = 1; X[6][1] = 1; X[6][2] = 0; Y[6][0] = 0;
    X[7][0] = 1; X[7][1] = 1; X[7][2] = 1; Y[7][0] = 1;

    network *net = network_new(ni, nh, no, lrate, sigmoid, sigmoid_derivative, mse); // Cria a rede neural

    network_train(X, Y, train_size, net, 100000, 1e-2); // Treina a rede neural

    for (int i = 0; i < train_size; i++) {
        network_predict(X[i], net); // Prediz a saída para cada entrada no conjunto de treinamento
        printf("ann output: %lf XOR: %lf\n", net->layers[1]->output[0], Y[i][0]); // Imprime a saída predita e a esperada
    }

    network_free(net); // Libera a memória alocada para a rede neural

    for (int i = 0; i < train_size; i++) {
        free(X[i]);
        free(Y[i]);
    }
    free(X);
    free(Y);

    return 0;
}
