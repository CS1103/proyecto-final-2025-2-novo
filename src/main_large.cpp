#include <iostream>
#include <cmath>
#include <chrono>
#include "Network.h"
#include "layers/Dense.h"
#include "layers/Activation.h"

int main() {
    std::cout << "=== Demo de Entrenamiento de Red Neuronal a Gran Escala ===" << std::endl;
    std::cout << "Este demo entrena con un dataset sintético más grande para demostrar" << std::endl;
    std::cout << "el rendimiento en problemas más realistas (~2-3 minutos de entrenamiento)" << std::endl;
    std::cout << std::endl;

    // Crear un dataset sintético más grande: clasificación de ondas seno/coseno
    // Tarea: Clasificar si sin(x1) + cos(x2) > 0
    const int num_samples = 1000;
    const int input_dim = 10;
    const int output_dim = 1;

    std::cout << "Generando dataset sintético..." << std::endl;
    std::cout << "Muestras: " << num_samples << std::endl;
    std::cout << "Dimensiones de entrada: " << input_dim << std::endl;
    std::cout << "Dimensiones de salida: " << output_dim << std::endl;
    std::cout << std::endl;

    Matrix X(num_samples, input_dim);
    Matrix Y(num_samples, output_dim);

    // Generar datos sintéticos
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-3.14159, 3.14159);

    for (int i = 0; i < num_samples; ++i) {
        double sum = 0;
        for (int j = 0; j < input_dim; ++j) {
            X.data[i][j] = dis(gen);
            // Función compleja: suma ponderada de seno y coseno
            if (j % 2 == 0) {
                sum += std::sin(X.data[i][j]) * (j + 1) / input_dim;
            } else {
                sum += std::cos(X.data[i][j]) * (j + 1) / input_dim;
            }
        }
        // Clasificación binaria: ¿la suma es positiva?
        Y.data[i][0] = (sum > 0) ? 1.0 : 0.0;
    }

    std::cout << "¡Dataset generado exitosamente!" << std::endl;
    std::cout << std::endl;

    // Crear una red más profunda: 10 -> 50 -> 30 -> 10 -> 1
    std::cout << "Construyendo arquitectura de red neuronal..." << std::endl;
    std::cout << "Arquitectura: " << input_dim << " -> 50 -> 30 -> 10 -> " << output_dim << std::endl;
    std::cout << std::endl;

    Network net;
    
    // Capa 1: Entrada a primera capa oculta
    net.add(new Dense(input_dim, 50));
    net.add(new Tanh());
    
    // Capa 2: Primera a segunda capa oculta
    net.add(new Dense(50, 30));
    net.add(new Tanh());
    
    // Capa 3: Segunda a tercera capa oculta
    net.add(new Dense(30, 10));
    net.add(new Tanh());
    
    // Capa 4: Tercera capa oculta a salida
    net.add(new Dense(10, output_dim));
    net.add(new Sigmoid());

    std::cout << "¡Red construida exitosamente!" << std::endl;
    std::cout << "Total de capas: 8 (4 Dense + 4 Activation)" << std::endl;
    std::cout << std::endl;

    // Parámetros de entrenamiento
    const int epochs = 1000;
    const double learning_rate = 0.01;

    std::cout << "Parámetros de entrenamiento:" << std::endl;
    std::cout << "Épocas: " << epochs << std::endl;
    std::cout << "Tasa de aprendizaje: " << learning_rate << std::endl;
    std::cout << std::endl;

    std::cout << "Iniciando entrenamiento..." << std::endl;
    std::cout << "(Esto tomará aproximadamente 2-3 minutos)" << std::endl;
    std::cout << std::endl;

    // Medir tiempo de entrenamiento
    auto start_time = std::chrono::high_resolution_clock::now();

    // Entrenar
    net.train(X, Y, epochs, learning_rate);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << std::endl;
    std::cout << "¡Entrenamiento completado!" << std::endl;
    std::cout << "Tiempo total de entrenamiento: " << duration.count() << " segundos";
    std::cout << " (" << duration.count() / 60 << "m " << duration.count() % 60 << "s)" << std::endl;
    std::cout << std::endl;

    // Probar con un subconjunto de los datos
    std::cout << "Probando con las primeras 10 muestras..." << std::endl;
    std::cout << std::endl;

    Matrix X_test(10, input_dim);
    Matrix Y_test(10, output_dim);
    
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            X_test.data[i][j] = X.data[i][j];
        }
        Y_test.data[i][0] = Y.data[i][0];
    }

    Matrix predictions = net.predict(X_test);

    // Calcular métricas de precisión
    int correct = 0;
    double total_error = 0.0;

    std::cout << "--- Resultados de Prueba (Primeras 10 Muestras) ---" << std::endl;
    for (int i = 0; i < 10; ++i) {
        double expected = Y_test.data[i][0];
        double predicted = predictions.data[i][0];
        double error = std::abs(expected - predicted);

        // Clasificación binaria (umbral = 0.5)
        int predicted_class = (predicted >= 0.5) ? 1 : 0;
        int expected_class = (expected >= 0.5) ? 1 : 0;

        if (predicted_class == expected_class) {
            correct++;
        }

        total_error += error;

        std::cout << "Muestra " << (i + 1) << ": "
                  << "Esperado=" << expected << ", Predicho=" << predicted
                  << ", Error=" << error 
                  << " [" << (predicted_class == expected_class ? "✓" : "✗") << "]"
                  << std::endl;
    }

    std::cout << std::endl;

    // Calcular precisión final en conjunto de prueba
    double binary_precision = (double)correct / 10 * 100.0;
    double mae = total_error / 10;
    double numerical_precision = (1.0 - mae) * 100.0;

    std::cout << "=== PRECISIÓN FINAL (Conjunto de Prueba) ===" << std::endl;
    std::cout << "Precisión de Clasificación Binaria: " << binary_precision << "% ("
              << correct << "/10 correctas)" << std::endl;
    std::cout << "Error Absoluto Medio (MAE): " << mae << std::endl;
    std::cout << "Precisión Numérica: " << numerical_precision << "%" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << std::endl;

    // Evaluación en dataset completo
    std::cout << "Evaluando en dataset completo (" << num_samples << " muestras)..." << std::endl;
    Matrix full_predictions = net.predict(X);
    
    int full_correct = 0;
    double full_total_error = 0.0;
    
    for (int i = 0; i < num_samples; ++i) {
        double expected = Y.data[i][0];
        double predicted = full_predictions.data[i][0];
        double error = std::abs(expected - predicted);
        
        int predicted_class = (predicted >= 0.5) ? 1 : 0;
        int expected_class = (expected >= 0.5) ? 1 : 0;
        
        if (predicted_class == expected_class) {
            full_correct++;
        }
        
        full_total_error += error;
    }
    
    double full_binary_precision = (double)full_correct / num_samples * 100.0;
    double full_mae = full_total_error / num_samples;
    double full_numerical_precision = (1.0 - full_mae) * 100.0;
    
    std::cout << std::endl;
    std::cout << "=== PRECISIÓN FINAL (Dataset Completo) ===" << std::endl;
    std::cout << "Precisión de Clasificación Binaria: " << full_binary_precision << "% ("
              << full_correct << "/" << num_samples << " correctas)" << std::endl;
    std::cout << "Error Absoluto Medio (MAE): " << full_mae << std::endl;
    std::cout << "Precisión Numérica: " << full_numerical_precision << "%" << std::endl;
    std::cout << "===========================================" << std::endl;

    return 0;
}
