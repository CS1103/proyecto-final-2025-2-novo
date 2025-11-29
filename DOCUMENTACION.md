# Red Neuronal Artificial en C++

## Descripción del Proyecto

Este proyecto implementa una **red neuronal artificial multicapa** desde cero en C++, sin utilizar bibliotecas de aprendizaje automático externas. La implementación incluye capas densas (fully connected), funciones de activación (Tanh y Sigmoid), y un algoritmo de entrenamiento basado en retropropagación (backpropagation).

El proyecto está diseñado con fines educativos para demostrar los fundamentos de las redes neuronales y el aprendizaje profundo.

---

## Características Principales

- ✅ **Implementación desde cero**: No depende de bibliotecas como TensorFlow o PyTorch
- ✅ **Clase Matrix personalizada**: Operaciones matriciales básicas implementadas manualmente
- ✅ **Capas modulares**: Arquitectura basada en clases abstractas para fácil extensión
- ✅ **Funciones de activación**: Tanh y Sigmoid implementadas
- ✅ **Retropropagación**: Algoritmo completo de backpropagation
- ✅ **Pruebas unitarias**: Suite completa de tests para validar componentes
- ✅ **Ejemplo XOR**: Demostración del problema XOR clásico

---

## Estructura del Proyecto

```
proyecto-final/
├── src/
│   ├── Matrix.h              # Clase para operaciones matriciales
│   ├── Network.h/cpp         # Clase principal de la red neuronal
│   ├── layers/
│   │   ├── Layer.h           # Clase base abstracta para capas
│   │   ├── Dense.h/cpp       # Capa densa (fully connected)
│   │   └── Activation.h      # Funciones de activación (Tanh, Sigmoid)
│   ├── losses/
│   │   └── MSE.h             # Función de pérdida (Mean Squared Error)
│   └── main.cpp              # Programa principal con ejemplo XOR
├── tests/
│   ├── test_matrix.cpp       # Tests unitarios para Matrix
│   ├── test_dense.cpp        # Tests unitarios para Dense layer
│   ├── test_activation.cpp   # Tests unitarios para activaciones
│   └── test_xor.cpp          # Test de integración con XOR
├── CMakeLists.txt            # Configuración de CMake
└── DOCUMENTACION.md          # Este archivo
```

---

## Requisitos del Sistema

### Compilador
- **GCC 11+** o **Clang 12+** o **MSVC 2019+**
- Soporte para **C++17**

### Herramientas de Compilación
- **CMake 3.18+** (recomendado)
- O compilación directa con g++/clang++

---

## Instalación y Compilación

### Opción 1: Usando CMake (Recomendado)

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd proyecto-final-2025-2-novo

# Crear directorio de compilación
mkdir build
cd build

# Configurar con CMake
cmake ..

# Compilar
cmake --build .

# Ejecutar el demo
./neural_net_demo

# Ejecutar tests
./test_matrix
./test_dense
./test_activation
./test_xor
```

### Opción 2: Compilación Directa con g++

```bash
# Compilar el programa principal
g++ src/main.cpp src/Network.cpp src/layers/Dense.cpp -o neural_net_demo -I src -std=c++17

# Ejecutar
./neural_net_demo

# Compilar tests
g++ tests/test_matrix.cpp -o test_matrix -I src -std=c++17
g++ tests/test_dense.cpp src/Network.cpp src/layers/Dense.cpp -o test_dense -I src -std=c++17
g++ tests/test_activation.cpp -o test_activation -I src -std=c++17
g++ tests/test_xor.cpp src/Network.cpp src/layers/Dense.cpp -o test_xor -I src -std=c++17
```

---

## Uso y Ejemplos

### Ejemplo Básico: Problema XOR

El archivo `src/main.cpp` contiene un ejemplo completo del problema XOR:

```cpp
#include "Network.h"
#include "layers/Dense.h"
#include "layers/Activation.h"

int main() {
    // Datos de entrenamiento XOR
    Matrix X(4, 2);
    X.data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    
    Matrix Y(4, 1);
    Y.data = {{0}, {1}, {1}, {0}};
    
    // Crear red neuronal
    Network net;
    net.add(new Dense(2, 3));    // Capa de entrada a oculta
    net.add(new Tanh());         // Activación
    net.add(new Dense(3, 1));    // Capa oculta a salida
    net.add(new Tanh());         // Activación de salida
    
    // Entrenar
    net.train(X, Y, 10000, 0.1);
    
    // Predecir
    Matrix prediction = net.predict(X);
    prediction.print();
    
    return 0;
}
```

### Crear una Red Neuronal Personalizada

```cpp
Network net;

// Capa de entrada: 4 neuronas -> 8 neuronas ocultas
net.add(new Dense(4, 8));
net.add(new Tanh());

// Capa oculta: 8 -> 4 neuronas
net.add(new Dense(8, 4));
net.add(new Sigmoid());

// Capa de salida: 4 -> 1 neurona
net.add(new Dense(4, 1));
net.add(new Sigmoid());

// Entrenar con tus datos
net.train(X_train, Y_train, epochs=5000, learning_rate=0.01);

// Hacer predicciones
Matrix predictions = net.predict(X_test);
```

---

## Arquitectura de la Solución

### 1. Clase Matrix

Implementa operaciones matriciales básicas:
- Multiplicación de matrices
- Transposición
- Suma y resta
- Multiplicación por escalar
- Producto de Hadamard (elemento por elemento)

### 2. Capas (Layers)

**Clase Base: Layer**
```cpp
class Layer {
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& output_gradient, double learning_rate) = 0;
};
```

**Dense Layer (Capa Densa)**
- Implementa una capa totalmente conectada
- Forward pass: `Y = X * W + B`
- Backward pass: Calcula gradientes para pesos, bias y entrada
- Actualiza parámetros usando descenso de gradiente

**Activation Layer (Capa de Activación)**
- Aplica funciones no lineales
- Tanh: `f(x) = tanh(x)`, `f'(x) = 1 - tanh²(x)`
- Sigmoid: `f(x) = 1/(1+e^-x)`, `f'(x) = f(x)(1-f(x))`

### 3. Network (Red Neuronal)

Gestiona múltiples capas:
- `add(Layer*)`: Añade una capa a la red
- `predict(Matrix)`: Propagación hacia adelante
- `train(X, Y, epochs, lr)`: Entrenamiento con backpropagation

### 4. Función de Pérdida

**Mean Squared Error (MSE)**
- `loss(y_true, y_pred)`: Calcula el error
- `prime(y_true, y_pred)`: Calcula el gradiente

---

## Pruebas Unitarias

### Test de Matrix
- Creación de matrices
- Multiplicación de matrices
- Transposición
- Operaciones aritméticas
- Producto de Hadamard

### Test de Dense Layer
- Forward pass con valores conocidos
- Backward pass y actualización de gradientes
- Verificación de dimensiones

### Test de Activaciones
- Forward pass de Tanh y Sigmoid
- Backward pass y derivadas
- Verificación de rangos de salida

### Test de Integración XOR
- Entrenamiento completo en problema XOR
- Verificación de convergencia
- Validación de predicciones

---

## Fundamentos Teóricos

### Retropropagación (Backpropagation)

El algoritmo de backpropagation se basa en la regla de la cadena del cálculo:

1. **Forward Pass**: Calcular la salida de la red
2. **Calcular Error**: Comparar con el valor esperado
3. **Backward Pass**: Propagar el gradiente hacia atrás
4. **Actualizar Pesos**: `W = W - α * ∇W`

### Descenso de Gradiente

La actualización de parámetros sigue la fórmula:
```
θ_nuevo = θ_viejo - α * ∂L/∂θ
```

Donde:
- `θ`: Parámetros (pesos y bias)
- `α`: Tasa de aprendizaje (learning rate)
- `∂L/∂θ`: Gradiente de la función de pérdida

---

## Análisis de Rendimiento

### Problema XOR

**Configuración:**
- Arquitectura: 2 → 3 → 1 (con Tanh)
- Épocas: 10,000
- Learning rate: 0.1

**Resultados:**
- Convergencia: ~5,000 épocas
- Error final: < 0.01
- Precisión: 100% en clasificación binaria

### Ventajas

✅ Código educativo y fácil de entender  
✅ Sin dependencias externas complejas  
✅ Arquitectura modular y extensible  
✅ Tests completos para validación  

### Limitaciones

⚠️ No optimizado para grandes datasets  
⚠️ Sin paralelización (CPU single-thread)  
⚠️ Operaciones matriciales no optimizadas  
⚠️ Solo funciones de activación básicas  

---

## Mejoras Futuras

### Optimizaciones de Rendimiento
- [ ] Usar BLAS/LAPACK para operaciones matriciales
- [ ] Implementar mini-batch gradient descent
- [ ] Paralelización con OpenMP o threads
- [ ] Optimizaciones de memoria

### Nuevas Características
- [ ] Más funciones de activación (ReLU, Leaky ReLU, Softmax)
- [ ] Más funciones de pérdida (Cross-Entropy, Hinge Loss)
- [ ] Optimizadores avanzados (Adam, RMSprop, Momentum)
- [ ] Regularización (L1, L2, Dropout)
- [ ] Capas convolucionales (CNN)
- [ ] Guardado/carga de modelos

### Herramientas
- [ ] Visualización de pérdida durante entrenamiento
- [ ] Exportación de métricas
- [ ] Interfaz de línea de comandos mejorada

---

## Solución de Problemas

### Error: "cmake: command not found"
**Solución**: Usa compilación directa con g++ (ver sección de instalación)

### Error de compilación con g++
**Solución**: Asegúrate de tener C++17:
```bash
g++ --version  # Verifica versión
g++ -std=c++17 ...  # Especifica estándar
```

### La red no converge
**Soluciones**:
- Reducir learning rate (ej: 0.01 en vez de 0.1)
- Aumentar número de épocas
- Añadir más neuronas en capa oculta
- Cambiar inicialización de pesos
- Normalizar datos de entrada

---

## Contribuciones

Este es un proyecto académico. Si deseas contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request

---

## Licencia

Este proyecto usa la licencia **MIT**. Ver archivo LICENSE para más detalles.

---

## Referencias Bibliográficas

[1] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.

[2] M. Nielsen, "Neural Networks and Deep Learning," Determination Press, 2015. [Online]. Available: http://neuralnetworksanddeeplearning.com/

[3] C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.

[4] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436-444, 2015.

---

## Contacto y Soporte

Para preguntas o problemas, por favor abre un issue en el repositorio de GitHub.

**Grupo**: group_3_custom_name  
**Curso**: CS2013 Programación III  
**Año**: 2025-1
