# Instrucciones para Compilar y Probar en CLion

## Opción 1: Usar CLion (Recomendado)

### Paso 1: Recargar el Proyecto CMake
1. Abre CLion
2. Ve a `File` → `Reload CMake Project`
3. CLion detectará automáticamente todos los ejecutables en `CMakeLists.txt`

### Paso 2: Compilar Todo
1. Ve a `Build` → `Build Project` (Ctrl+F9)
2. O haz clic en el ícono de martillo 🔨

### Paso 3: Ejecutar los Tests

En la esquina superior derecha verás un dropdown con las configuraciones:
- `neural_net_demo` - Programa principal (XOR)
- `test_matrix` - Tests de Matrix
- `test_dense` - Tests de Dense layer
- `test_activation` - Tests de Activación
- `test_xor` - Test de integración XOR

**Para ejecutar cada uno:**
1. Selecciona la configuración del dropdown
2. Haz clic en el botón verde ▶️ (Run) o Shift+F10
3. Verás la salida en la consola de CLion

### Paso 4: Ejecutar Todos los Tests
Puedes ejecutar cada test uno por uno desde el dropdown.

---

## Opción 2: Usar Git Bash

Si prefieres la línea de comandos:

### Paso 1: Abrir Git Bash
1. Haz clic derecho en la carpeta del proyecto
2. Selecciona "Git Bash Here"

### Paso 2: Compilar con g++

```bash
# Test de Matrix
g++ tests/test_matrix.cpp -o test_matrix.exe -I src -std=c++17

# Test de Dense
g++ tests/test_dense.cpp src/Network.cpp src/layers/Dense.cpp -o test_dense.exe -I src -std=c++17

# Test de Activation
g++ tests/test_activation.cpp -o test_activation.exe -I src -std=c++17

# Test de XOR
g++ tests/test_xor.cpp src/Network.cpp src/layers/Dense.cpp -o test_xor.exe -I src -std=c++17

# Programa principal
g++ src/main.cpp src/Network.cpp src/layers/Dense.cpp -o neural_net_demo.exe -I src -std=c++17
```

### Paso 3: Ejecutar los Tests

```bash
./test_matrix.exe
./test_dense.exe
./test_activation.exe
./test_xor.exe
./neural_net_demo.exe
```

---

## Opción 3: Usar Visual Studio Developer Command Prompt

Si tienes Visual Studio instalado:

### Paso 1: Abrir Developer Command Prompt
1. Busca "Developer Command Prompt for VS" en el menú inicio
2. Navega al directorio del proyecto:
   ```
   cd C:\Users\glmpl\CLionProjects\proyecto-final-2025-2-novo
   ```

### Paso 2: Compilar con cl (MSVC)

```cmd
cl /EHsc /std:c++17 /I src tests\test_matrix.cpp /Fe:test_matrix.exe
cl /EHsc /std:c++17 /I src tests\test_dense.cpp src\Network.cpp src\layers\Dense.cpp /Fe:test_dense.exe
cl /EHsc /std:c++17 /I src tests\test_activation.cpp /Fe:test_activation.exe
cl /EHsc /std:c++17 /I src tests\test_xor.cpp src\Network.cpp src\layers\Dense.cpp /Fe:test_xor.exe
cl /EHsc /std:c++17 /I src src\main.cpp src\Network.cpp src\layers\Dense.cpp /Fe:neural_net_demo.exe
```

---

## Resultados Esperados

### test_matrix.exe
```
Running Matrix tests...
[PASS] Matrix creation test
[PASS] Matrix multiplication test
[PASS] Matrix transpose test
[PASS] Matrix addition test
[PASS] Matrix subtraction test
[PASS] Matrix scalar multiplication test
[PASS] Matrix Hadamard product test

All Matrix tests passed!
```

### test_dense.exe
```
Running Dense layer tests...
[PASS] Dense forward pass test
[PASS] Dense backward pass test
[PASS] Dense dimensions test

All Dense layer tests passed!
```

### test_activation.exe
```
Running Activation function tests...
[PASS] Tanh forward pass test
[PASS] Tanh backward pass test
[PASS] Sigmoid forward pass test
[PASS] Sigmoid backward pass test

All Activation function tests passed!
```

### test_xor.exe
```
Running XOR integration test...
Testing XOR problem training...
Training network...
Epoch 100/5000 error=0.25
Epoch 200/5000 error=0.18
...
Epoch 5000/5000 error=0.001

Predictions:
-0.95
0.92
0.91
-0.94

Expected:
0
1
1
0

[PASS] XOR training test - Network learned the pattern!

XOR integration test completed!
```

### neural_net_demo.exe
```
Epoch 100/10000 error=0.25
Epoch 200/10000 error=0.18
...
Epoch 10000/10000 error=0.001
Predictions:
-0.95
0.92
0.91
-0.94
```

---

## Solución de Problemas

### Error: "No se reconoce como comando"
- Asegúrate de usar Git Bash o Developer Command Prompt
- CLion debería funcionar automáticamente

### Error de compilación: "cannot find file"
- Verifica que estés en el directorio correcto del proyecto
- Usa rutas relativas como se muestra arriba

### La red no converge en test_xor
- Es normal que tome varias épocas
- Los valores pueden variar debido a inicialización aleatoria
- Si los valores están cerca de -1/0/1, está funcionando correctamente
