# Projetos CUDA Ordenados por Dificuldade

## Nível 1 – Iniciante

### 1. Hello World na GPU

- **Descrição:** Exibir mensagens da CPU e GPU para entender kernels e threads.
- **Requisitos:** CUDA Toolkit instalado, compilador `nvcc`, GPU compatível.

### 2. Soma de Vetores

- **Descrição:** Somar dois vetores de tamanho N em paralelo usando threads.
- **Requisitos:** Conhecimento básico de C++, operações básicas de CUDA, `cudaMalloc`, `cudaMemcpy`.

### 3. Conversão de Imagem Preto e Branco

- **Descrição:** Transformar uma imagem colorida em preto e branco na GPU.
- **Requisitos:** Biblioteca para leitura de imagens (ex: OpenCV ou stb_image), kernels CUDA para processamento pixel a pixel.

---

## Nível 2 – Intermediário

### 4. Multiplicação de Matrizes

- **Descrição:** Multiplicar duas matrizes grandes usando CUDA, comparando performance CPU vs GPU.
- **Requisitos:** Manipulação de arrays multidimensionais, conhecimento de blocos e threads, memória compartilhada para otimização.

### 5. Filtro de Imagem (Blur ou Sobel)

- **Descrição:** Aplicar filtros como blur ou detecção de bordas a imagens.
- **Requisitos:** Manipulação de imagens, kernels 2D, memória compartilhada para otimização de vizinhança de pixels.

### 6. Fractais (Mandelbrot / Julia)

- **Descrição:** Gerar fractais coloridos calculando cada pixel em paralelo.
- **Requisitos:** Operações matemáticas complexas, paralelismo pixel a pixel, salvar imagem final (ex: PNG).

---

## Nível 3 – Avançado

### 7. Simulação de Partículas

- **Descrição:** Simular partículas se movendo com gravidade ou colisões elásticas.
- **Requisitos:** Vetores de posições e velocidades, kernels para atualização paralela, potencial uso de memória compartilhada para partículas próximas.

### 8. Monte Carlo (Ex: Estimar π)

- **Descrição:** Gerar números aleatórios em threads independentes e calcular estatísticas.
- **Requisitos:** Números aleatórios na GPU (`curand`), paralelismo massivo, redução de resultados na CPU.

### 9. Redes Neurais Simples

- **Descrição:** Implementar um perceptron ou rede feedforward na GPU, com multiplicação de matrizes para propagação.
- **Requisitos:** Álgebra linear na GPU, kernels para multiplicação e soma de matrizes, conhecimento básico de aprendizado de máquina.

---

## Nível 4 – Expert

### 10. Simulação de Fluidos (SPH)

- **Descrição:** Simular fluidos usando Partículas de Suavização (Smoothed Particle Hydrodynamics) na GPU.
- **Requisitos:** Física computacional, estruturas de dados para vizinhança, memória compartilhada e global, otimização de performance.

### 11. Ray Tracing Paralelo

- **Descrição:** Implementar um renderizador de raios simples usando CUDA.
- **Requisitos:** Matemática 3D, operações de interseção de raios, paralelismo por pixel, geração de imagens de alta resolução.

### 12. Treinamento de Rede Neural Convolucional

- **Descrição:** Implementar CNN do zero, treinando um dataset simples como MNIST.
- **Requisitos:** Multiplicação de tensores, convoluções 2D, operações de pooling, memória eficiente, kernel para propagação e retropropagação.

---

> **Dica:** Comece pelos níveis iniciante e intermediário. Conforme ganhar confiança com kernels, blocos e memória compartilhada, avance para projetos avançados e expert.

