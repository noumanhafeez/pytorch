# What are the Tensors?

### Tensor is specialized multi-dimensional array (data structure) designed for mathematical and computational efficiency. Just like arrays, vector, numpy, tensor is optmized multi-dimensional array data structure. 

## Real World Examples:

### 1. Scalar: 0 dimensional tensors (a single number). For example: any cost or loss value i.e 5.0, 14,5 etc

### 2. Vectors: 1 dimensional tensors (a list of numbers). For example: if you give text "Hello" to neural network and ask for embedding it, then Hello will convert to 1 dimensional vector array: [0.12, 0.22, 0.32, 0.11, 0.99] -> "Hello". Embeddings means conversion of text to number arrays so machine could understand easily. 

### Note: You can train embeddings on text, but it will be costly expensive. So, for this, you can use pretrained embedding models like: Word2vec, Glove, FastText, BERT, RoBERTA, GPT etc

### 3. Matrix: 2D dimensional tensors (a 2D grid of numbers). For example: a grayscale image can be represented as a 2D matrix. Like: [[1,2,3], [5,3,2]]

### 4. 3D tensors: Colored Images. For example: A single RGB image can be represented by 3D tensors (width * height * channels). RGB image shape (256 * 256): [256, 256, 3]

### 5. 4D tensors: Batches of RGB images. For example: if you have huge dataset of images, then you give data to neural networks in batches like 32, 64 etc. A RGB image size is (128 * 128). So, you want to give it to model in batches 32, then dimension will become [32, 128, 128, 3]

### 6: 5D tensors: Video Data. Adds a time dimension (i.e video frame). For example: a batch of 10 video clips, each with 16 frames of size 64 * 64 with RGB colors, then it would have shape of [10, 64, 64, 3]

# Where are Tensors used in Deep Learning?

### 1. Data Storage: Training data (image, text, audio) is stored in tensors.

### 2. Weights and Biases: The parameters are stored as tensors.

### 3. Matrix Operation: For operation between matrix, like add, dot product, tensor is used.

### 4. Training Process: During forward pass, or backward pass, tensor is use for calculation.

