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

# What is derivatives and why do we need this?

### Derivative = “How much does output change if I slightly change input?”

### In ML: Model output depends on weights Derivative tells: “If I slightly change this weight, does my prediction get better or worse?”

### Without derivatives, the model has no idea which direction to improve.

## What is the Chain Rule Multiplication (the heart of deep learning ❤️)

### Chain rule answers: If A affects B, and B affects C, how much does A affect C?

### Similarly in math: Suppose y = (x² + 1) z = 3y. Then we want: dy/dz. So, chain rule will apply here:

### dz/dx = dz/dy * dy/dx. Solve it Steps by Steps: dy/dx = 2x, dz/dy = 3. dz/dx = dz/dy × dy/dx = 3 × 2x = 6x. It's means, the rate changing regarding to parameter x and y is 6x. I just use 2 equation here. Assume we have 100, 1000 or maybe millions of equations, then apply chain rule to each individually will impossible by hand.  

### So, here, we can use backpropagation method for compute fast

## Now the BIG PICTURE (training loop)

### Training a neural network has 4 steps, repeated again and again 🔁


## Step 1: 

### Forward Pass (prediction time):  Input goes forward through the network. Input image → CNN → FC → Output (cat or dog) Here: Model uses current weights Produces a prediction

### 📌 No learning yet — just guessing

## Step 2:

### Compute Loss (how wrong am I?) Loss = “How bad is my prediction?” loss = (prediction - true)² Loss is a single number that summarizes error.


## Step 3: 

### Backward Pass (learning happens here). Now we ask: “Which weight caused the error, and by how much?"  Backward pass:

### Start from loss

### Move backward layer by layer

### Apply chain rule

### Compute gradients

### Meaning: “If I change this weight a little, how much will loss change?”

### 📌 This is why it’s called backpropagation.

## Step 4: 

### Update Weights (become smarter): Now we update weights using gradients. 

### Rule: new_weight = old_weight - learning_rate × gradient


## These are the core steps we use use in trainning process. For backpropagation, we use autograd (a library  of pytorch) for auto derivatives 