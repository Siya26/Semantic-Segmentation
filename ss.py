import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# def relu(x):
#     return np.maximum(0.01 * x, x)

def relu(x):
    return np.maximum(0, x)

def normalize_data(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data

lr = 0.01
actual = np.array([[0, 1, 0], 
                   [1, 1, 1], 
                   [0, 1, 0]])
#forward
input = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

input = normalize_data(input)

list_kernel1 = []
for i in range(2):
    list_kernel1.append(np.random.randn(2, 2))

bias_conv1 = np.zeros([2, 2, 2])

conv = np.random.rand(2, 2, 2)

list_kernel2 = []
for i in range(4):
    list_kernel2.append(np.random.randn(2, 2, 2))

bias_conv2 = np.zeros([4, 1])

flatten = np.random.rand(4, 1)

weight1 = np.random.randn(2, 4)
bias1 = np.zeros([2, 1])

weight2 = np.random.randn(2, 2)
bias2 = np.zeros([2, 1])

label = np.array([1, 0])
label = np.reshape(label, (2, 1))

#backward
weight2_b = np.random.randn(2, 2)
bias2_b = np.zeros([2, 1])

weight1_b = np.random.randn(3, 1)
weight1_b = np.vstack([weight1_b, [0]])
weight1_b_new = np.roll(weight1_b, 1, axis=0)
weight1_b = np.hstack((weight1_b, weight1_b_new))
bias1_b = np.zeros([4, 1])

list_kernel_b = []
for i in range(2):
    list_kernel_b.append(np.random.randn(4, 2, 2))

bias_flatten_b = np.zeros([2, 2, 2])
conv_b = np.zeros([2, 2, 2])

kernel_conv_b = np.random.randn(2, 2, 2)
bias_conv_b = np.zeros([3, 3])
input_b = np.zeros([3, 3])

for l in range(500):
    #train-forward

    for k in range(2):
        for i in range(2):
            for j in range(2):
                input_slice = input[i:i+2, j:j+2]
                conv[k][i][j] = np.mean(input_slice * list_kernel1[k]) 
    
    conv += bias_conv1
    conv = sigmoid(conv)

    for i in range(4):  
        flatten[i][0] = np.mean(conv * list_kernel2[i]) 

    flatten += bias_conv2
    flatten = sigmoid(flatten)

    hidden = np.dot(weight1, flatten) + bias1
    hidden = sigmoid(hidden)

    output = np.dot(weight2, hidden) + bias2
    output = sigmoid(output)


    #train-backward
    hidden_b = np.dot(weight2_b, output) + bias2_b
    hidden_b = sigmoid(hidden_b)

    flatten_b = np.dot(weight1_b, hidden_b) + bias1_b
    flatten_b = sigmoid(flatten_b)

    for k in range(2):
        for i in range(4):
            conv_b[k] += (flatten_b[i][0] * list_kernel_b[k][i])


    # k = 0
    # for i in range(2):
    #     for j in range(2):
    #         conv_b[i][j] = (x * flatten_b[k][0])
    #         k += 1

    conv_b += bias_flatten_b
    conv_b = sigmoid(conv_b)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                input_b[i:i+2, j:j+2] += (conv_b[k][i][j] * kernel_conv_b[k])

    input_b += bias_conv_b
    input_b = softmax(input_b)

    #back-prop

    loss = np.log(input_b) * actual
    loss = -np.sum(loss)

    if(l % 50 == 0):
        print(loss)
        #print()
    #back1
    dL = input_b - actual
    dA = dL * input_b * (1 - input_b)
    dw = np.zeros([2, 2, 2])

    for k in range(2):
        for i in range(2):
            for j in range(2):
                input_slice = dA[i:i+2, j:j+2]
                dw[k][i][j] = np.sum(input_slice * conv_b[k][i][j])

    kernel_conv_b -= lr * dw
    bias_conv_b -= lr * dA

    #back2
    dL = np.zeros([2, 2, 2])
    for k in range(2):
        for i in range(2):
            for j in range(2):
                input_slice = dA[i:i+2, j:j+2]
                dL[k][i][j] = np.sum(input_slice * kernel_conv_b[k])

    dA = dL * conv_b * (1 - conv_b)

    for k in range(2):
        dw = np.zeros([4, 2, 2])
        for i in range(4):
            dw[i] = dA[k] * flatten_b[i][0]
        list_kernel_b[k] -= lr * dw

    bias_flatten_b -= lr * dA

    #back3
    dL = np.zeros([4, 1])
    for i in range(4):
        for k in range(2):
            dL[i] += np.sum(dA[k] * list_kernel_b[k][i])

    dA = dL * flatten_b * (1 - flatten_b)
    dw = np.dot(dA, hidden_b.T)

    weight1_b -= lr * dw
    bias1_b -= lr * dA

    #back4
    dL = np.dot(weight1_b.T, dA)
    dA = dL * hidden_b * (1 - hidden_b)
    dw = np.dot(dA, output.T)

    weight2_b -= lr * dw
    bias2_b -= lr * dA

    #back5
    dL = np.dot(weight2_b.T, dA)
    dA = dL * output * (1 - output)
    dw = np.dot(dA, hidden.T)

    weight2 -= lr * dw
    bias2 -= lr * dA

    #back6
    dL = np.dot(weight2.T, dA)
    dA = dL * hidden * (1 - hidden)
    dw = np.dot(dA, flatten.T)

    weight1 -= lr * dw
    bias1 -= lr * dA

    #back7
    dL = np.dot(weight1.T, dA)
    dA = dL * flatten * (1 - flatten)
    for i in range(4):
        dw = dA[i] * conv
        list_kernel2[i] -= lr * dw

    bias_conv2 -= lr * dA

    #back8
    dL = np.zeros([2, 2, 2])
    for i in range(4):
        dL += dA[i] * list_kernel2[i]

    dA = dL * conv * (1 - conv)
    for k in range(2):
        dw = np.zeros([2, 2])
        for i in range(2):
            for j in range(2):
                input_slice = input[i:i+2, j:j+2]
                dw[i][j] = np.sum(input_slice * dA[k])
        list_kernel1[k] -= lr * dw

    bias_conv1 -= lr * dA