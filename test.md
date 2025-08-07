Mathematical Principles of LSTM Networks
The Long Short-Term Memory (LSTM) network is a specialized Recurrent Neural Network (RNN) designed to model sequential data with long-term dependencies. Its core mechanism involves gates that control the flow and retention of information. This document provides a mathematical description of the LSTM's operations in a format suitable for GitHub rendering.
LSTM Structure
An LSTM processes an input sequence ( x_t \in \mathbb{R}^d ) at each time step ( t ), where ( d ) is the input dimension. It maintains a hidden state ( h_t \in \mathbb{R}^h ) and a cell state ( c_t \in \mathbb{R}^h ), where ( h ) is the hidden dimension. The LSTM uses four main components:

Forget Gate: Determines which information to discard from the cell state.
Input Gate: Controls which new information to add to the cell state.
Cell State: Maintains long-term memory across the sequence.
Output Gate: Generates the hidden state output.

Mathematical Formulation
The LSTM updates at each time step ( t ) are defined by the following equations:
1. Forget Gate
The forget gate decides which parts of the previous cell state ( c_{t-1} ) to retain:[f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)]

( f_t \in \mathbb{R}^h ): Forget gate activation vector, with values in ([0, 1]).
( W_f \in \mathbb{R}^{h \times (h+d)} ): Weight matrix for the forget gate.
( b_f \in \mathbb{R}^h ): Bias vector.
( [h_{t-1}, x_t] \in \mathbb{R}^{h+d} ): Concatenation of the previous hidden state and current input.
( \sigma(x) = \frac{1}{1 + e^{-x}} ): Sigmoid activation function.

2. Input Gate
The input gate determines which new information to incorporate:[i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)]The candidate cell state is computed as:[\tilde{c}t = \tanh(W_c \cdot [h{t-1}, x_t] + b_c)]

( i_t \in \mathbb{R}^h ): Input gate activation vector.
( \tilde{c}_t \in \mathbb{R}^h ): Candidate cell state.
( W_i, W_c \in \mathbb{R}^{h \times (h+d)} ): Weight matrices.
( b_i, b_c \in \mathbb{R}^h ): Bias vectors.
( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} ): Hyperbolic tangent function, output in ([-1, 1]).

3. Cell State Update
The cell state is updated by combining the previous cell state and the candidate state:[c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t]

( c_t \in \mathbb{R}^h ): Current cell state.
( \odot ): Element-wise (Hadamard) product.

4. Output Gate
The output gate controls the hidden state output:[o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)]The hidden state is computed as:[h_t = o_t \odot \tanh(c_t)]

( o_t \in \mathbb{R}^h ): Output gate activation vector.
( h_t \in \mathbb{R}^h ): Current hidden state.
( W_o \in \mathbb{R}^{h \times (h+d)} ): Weight matrix.
( b_o \in \mathbb{R}^h ): Bias vector.

Data Flow
At each time step ( t ):

Compute the forget gate ( f_t ) to retain relevant information from ( c_{t-1} ).
Compute the input gate ( i_t ) and candidate cell state ( \tilde{c}_t ) to add new information.
Update the cell state ( c_t ) using the forget and input gates.
Compute the output gate ( o_t ) and generate the hidden state ( h_t ).

The hidden state ( h_t ) and cell state ( c_t ) are passed to the next time step or used for task-specific outputs (e.g., classification, regression).
Parameter and Computational Complexity

Parameter Count: Each gate (forget, input, candidate, output) has a weight matrix and bias, resulting in a total of:[4 \times (h \cdot (h + d) + h)]parameters. For example, with ( d = 50 ) and ( h = 100 ), the total parameters are approximately ( 4 \times (100 \cdot 150 + 100) = 60,400 ).

Computational Complexity: The complexity per time step is ( O(h^2 + h d) ). For a sequence of length ( T ), the total complexity is:[O(T \cdot (h^2 + h d))]


Training
LSTM parameters are optimized using Backpropagation Through Time (BPTT):

A loss function (e.g., mean squared error or cross-entropy) is defined.
Gradients are computed with respect to the weights ( W_f, W_i, W_c, W_o ) and biases ( b_f, b_i, b_c, b_o ).
The gate-based structure mitigates the vanishing gradient problem, enabling stable training for long sequences.

Key Features

Long-Term Dependencies: The cell state ( c_t ) allows information to persist across long sequences via additive updates.
Gated Mechanism: The forget, input, and output gates dynamically control information flow.
Applications: Suitable for time-series prediction, natural language processing (e.g., language modeling), and speech recognition.

This formulation ensures that LSTM networks can effectively model complex sequential data while maintaining computational efficiency.
