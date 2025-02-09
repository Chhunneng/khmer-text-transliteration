## Model Architecture

### Seq2Seq with Attention
- Encoder: 256-unit BiLSTM
- Decoder: 256-unit LSTM with Bahdanau Attention
- Embedding: 128D character embeddings

### Training Details
- Optimizer: Adam (lr=0.001)
- Batch Size: 64
- Early Stopping: Patience=5