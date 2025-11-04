import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLanguageModel(nn.Module):
    """
    MLP-based language model with fixed context window.
    Uses learned embeddings and hidden layers (unlike n-gram lookup tables).
    """
    def __init__(self, vocab_size, context_length, embed_dim, hidden_dim):
        super().__init__()
        self.context_length = context_length
        
        # The embedding layer maps each token to a learned vector of size embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.hidden_layer_1 = nn.Linear(context_length*embed_dim, hidden_dim)
        
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx):
        batch_size = idx.shape[0]

        # Get embeddings for context tokens
        # Shape: (batch, context_length, embed_dim)
        tokens = self.embedding(idx)
        
        # Flatten embeddings
        # Shape: (batch, context_length * embed_dim)
        flatted_tokens = tokens.view(batch_size, -1)

        # Pass through first hidden layer + activation
        first_layer = F.relu( self.hidden_layer_1(flatted_tokens) )
        
        # Pass through second hidden layer + activation
        second_layer = F.relu( self.hidden_layer_2(first_layer) )
        
        # Pass through output layer (no activation - raw logits)
        # Return shape: (batch, vocab_size)
        logits = self.output_layer(second_layer)
        
        return logits
    
    def generate(self, idx, max_new_tokens):
        """Generate text by sampling from the learned distribution."""
        for _ in range(max_new_tokens):
            # Crop context to last context_length tokens
            idx_cond = idx[:, -self.context_length:]
            
            # Pad if needed (for start of sequence)
            if idx_cond.shape[1] < self.context_length:
                padding = torch.zeros(idx.shape[0], self.context_length - idx_cond.shape[1], 
                                     dtype=torch.long)
                idx_cond = torch.cat([padding, idx_cond], dim=1)
            
            # Get predictions
            logits = self(idx_cond)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def get_batch(data, context_length, batch_size):
    """
    Sample random batches from data.
    Returns context windows and targets.
    """
    # Sample random starting indices
    ix = torch.randint(len(data) - context_length, (batch_size,))

    # Extract context windows
    x = torch.stack([data[i:i+context_length] for i in ix])
    
    # Get target tokens (next token after each context)
    y = data[ix + context_length]
    
    return x, y


def estimate_loss(model, data, context_length, batch_size, eval_iters=100):
    """Evaluate model loss over multiple batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        X, Y = get_batch(data, context_length, batch_size)
        # Get model predictions
        logits = model(X)
        
        # Calculate loss
        loss = F.cross_entropy(logits, Y)
        
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    max_iters = 5000
    eval_interval = 500
    learning_rate = 1e-3
    
    # Model hyperparameters
    context_length = 8      # Look at 8 previous characters
    embed_dim = 24          # Size of token embeddings
    hidden_dim = 128        # Size of hidden layers
    
    # Load and encode data
    with open('Dataset/Anne_of_Green_Gables.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {len(train_data)} tokens")
    print(f"Context length: {context_length}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Hidden dim: {hidden_dim}")
    
    # Create model
    model = MLPLanguageModel(vocab_size, context_length, embed_dim, hidden_dim)
    
    # Print parameter count
    print( sum(p.numel() for p in model.parameters()) )
    
    
    # Create optimizer (Adam is good for MLPs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nTraining...")
    for iter in range(max_iters):
        xb, yb = get_batch(train_data, context_length, batch_size)
        
        # Forward pass
        logits = model(xb)  # (batch, vocab_size)
        
        # Calculate loss
        loss = F.cross_entropy(logits, yb)
        
        # Backward pass
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = estimate_loss(model, train_data, context_length, batch_size)
            val_loss = estimate_loss(model, val_data, context_length, batch_size)
            print(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f}")
    
    # Generate
    print("\nGenerated text:")
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, 500)[0].tolist()))