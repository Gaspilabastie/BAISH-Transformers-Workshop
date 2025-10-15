import torch
import torch.nn.functional as F
import sys

class CountBasedBigram:
    """
    Count-based bigram model using PyTorch tensors.
    Builds a frequency table, no gradient descent needed!
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.counts = torch.ones(vocab_size, vocab_size)
    
    def train(self, data):
        """
        Count bigram frequencies in the data.
        Args:
            data: torch.LongTensor of token indices
        """
        for int_1, int_2 in zip(data[:-1], data[1:]):
            self.counts[int_1,int_2] += 1
    
    def get_probabilities(self):
        """
        Convert counts to probabilities.
        Returns: (vocab_size, vocab_size) tensor where [i, j] = P(j | i)
        """
        probablilities = torch.zeros(vocab_size, vocab_size)
        probablilities = self.counts/self.counts.sum(dim=1, keepdim=True)
        return probablilities
    
    def generate(self, idx, max_new_tokens):
        """
        Generate tokens by sampling from the count-based distribution.
        Args:
            idx: batch starting context
            max_new_tokens: number of tokens to generate
        """
        probs = self.get_probabilities()
        for _ in range(max_new_tokens):
            current_token = idx[:, -1]
            next_probs = probs[current_token]
            idx_next = torch.multinomial(next_probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def calculate_loss(self, data):
        """
        Calculate cross-entropy loss (negative log-likelihood).
        Same metric as neural models!
        """
        probs = self.get_probabilities()
        
        # Get current and next tokens
        current_tokens = data[:-1]
        next_tokens = data[1:]
        
        # Get probabilities for the actual next tokens that occurred
        token_probs = probs[current_tokens, next_tokens]
        
        # Calculate negative log likelihood
        epsilon = sys.float_info.min
        loss = -torch.log(torch.clamp(token_probs, min=epsilon))
        mean = torch.mean(loss)
        return mean

    def decode(self, data):
        pass


if __name__ == "__main__":
    # Load and encode data
    with open('Dataset/Anne_of_Green_Gables.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize data
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {}
    itos = {}
    for i, char in enumerate(chars):
        stoi[char] = i
        itos[i] = char

    data = []
    for char in text:
        data += [stoi[char]]
    
    # Split data
    n = 0.8
    index = int(len(data)*0.8)
    train_data = data[:index]
    val_data = data[index:]
    """
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {len(train_data)} tokens")
    print(f"Testing on {len(val_data)} tokens")
    print(f"Train/ Test ratio: {len(train_data)/(len(train_data)+len(val_data))}")
    """

    # Create model
    model = CountBasedBigram(vocab_size)
    
    # Train (just count!)
    print("\nCounting bigrams...")
    model.train(train_data)
    print("Done!")
    
    # Calculate losses
    train_loss = model.calculate_loss(train_data[:10000])
    val_loss = model.calculate_loss(val_data[:10000])
    print(f"\nTrain loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")
    
    # Generate
    print("\nGenerated text:")
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, 500)
    print(decode(generated[0].tolist()))