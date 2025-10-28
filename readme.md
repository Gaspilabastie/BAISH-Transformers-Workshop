# AI Safety: Introduction to Transformer Architecture

**A hands-on workshop where I'll build language models from scratch to understand why transformers became the dominant architecture in modern AI.**

## What I'll Build

Starting from the simplest possible model, I'll implement increasingly sophisticated architectures, discovering their strengths and limitations through experimentation. By the end, I hope to understand:

- What problems each architecture solves
- Why certain designs won over others
- How the attention mechanism works
- The path from simple n-grams to modern transformers

## Prerequisites (if you want to fork it)

- Python 3.8+
- Basic neural network concepts
- Some PyTorch familiarity

## Setup

```bash
# Clone this repository
git clone <repo-url>
cd transformer-workshop

# Install dependencies
pip install torch

# Dataset location
# Files should be in Dataset/ folder
```

## Structure

### Part 1: Statistical Baselines
**Count-Based Models**
- Implement bigram and trigram models using frequency counting
- Understand probabilistic language modeling
- Establish baseline performance

### Part 2: Neural N-grams
**From Counting to Learning**
- Implement the same models with gradient descent
- Compare neural vs. statistical approaches
- Learn about embedding layers and lookup tables

### Part 3: Multi-Layer Perceptrons
**Adding Depth and Representation Learning**
- Build an MLP with learned embeddings and hidden layers
- Experiment with context length and model size
- Discover the parallel processing advantage
- Learn about regularization (weight decay, dropout)

**Key Question**: Why does this fixed-window approach work so well?

### Part 4: Recurrent Models
**Sequential Processing**
- Implement vanilla RNN
- Upgrade to LSTM with gated cells
- Compare sequential vs. parallel processing
- Experiment with context length

**Key Questions**: 
- Why don't RNNs beat MLPs in our setup?
- What problems do gates solve?
- When would you choose RNN over MLP?

### Part 5: Transformers
**The Modern Approach**
- Implement multi-head self-attention
- Build complete transformer blocks
- Add positional encodings
- Compare to all previous models

**Key Question**: How does attention solve the limitations of both MLPs and RNNs?

## To experiment and think

### 🔬 Experiments to Try

- **Context length**: What happens with shorter/longer contexts?
- **Model size**: Try different hidden dimensions
- **Regularization**: Add/remove dropout, weight decay
- **Training time**: Does more training always help?
- **Data size**: Try with 1 book vs 2 books

### 💬 Discussion Points

Key topics to think and discuss:
- What makes a good language model?
- Trade-offs between different architectures
- When would you choose each approach?
- How do these principles apply to modern LLMs?

## Code Structure

All models follow this pattern:

```python
class ModelName(nn.Module):
    def __init__(self, vocab_size, ...):
    
    def forward(self, idx):
        return logits
    
    def generate(self, idx, max_new_tokens):
        # Autoregressive text generation
        return idx

# Training loop
# Evaluation and visualization
```

## Tips for Success

### 🐛 Debugging
- Check tensor shapes frequently
- Use small models first to debug faster
- Print intermediate outputs
- Compare with solution if stuck

### ⚡ Performance
- Start with small models (faster iteration)
- Use `steps_per_epoch=500` for quick experiments
- GPU not required but helps
- Watch your CPU temperature!

### 📊 Comparing Models
- Keep hyperparameters consistent across models
- Look at both train AND validation loss
- Generated text quality matters too
- Training time is a real constraint

### 🤔 When Stuck
2. Verify tensor shapes match expected dimensions
3. Look at this solution for guidance
4. Ask peers!

## Common Questions

**Q: Do I need a GPU?**
A: No! All models train in minutes on CPU.

**Q: What if my model performs worse than expected?**
A: That's part of the experiment! Certain architectures struggle in certain scenarios.

**Q: Can I use different hyperparameters?**
A: Absolutely! Experimentation is encouraged.

**Q: How do I know if my implementation is correct?**
A: If it trains (loss goes down) and generates somewhat coherent text, you're on the right track!

## Competition (Optional)
Want to compete? Try to achieve the **lowest validation loss**!

**Aclarations:**
- In the repository I provide training data. Feel free to add your own.
- Feel free to do any architecture modification too.
- I strongly recommend to document your approach.

**If you wanna upload your own repository, only upload:**
- Your best model code
- Final train/val loss
- Brief explanation of what you tried

**Categories:**
- Overall best validation loss
- Most creative architecture
- Best efficiency (loss per parameter)

## Resources

### For Later Study
- Bengio et al. (2003) - Neural Language Models
- Vaswani et al. (2017) - "Attention Is All You Need"
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Karpathy's "Unreasonable Effectiveness of RNNs"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Why do this project:

You can learn how to:

- Explain how language models work from first principles
- Implement key architectures from scratch
- Understand the attention mechanism intuitively
- Recognize trade-offs between different approaches
- Debug and experiment with neural architectures
- Connect historical developments to modern AI systems


**Let's build some transformers! 🚀**
