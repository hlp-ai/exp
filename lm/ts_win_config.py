
dtype = "float16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster, not on Windows

vocab_source = "custom"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 4096  # the Llama 2 tokenizer has 32K tokens
batch_size = 128

# model
dim = 64
n_layers = 5
n_heads = 8
n_kv_heads = 4

flops_promised = 15e12
