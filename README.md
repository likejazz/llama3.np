# llama3.np

llama3.np is pure NumPy implementation for Llama 3 model.

한글로 설명한 상세한 내용은 [NumPy로 구현하는 라마 3 모델](https://docs.likejazz.com/llama3/)을 참고하세요.

## Usage

```shell
$ python llama3.py "I have a dream"
"""
I have a dream. He dream of a big, beautiful garden full of flower and tree. He dream of playing with hi friend and eating yummy snack.
One day, he wa walking in the garden when he saw

Token count: 50, cost: 1.53s, 33 tokens/s
"""
```

## Citing llama3.np

If you use or discuss `llama3.np` in your academic research, please cite the project to help spread awareness:

```
@misc{llama3.np,
  author = {Sang Park}, 
  title = {Inference Llama 3 in single file of pure NumPy},
  year = {2024},
  month = {05},
  howpublished = {\url{https://github.com/likejazz/llama3.np}},
  note = {Llama3.np, Apache License}
}
```

# References

- [llama2.c](https://github.com/karpathy/llama2.c)
- [llama.np](https://github.com/hscspring/llama.np)
- [transformers/src/transformers/models/llama/modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [Exploring and building the LLaMA 3 Architecture : A Deep Dive into Components, Coding, and Inference Techniques](https://medium.com/@vi.ai_/exploring-and-building-the-llama-3-architecture-a-deep-dive-into-components-coding-and-43d4097cfbbb)
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)

# License
MIT