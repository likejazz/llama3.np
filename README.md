# llama3.np

`llama3.np` is pure NumPy implementation for Llama 3 model.

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
  title = {llama3.np is pure NumPy implementation for Llama 3 model},
  author = {Sang Park}, 
  howpublished = {\url{https://github.com/likejazz/llama3.np}},
  note = {llama3.np, MIT License}
  year = {2024},
}
```

# References
Thank you to to the creators of the following libraries and tools and their contributors:
- [llama2.c](https://github.com/karpathy/llama2.c) - @karpathy
- [llama.np](https://github.com/hscspring/llama.np) - @hscspring
- [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

I got a lot of information from the articles below:
- [Exploring and building the LLaMA 3 Architecture : A Deep Dive into Components, Coding, and Inference Techniques](https://medium.com/@vi.ai_/exploring-and-building-the-llama-3-architecture-a-deep-dive-into-components-coding-and-43d4097cfbbb)
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)

# License
MIT