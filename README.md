# llama3.np

`llama3.np` is pure NumPy implementation for Llama 3 model. For an accurate implementation, I ran the [stories15M model](https://github.com/karpathy/llama2.c?tab=readme-ov-file#models) trained by Andrej Karpathy. 

- For a detailed explanation in English, see [Llama 3 implemented in pure NumPy](https://docs.likejazz.com/llama3.np/). **[English Version]**
- 한글로 작성된 상세한 설명은 [NumPy로 구현하는 라마 3 모델](https://docs.likejazz.com/llama3.np-ko/)을 참고하세요. **[Korean Version]**

## Usage

```shell
$ python llama3.py "I have a dream"
"""
I have a dream. He dream of a big, beautiful garden full of flower and tree. He dream of playing with hi friend and eating yummy snack.
One day, he wa walking in the garden when he saw

Token count: 50, elapsed: 1.53s, 33 tokens/s
"""
```

## Citing llama3.np

If you use or discuss `llama3.np` in your academic research, please cite the project to help spread awareness:

```
@misc{llama3.np,
  title = {llama3.np: pure NumPy implementation for Llama 3 model},
  author = {Sang Park}, 
  howpublished = {\url{https://github.com/likejazz/llama3.np}},
  note = {llama3.np, MIT License}
  year = {2024},
}
```

# References
Thank you to the creators of the following libraries and tools and their contributors:
- [llama2.c](https://github.com/karpathy/llama2.c) - @karpathy
- [llama.np](https://github.com/hscspring/llama.np) - @hscspring
- [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) - Hugging Face's Transformers

I got a lot of information from the articles below:
- [42dot LLM 1.3B](https://42dot.ai/blog/178) - 42dot
- [Exploring and building the LLaMA 3 Architecture : A Deep Dive into Components, Coding, and Inference Techniques](https://medium.com/@vi.ai_/exploring-and-building-the-llama-3-architecture-a-deep-dive-into-components-coding-and-43d4097cfbbb) - @vi.ai_
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) - EleutherAI
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - NVIDIA

# License
MIT