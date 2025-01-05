## 模型微调到部署服务一键式指南
#### 转换safetensors为gguf文件
``conda activate llama``
``python convert_hf_to_gguf.py --outfile /root/autodl-tmp/hf2gguf/Qwen25.gguf sft_trained_full_model_full_path``
#### 写MakeFile将gguf变成ollama模型
*ModelFile 文件内容*
```
FROM /root/autodl-tmp/hf2gguf/Qwen25.gguf
# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.05
PARAMETER top_k 20

TEMPLATE """{{ if and .First .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}"""

# set the system message
SYSTEM """
You are a helpful assistant.
"""
```
*基于ModelFile部署ollama对象*

``ollama create qwen2_13b_gguf -f Modelfile``

#### 基于litellm部署http服务
``conda activate llama``
``litellm --model ollama/qwen2_13b_gguf``

- litellm会分配端口，默认端口为4000
- curl http://localhost:4000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"model": "qwen2_13b_gguf", "messages": [{"role": "user", "content": "Hello! What is your name?"}]}'

