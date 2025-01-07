# -*- coding: utf-8 -*-
# @Time : 2025/1/6 17:58
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : eval_main.py
# @Project : ai_model_finetune
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask

class Local_llms(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Local llms"

model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/sft_model/full_model")
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/sft_model/full_model")

mistral_7b = Local_llms(model=model, tokenizer=tokenizer)

# Define benchmark with specific tasks and shots
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=3
)

# Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)






