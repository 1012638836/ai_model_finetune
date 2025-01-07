# -*- coding: utf-8 -*-
# @Time : 2025/1/6 18:16
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : g_eval.py
# @Project : ai_model_finetune
import requests, json, re
import pandas as pd
from openai import AzureOpenAI

IMPROVED_JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
Evaluation: """

def gpt4_evaluate(prompt):
    client = AzureOpenAI(
        api_key="02855675d52d4abfa48868c00c6f2773",
        api_version="2023-05-15",
        azure_endpoint="https://test-az-eus-ai-openai01.openai.azure.com/")
    response = client.chat.completions.create(
        model="test-az-eus-gpt-4o",  # model = "deployment_name".
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    llms_content = response.choices[0].message.content
    return llms_content

def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None

# 后端请求处理函数
def send_request(llms_name = 'qwen2_13b_gguf', port = 4000, llms_input = ''):

    # 构建请求的URL
    url = f"http://localhost:{port}/v1/chat/completions"

    # 请求的payload
    payload = {
        "model": llms_name,
        "messages": [{"role": "user", "content": llms_input}]
    }

    headers = {"Content-Type": "application/json"}

    try:
        # 发送POST请求
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # 如果请求失败，抛出异常
        response_json = response.json()
        llms_output = response_json.get('choices', [{}])[0].get('message', {}).get('content', '无响应内容')
    except requests.exceptions.RequestException as e:
        llms_output = f"请求失败: {e}"

    return llms_output

def main(n = 100):
    data = pd.read_csv('../data/启牛话术文档rewrite.csv').sample(n = n).reset_index(drop = True)
    data['llms_result'], data['eval_score'] = '', ''
    data_df_list = []
    for index, row in data.iterrows():
        question = row['rewrite']
        llms_result = send_request(llms_input=question)
        current_prompt = IMPROVED_JUDGE_PROMPT.format(question = question, answer = llms_result)
        gpt4_evl_result = gpt4_evaluate(current_prompt)
        eval_score = extract_judge_score(gpt4_evl_result)
        data_df_list.append([question, llms_result, gpt4_evl_result, eval_score])
        print(f'{index} is done! {question} -{llms_result} - {gpt4_evl_result}')
    data_df = pd.DataFrame(data_df_list, columns = ['question', 'llms_result', 'gpt4_evl_result', 'eval_score'])
    data_df.to_csv('../data/评测结果.csv', index = False, encoding = 'utf-8-sig')

if __name__ == '__main__':
    main()



