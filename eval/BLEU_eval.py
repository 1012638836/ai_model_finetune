# -*- coding: utf-8 -*-
# @Time : 2025/1/7 18:58
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : BLEU_eval.py
# @Project : ai_model_finetune
import json, requests, nltk, jieba
import pandas as pd

'''
该脚本基于BLEU测试大模型的垂直领域能力，模型是基于litellm部署的ollama服务
'''

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

def bleu_cal(reference, hypothesis):
    hypothesis = jieba.lcut(hypothesis)
    reference = jieba.lcut(reference)
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=[0.25, 0.25, 0.25, 0.25])
    return BLEUscore

def main(n = 100):
    data = pd.read_csv('../data/启牛话术文档rewrite.csv').sample(n = n).reset_index(drop = True)
    data['llms_result'], data['eval_score'] = '', ''
    data_df_list = []
    for index, row in data.iterrows():
        question, answer = row['rewrite'], row['answer']
        llms_result = send_request(llms_input=question)
        bleu_score = bleu_cal(answer, llms_result)
        data_df_list.append([question, answer, llms_result, bleu_score])
        print(f'{index} is done! {question} - {answer} -{llms_result} - {bleu_score}')
    data_df = pd.DataFrame(data_df_list, columns = ['question', 'answer', 'llms_result', 'bleu_score'])
    data_df.to_csv('../data/评测结果_bleu_score.csv', index = False, encoding = 'utf-8-sig')

if __name__ == '__main__':
    main()
