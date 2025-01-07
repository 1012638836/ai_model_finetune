# -*- coding: utf-8 -*-
# @Time : 2025/1/6 10:27
# @Author : lijinze
# @Email : lijinze@lzzg365.cn
# @File : app.py
# @Project : ai_model_finetune
import gradio as gr
import requests
import json

# 默认配置
DEFAULT_LLMS_NAME = "qwen2_13b_gguf"
DEFAULT_PORT = 4000


# 后端请求处理函数
def send_request(llms_name, port, llms_input):
    # 如果llms_name为空，则使用默认值
    if not llms_name:
        llms_name = DEFAULT_LLMS_NAME

    # 如果port为空，则使用默认值
    if not port:
        port = DEFAULT_PORT

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


# Gradio界面配置
with gr.Blocks() as demo:
    with gr.Row():
        # 左侧区域
        with gr.Column():
            llms_output = gr.Textbox(label="llms_output", interactive=False)
            llms_input = gr.Textbox(label="llms_input", placeholder="请输入请求内容...")

        # 右侧区域
        with gr.Column():
            llms_name = gr.Textbox(label="llms_name", placeholder="请输入模型名称...")
            port = gr.Textbox(label="port", placeholder="请输入端口...")
            submit_btn = gr.Button("提交")

    # 按钮点击后发送请求并更新llms_output
    submit_btn.click(send_request, inputs=[llms_name, port, llms_input], outputs=llms_output)

# 启动Gradio界面
demo.launch(server_name="0.0.0.0", server_port=6006)
