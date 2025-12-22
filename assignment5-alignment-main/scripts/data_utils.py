import json
import os
import logging
import pathlib
from typing import List, Dict

logger = logging.getLogger(__name__)

def load_r1_zero_prompt() -> str:
    prompt_path = pathlib.Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def format_r1_zero_prompt(question: str) -> str:
    prompt_template = load_r1_zero_prompt()
    return prompt_template.format(question=question)

def convert_gsm8k_to_r1_format(input_path, output_path):
    """
    将 GSM8K 格式转换为 R1-Zero 推理格式。
    GSM8K 原始格式: "reasoning steps... #### final_answer"
    目标格式: "<think> reasoning steps... </think> <answer> final_answer </answer>"
    """

    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            try:
                item = json.loads(line)
                
                # 1. 获取原始问题和答案
                question = item.get('question', '')
                original_answer = item.get('answer', '')
                
                # 2. 根据 GSM8K 的分隔符 #### 拆分推理过程和答案
                if "####" not in original_answer:
                    continue # 跳过格式错误的数据
                
                # split(..., 1) 确保只切分最后一部分作为答案
                reasoning, final_answer = original_answer.split("####", 1)
                
                # 3. 清洗空白字符
                reasoning = reasoning.strip()
                final_answer = final_answer.strip()
                
                # 4. 构建符合 R1-Zero 要求的结构化输出 
                # 格式: <think> reasoning </think> <answer> answer </answer>
                structured_response = f"<think>\n{reasoning}\n</think>\n<answer> {final_answer} </answer>"
                
                # 5. 构建 SFT 数据条目 
                output_obj = {
                    "prompt": format_r1_zero_prompt(question),
                    "response": structured_response
                }
                
                fout.write(json.dumps(output_obj) + '\n')
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing line: {e}")

    print(f"转换完成。共处理 {processed_count} 条数据。")
    print(f"输出文件: {output_path}")
    
    
if __name__ == "__main__":
    input_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "gsm8k" / "train.jsonl"
    output_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "gsm8k" / "train_convert.jsonl"
    convert_gsm8k_to_r1_format(input_path, output_path)