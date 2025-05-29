import os
import requests
from tqdm import tqdm
import json

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    model_dir = "/opt/maxkb/model/base"
    os.makedirs(model_dir, exist_ok=True)
    
    # 使用镜像站点
    base_url = "https://hf-mirror.com/shibing624/text2vec-base-chinese/resolve/main"
    files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt"
    ]
    
    # 创建必要的配置文件
    config = {
        "architectures": ["BertForMaskedLM"],
        "model_type": "bert",
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "pad_token_id": 0,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": 21128
    }
    
    # 创建 tokenizer.json
    tokenizer_json = {
        "do_lower_case": True,
        "do_basic_tokenize": True,
        "never_split": ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"],
        "tokenizer_class": "BertTokenizer",
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]"
    }
    
    # 保存配置文件
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # 保存 tokenizer.json
    with open(os.path.join(model_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_json, f, indent=2)
    
    # 下载其他文件
    for file in files:
        if file in ["config.json", "tokenizer.json"]:
            continue
        url = f"{base_url}/{file}"
        save_path = os.path.join(model_dir, file)
        if not download_file(url, save_path):
            print(f"Failed to download {file}")
            return False
    
    print("All files downloaded successfully")
    return True

if __name__ == "__main__":
    main()
