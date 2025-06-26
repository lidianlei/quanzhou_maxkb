from typing import List, Dict, Any
from setting.models_provider.base_model_provider import MaxKBBaseModel
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import os
import json
import shutil


class LocalEmbedding(MaxKBBaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 使用用户主目录下的模型目录
        self.user_model_path = os.path.expanduser("~/maxkb_model")
        # 使用serveLocal目录作为源目录
        self.source_model_path = os.path.join("/home/gpu/MaxKB", "serveLocal", "maxkb", "model", "base")

    @staticmethod
    def new_instance(model_type: str, model_name: str, model_credential: Dict[str, object], **model_kwargs) -> 'LocalEmbedding':
        instance = LocalEmbedding()
        instance.model_type = model_type
        instance.model_name = model_name
        instance.model_credential = model_credential
        return instance

    def _copy_model_files(self):
        """复制模型文件到用户目录"""
        try:
            print(f"Creating user model directory: {self.user_model_path}")
            if not os.path.exists(self.user_model_path):
                os.makedirs(self.user_model_path, exist_ok=True)
            
            # 检查源目录是否存在
            if not os.path.exists(self.source_model_path):
                print(f"Source model path does not exist: {self.source_model_path}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Directory contents of parent: {os.listdir(os.path.dirname(self.source_model_path))}")
                raise Exception(f"Source model path does not exist: {self.source_model_path}")
            
            print(f"Source directory contents: {os.listdir(self.source_model_path)}")
            
            # 复制所有文件
            for item in os.listdir(self.source_model_path):
                src_path = os.path.join(self.source_model_path, item)
                dst_path = os.path.join(self.user_model_path, item)
                
                print(f"Copying {src_path} to {dst_path}")
                
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied file: {item}")
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    print(f"Copied directory: {item}")

            print(f"User directory contents after copy: {os.listdir(self.user_model_path)}")
            
            # 确保必要的文件存在
            required_files = [
                'config.json',
                'model.safetensors'
            ]
            
            missing_files = []
            for file in required_files:
                file_path = os.path.join(self.user_model_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
                else:
                    size = os.path.getsize(file_path)
                    print(f"File {file} exists, size: {size} bytes")
            
            if missing_files:
                raise Exception(f"Missing required model files in user directory: {', '.join(missing_files)}")
                
        except Exception as e:
            print(f"Error during file copy: {str(e)}")
            raise

    def _load_model(self):
        if self.model is None:
            try:
                print(f"Loading model from source path: {self.source_model_path}")
                
                # 复制模型文件到用户目录
                self._copy_model_files()
                
                # 加载配置
                config_path = os.path.join(self.user_model_path, 'config.json')
                if not os.path.exists(config_path):
                    raise Exception(f"Config file not found: {config_path}")
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                    print(f"Model config: {config_dict}")
                
                config = BertConfig(**config_dict)
                
                # 创建tokenizer
                self.tokenizer = BertTokenizer(
                    vocab_file=os.path.join(self.user_model_path, 'vocab.txt'),
                    do_lower_case=True,
                    do_basic_tokenize=True,
                    never_split=None,
                    unk_token="[UNK]",
                    sep_token="[SEP]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    mask_token="[MASK]",
                    tokenize_chinese_chars=True,
                    strip_accents=None,
                    model_max_length=512
                )
                
                # 加载模型
                self.model = BertModel(config).from_pretrained(
                    self.user_model_path,
                    local_files_only=True
                ).to(self.device)
                
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print(f"Current directory contents: {os.listdir(self.user_model_path)}")
                raise

    def _mean_pooling(self, model_output, attention_mask):
        """平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 0) / torch.clamp(input_mask_expanded.sum(0), min=1e-9)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            self._load_model()
            print(f"Embedding {len(texts)} texts")
            
            # 对文本进行编码
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # 获取模型输出
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # 进行平均池化
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 转换为numpy并返回
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            print(f"Error in embed_documents: {str(e)}")
            raise Exception(f"Error in embedding: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        try:
            self._load_model()
            print(f"Embedding query: {text[:50]}...")
            
            # 对文本进行编码
            encoded_input = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # 获取模型输出
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # 进行平均池化
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 转换为numpy并返回
            return embeddings.cpu().numpy()[0].tolist()
        except Exception as e:
            print(f"Error in embed_query: {str(e)}")
            raise Exception(f"Error in embedding: {str(e)}")
