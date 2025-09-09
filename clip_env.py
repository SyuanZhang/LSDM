import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMTextGenerator:
    """支持多种LLM的文本生成器"""
    
    def __init__(self, model_type="rule_based", model_name=None, device="cuda"):
        self.model_type = model_type.lower()
        self.device = device
        self.model = None
        self.tokenizer = None
        
        print("🤖 Initializing LLM Text Generator...")
        with tqdm(total=1, desc="Loading LLM components", colour="blue") as pbar:
            if self.model_type == "huggingface":
                self._init_huggingface_model(model_name or "distilgpt2")
            elif self.model_type == "openai":
                self._init_openai()
            elif self.model_type == "claude":
                self._init_claude()
            else:
                print("✅ Using enhanced rule-based text generation")
                self.model_type = "rule_based"
            pbar.update(1)
    
    def _init_huggingface_model(self, model_name):
        """初始化HuggingFace模型"""
        try:
            print(f"📥 Loading HuggingFace model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ HuggingFace model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load HuggingFace model: {e}")
            self.model_type = "rule_based"
    
    def _init_openai(self):
        """初始化OpenAI API"""
        try:
            import openai
            self.openai = openai
            print("✅ OpenAI API initialized (remember to set API key)")
        except ImportError:
            print("❌ OpenAI package not installed: pip install openai")
            self.model_type = "rule_based"
    
    def _init_claude(self):
        """初始化Claude API"""
        try:
            import anthropic
            print("✅ Claude API initialized (remember to set API key)")
        except ImportError:
            print("❌ Anthropic package not installed: pip install anthropic")
            self.model_type = "rule_based"

class POITextGenerator:
    """POI文本生成器"""
    
    def __init__(self, llm_generator: Optional[LLMTextGenerator] = None):
        self.llm_generator = llm_generator
        
        # POI类别名称
        self.poi_categories = [
            "Medical care", "Hotel", "Business affairs", "Life service", 
            "Transportation hub", "Culture", "Sports", "Residence", 
            "Entertainment and leisure", "Scenic spot", "Government", 
            "Factory", "Shopping", "Restaurant", "Education", "Landmark", "Other"
        ]
        
        # 详细描述词汇
        self.poi_descriptions = {
            'Medical care': ['hospital', 'clinic', 'pharmacy', 'medical center', 'healthcare facility'],
            'Hotel': ['hotel', 'accommodation', 'lodge', 'inn', 'resort'],
            'Business affairs': ['office building', 'business center', 'corporate area', 'commercial district'],
            'Life service': ['service center', 'utility office', 'community service', 'public service'],
            'Transportation hub': ['station', 'terminal', 'transport hub', 'transit center', 'airport'],
            'Culture': ['museum', 'gallery', 'cultural center', 'art venue', 'library'],
            'Sports': ['gym', 'sports center', 'stadium', 'fitness facility', 'athletic venue'],
            'Residence': ['residential area', 'housing complex', 'apartment building', 'neighborhood'],
            'Entertainment and leisure': ['entertainment venue', 'leisure center', 'recreation area', 'amusement'],
            'Scenic spot': ['tourist attraction', 'scenic area', 'landmark', 'viewpoint', 'park'],
            'Government': ['government building', 'public office', 'administrative center', 'city hall'],
            'Factory': ['industrial area', 'manufacturing plant', 'factory', 'production facility'],
            'Shopping': ['shopping mall', 'retail store', 'market', 'shopping center', 'commercial area'],
            'Restaurant': ['restaurant', 'dining area', 'food court', 'cafe', 'eatery'],
            'Education': ['school', 'university', 'educational institution', 'campus', 'learning center'],
            'Landmark': ['landmark', 'monument', 'historic site', 'notable building', 'famous location'],
            'Other': ['mixed area', 'general facility', 'unspecified location', 'other venue']
        }
    
    def generate_enhanced_description(self, poi_counts: np.ndarray, user_id=None, time_step=None) -> str:
        """增强版POI描述生成"""
        # 找到主要的POI类别
        top_indices = np.argsort(poi_counts)[-5:][::-1]
        top_values = poi_counts[top_indices]
        
        # 筛选有意义的POI
        significant_pois = [(idx, val) for idx, val in zip(top_indices, top_values) if val > 0]
        
        if not significant_pois:
            return "This area has no significant points of interest."
        
        # 生成描述
        descriptions = []
        total_pois = poi_counts.sum()
        
        for poi_idx, count in significant_pois[:3]:
            category = self.poi_categories[poi_idx]
            desc_words = self.poi_descriptions[category]
            
            if count >= 10:
                prefix = "heavily concentrated with"
            elif count >= 5:
                prefix = "moderately populated with"
            elif count >= 1:
                prefix = "has some"
            else:
                continue
            
            if count == 1:
                desc = f"{prefix} 1 {desc_words[0]} facility"
            else:
                desc = f"{prefix} {int(count)} {desc_words[0]} facilities"
            descriptions.append(desc)
        
        # 组合描述
        if len(descriptions) == 0:
            text = "This area has minimal development."
        elif len(descriptions) == 1:
            text = f"This area {descriptions[0]}."
        elif len(descriptions) == 2:
            text = f"This area {descriptions[0]} and {descriptions[1]}."
        else:
            text = f"This area {', '.join(descriptions[:-1])}, and {descriptions[-1]}."
        
        # 添加区域特征
        if total_pois >= 30:
            text += " This represents a highly developed urban area with dense infrastructure."
        elif total_pois >= 15:
            text += " This indicates a moderately developed area with good amenities."
        elif total_pois >= 5:
            text += " This shows a developing area with basic facilities."
        else:
            text += " This appears to be a quiet area with limited development."
        
        return text
    
    def generate_text_description(self, poi_counts: np.ndarray, user_id=None, time_step=None) -> str:
        """生成文本描述"""
        if self.llm_generator and self.llm_generator.model_type != "rule_based":
            return self._generate_with_llm(poi_counts, user_id, time_step)
        else:
            return self.generate_enhanced_description(poi_counts, user_id, time_step)
    
    def _generate_with_llm(self, poi_counts: np.ndarray, user_id=None, time_step=None) -> str:
        """使用LLM生成描述"""
        try:
            # 构建提示
            poi_info = []
            for i, count in enumerate(poi_counts):
                if count > 0:
                    poi_info.append(f"{self.poi_categories[i]}: {int(count)}")
            
            if not poi_info:
                return "This area has no significant points of interest."
            
            prompt = f"Describe this urban area based on the following facilities: {', '.join(poi_info)}. Write 2-3 sentences:"
            
            if self.llm_generator.model_type == "huggingface":
                return self._generate_huggingface(prompt)
            elif self.llm_generator.model_type == "openai":
                return self._generate_openai(prompt)
            elif self.llm_generator.model_type == "claude":
                return self._generate_claude(prompt)
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return self.generate_enhanced_description(poi_counts, user_id, time_step)
    
    def _generate_huggingface(self, prompt: str) -> str:
        """HuggingFace生成"""
        try:
            inputs = self.llm_generator.tokenizer.encode(prompt, return_tensors="pt", max_length=200, truncation=True)
            inputs = inputs.to(self.llm_generator.device)
            
            with torch.no_grad():
                outputs = self.llm_generator.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 60,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm_generator.tokenizer.eos_token_id
                )
            
            response = self.llm_generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(self.llm_generator.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            # 清理输出
            sentences = generated.split('.')
            if len(sentences) > 3:
                generated = '. '.join(sentences[:3]) + '.'
            
            return generated if generated else self.generate_enhanced_description([])
        except Exception as e:
            logger.warning(f"HuggingFace generation error: {e}")
            return self.generate_enhanced_description([])
    
    def generate_all_descriptions(self, poi_matrix):
        """为所有用户的所有时间步生成文本描述"""
        descriptions = []
        total_points = poi_matrix.shape[0] * poi_matrix.shape[2]
        
        # 创建总进度条
        with tqdm(total=total_points, desc="🔤 Generating text descriptions", 
                  unit="points", colour="green") as pbar:
            
            for user_id in range(poi_matrix.shape[0]):
                user_descriptions = []
                
                # 用户级别的进度信息
                pbar.set_postfix({
                    'User': f"{user_id + 1}/{poi_matrix.shape[0]}",
                    'Current': f"Processing user {user_id + 1}"
                })
                
                for time_step in range(poi_matrix.shape[2]):
                    poi_counts = poi_matrix[user_id, :, time_step]
                    text = self.generate_text_description(poi_counts, user_id, time_step)
                    user_descriptions.append(text)
                    
                    # 更新进度条
                    pbar.update(1)
                    
                    # 每50个点显示一次详细信息
                    if (time_step + 1) % 50 == 0:
                        pbar.set_postfix({
                            'User': f"{user_id + 1}/{poi_matrix.shape[0]}",
                            'Point': f"{time_step + 1}/{poi_matrix.shape[2]}"
                        })
                
                descriptions.append(user_descriptions)
                
        return descriptions

class CLIPEnvironmentEncoder(nn.Module):
    """CLIP环境编码器"""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, output_dim=128):
        super(CLIPEnvironmentEncoder, self).__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print("🤖 Initializing CLIP model...")
        with tqdm(total=2, desc="Loading CLIP components", colour="blue") as pbar:
            try:
                self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                pbar.update(1)
                pbar.set_postfix({'Component': 'CLIP model loaded'})
                
                self.processor = CLIPProcessor.from_pretrained(model_name)
                pbar.update(1)
                pbar.set_postfix({'Component': 'CLIP processor loaded'})
                
            except Exception as e:
                print(f"❌ Failed to load CLIP model {model_name}: {e}")
                # 尝试备用模型
                try:
                    backup_model = "openai/clip-vit-base-patch16"
                    print(f"🔄 Trying backup model: {backup_model}")
                    self.model = CLIPModel.from_pretrained(backup_model).to(self.device)
                    self.processor = CLIPProcessor.from_pretrained(backup_model)
                    pbar.update(2)
                    print("✅ Backup CLIP model loaded successfully")
                except Exception as e2:
                    raise RuntimeError(f"❌ Failed to load any CLIP model: {e2}")
        
        # 获取嵌入维度
        self.clip_dim = self.model.config.projection_dim
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(self.device)
        
        # 可调参数
        self.alpha = nn.Parameter(torch.tensor(0.5, device=self.device))  # 图像权重
        self.beta = nn.Parameter(torch.tensor(0.5, device=self.device))   # 文本权重
        
        self.embedding_dim = output_dim
        print(f"✅ CLIP encoder initialized! Embedding dimension: {self.embedding_dim}")
        
        # 冻结CLIP参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def encode_image_batch(self, images: List[Optional[Image.Image]]) -> torch.Tensor:
        """批量编码图像"""
        batch_size = len(images)
        embeddings = torch.zeros(batch_size, self.clip_dim, device=self.device, dtype=torch.float32)
        
        # 找到有效图像
        valid_images = []
        valid_indices = []
        
        for i, image in enumerate(images):
            if image is not None and hasattr(image, 'size') and image.size[0] > 0 and image.size[1] > 0:
                valid_images.append(image)
                valid_indices.append(i)
        
        if valid_images:
            try:
                inputs = self.processor(images=valid_images, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features.float()
                
                for i, idx in enumerate(valid_indices):
                    embeddings[idx] = image_features[i]
                    
            except Exception as e:
                logger.warning(f"Batch image encoding failed: {e}")
                # 逐个处理
                for i, idx in enumerate(valid_indices):
                    try:
                        inputs = self.processor(images=[valid_images[i]], return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            image_features = self.model.get_image_features(**inputs)
                        embeddings[idx] = image_features[0].float()
                    except Exception as e2:
                        logger.warning(f"Individual image encoding failed: {e2}")
        
        return embeddings
    
    def encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        """批量编码文本"""
        try:
            inputs = self.processor(
                text=texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=77
            ).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features.float()
            return text_features
            
        except Exception as e:
            logger.warning(f"Batch text encoding failed: {e}")
            batch_size = len(texts)
            embeddings = torch.zeros(batch_size, self.clip_dim, device=self.device, dtype=torch.float32)
            
            for i, text in enumerate(texts):
                try:
                    inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
                    with torch.no_grad():
                        text_features = self.model.get_text_features(**inputs)
                    embeddings[i] = text_features[0].float()
                except Exception as e2:
                    logger.warning(f"Individual text encoding failed: {e2}")
            return embeddings
    
    def forward(self, images: List[Optional[Image.Image]], texts: List[str]) -> torch.Tensor:
        """前向传播"""
        # 获取CLIP特征
        z_I = self.encode_image_batch(images)
        z_T = self.encode_text_batch(texts)
        
        # 组合特征
        z_combined = self.alpha * z_I + self.beta * z_T
        
        # 投影到目标维度
        z_env = self.projection(z_combined)
        
        return z_env
    
    def encode_batch(self, image_batch: List[Optional[Image.Image]], text_batch: List[str], batch_size=32):
        """批量编码图像-文本对"""
        all_embeddings = []
        
        total_batches = (len(image_batch) + batch_size - 1) // batch_size
        
        with tqdm(total=total_batches, desc="🧠 Encoding multimodal batches", 
                  unit="batch", colour="yellow") as pbar:
            
            for i in range(0, len(image_batch), batch_size):
                batch_images = image_batch[i:i+batch_size]
                batch_texts = text_batch[i:i+batch_size]
                
                pbar.set_postfix({
                    'Batch size': len(batch_images),
                    'Progress': f"{i + len(batch_images)}/{len(image_batch)} pairs"
                })
                
                with torch.no_grad():
                    embeddings = self.forward(batch_images, batch_texts)
                    all_embeddings.append(embeddings.detach())
                
                pbar.update(1)
                
                # GPU内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                time.sleep(0.01)  # 显示效果
        
        return torch.cat(all_embeddings, dim=0)

class EnvironmentEmbeddingExtractor:
    """环境嵌入提取器"""
    
    def __init__(self, 
                 poi_file_path: str,
                 image_folder_path: str,
                 llm_type: str = "rule_based",
                 llm_model_name: str = None,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 device: str = None,
                 batch_size: int = 16,
                 output_dim: int = 128):
        
        self.poi_file_path = poi_file_path
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = output_dim
        
        print("🔧 Initializing Environment Embedding Extractor...")
        
        # 初始化LLM生成器
        if llm_type != "rule_based":
            try:
                self.llm_generator = LLMTextGenerator(llm_type, llm_model_name, self.device)
            except Exception as e:
                print(f"⚠️ LLM initialization failed: {e}")
                self.llm_generator = None
        else:
            self.llm_generator = None
        
        # 初始化POI文本生成器
        self.poi_text_generator = POITextGenerator(self.llm_generator)
        
        # 初始化CLIP编码器
        self.clip_encoder = CLIPEnvironmentEncoder(clip_model_name, self.device, output_dim)
        
        print("✅ All components initialized successfully!")
    
    def analyze_poi_distribution(self, poi_data):
        """分析POI数据分布"""
        print("\n" + "="*50)
        print("📊 POI Data Analysis")
        print("="*50)
        
        print(f"Shape: {poi_data.shape}")
        print(f"Mean: {poi_data.mean():.4f}")
        print(f"Std: {poi_data.std():.4f}")
        print(f"Min: {poi_data.min():.4f}")
        print(f"Max: {poi_data.max():.4f}")
        
        print("\nPer-category statistics:")
        with tqdm(self.poi_text_generator.poi_categories, desc="Analyzing categories", colour="cyan") as pbar:
            for i, category in enumerate(pbar):
                category_data = poi_data[:, i, :]
                pbar.set_postfix({
                    'Category': category[:20] + "..." if len(category) > 20 else category
                })
                print(f"  {category}: mean={category_data.mean():.4f}, std={category_data.std():.4f}")
                time.sleep(0.05)
    
    def load_poi_data(self) -> Tuple[np.ndarray, Dict[int, List[str]]]:
        """加载POI数据并生成文本描述"""
        print("📁 Loading POI data...")
        
        if not os.path.exists(self.poi_file_path):
            raise FileNotFoundError(f"❌ POI file not found: {self.poi_file_path}")
        
        data = np.load(self.poi_file_path)
        poi_matrix = data['poi_matrix']  # shape: (871, 17, 168)
        
        print(f"✅ POI data loaded! Shape: {poi_matrix.shape}")
        
        # 分析数据分布
        self.analyze_poi_distribution(poi_matrix)
        
        # 生成文本描述
        all_descriptions = self.poi_text_generator.generate_all_descriptions(poi_matrix)
        
        # 转换为字典格式
        text_descriptions = {}
        for user_id, user_desc in enumerate(all_descriptions):
            text_descriptions[user_id] = user_desc
        
        return poi_matrix, text_descriptions
    
    def load_image_safely(self, image_path: str) -> Optional[Image.Image]:
        """安全加载图像"""
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                if image.size[0] > 0 and image.size[1] > 0:
                    return image
            return None
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def create_image_text_pairs(self, text_descriptions: Dict[int, List[str]]) -> Dict[int, List[Dict[str, Any]]]:
        """创建图像-文本配对数据"""
        print("📸 Creating image-text pairs...")
        paired_data = {}
        
        total_users = len(text_descriptions)
        with tqdm(total=total_users, desc="🔗 Processing image-text pairs", 
                  unit="users", colour="purple") as pbar:
            
            for user_id in range(871):
                user_folder = os.path.join(self.image_folder_path, f"user_{user_id}")
                paired_data[user_id] = []
                
                pbar.set_postfix({
                    'User': f"{user_id + 1}/871",
                    'Folder': f"user_{user_id}"
                })
                
                for time_step in range(168):
                    text = text_descriptions[user_id][time_step]
                    
                    # 寻找对应图像
                    image = None
                    if os.path.exists(user_folder):
                        try:
                            for filename in os.listdir(user_folder):
                                if filename.startswith(f"point_{time_step}_") and filename.endswith('.png'):
                                    image_path = os.path.join(user_folder, filename)
                                    image = self.load_image_safely(image_path)
                                    if image:
                                        break
                        except Exception as e:
                            logger.warning(f"Error accessing folder {user_folder}: {e}")
                    
                    paired_data[user_id].append({
                        'text': text,
                        'image': image,
                        'time_step': time_step
                    })
                
                pbar.update(1)
        
        print("✅ Image-text pairs created successfully!")
        return paired_data
    
    def extract_embeddings_optimized(self, paired_data: Dict[int, List[Dict[str, Any]]]) -> np.ndarray:
        """优化的嵌入提取"""
        print("🧠 Starting embedding extraction...")
        embeddings = np.zeros((871, 168, self.output_dim), dtype=np.float32)
        
        # 准备批处理数据
        all_images = []
        all_texts = []
        all_indices = []
        
        print("📦 Preparing data for batch processing...")
        with tqdm(total=871*168, desc="Preparing data", colour="orange") as prep_pbar:
            for user_id in range(871):
                for time_step in range(168):
                    data_point = paired_data[user_id][time_step]
                    all_images.append(data_point['image'])
                    all_texts.append(data_point['text'])
                    all_indices.append((user_id, time_step))
                    prep_pbar.update(1)
        
        print(f"📊 Total samples prepared: {len(all_images)}")
        
        # 批量处理
        print("🚀 Starting batch encoding...")
        with torch.no_grad():
            batch_embeddings = self.clip_encoder.encode_batch(
                all_images, all_texts, batch_size=self.batch_size
            )
        
        # 重新组织数据
        print("📋 Reorganizing embeddings...")
        with tqdm(total=len(all_indices), desc="Organizing results", colour="green") as org_pbar:
            for i, (user_id, time_step) in enumerate(all_indices):
                embeddings[user_id, time_step, :] = batch_embeddings[i].cpu().numpy()
                org_pbar.update(1)
                
                if (i + 1) % 10000 == 0:
                    org_pbar.set_postfix({'Processed': f"{i + 1}/{len(all_indices)}"})
        
        print(f"✅ Embedding extraction completed! Shape: {embeddings.shape}")
        return embeddings
    
    def save_results(self, embeddings: np.ndarray, text_descriptions: Dict[int, List[str]], 
                    poi_matrix: np.ndarray, output_path: str = "environment_embeddings"):
        """保存结果"""
        print("\n" + "="*50)
        print("💾 Saving Results")
        print("="*50)
        
        save_tasks = ["NPZ file", "JSON file"]
        with tqdm(save_tasks, desc="Saving files", colour="green") as save_pbar:
            # 保存NPZ
            save_pbar.set_postfix({'File': f'{output_path}.npz'})
            np.savez_compressed(
                f'{output_path}.npz',
                embeddings=embeddings,
                poi_matrix=poi_matrix,
                descriptions=np.array([text_descriptions[i] for i in range(871)], dtype=object),
                categories=np.array(self.poi_text_generator.poi_categories),
                shape_info={
                    'users': 871,
                    'time_steps': 168,
                    'embedding_dim': embeddings.shape[2],
                    'poi_categories': len(self.poi_text_generator.poi_categories)
                },
                metadata={
                    'alpha': float(self.clip_encoder.alpha.item()),
                    'beta': float(self.clip_encoder.beta.item()),
                    'device': self.device,
                    'batch_size': self.batch_size,
                    'output_dim': self.output_dim
                }
            )
            save_pbar.update(1)
            
            # 保存JSON
            save_pbar.set_postfix({'File': f'{output_path}_descriptions.json'})
            json_data = {
                'descriptions': {str(k): v for k, v in text_descriptions.items()},
                'categories': self.poi_text_generator.poi_categories,
                'metadata': {
                    'total_users': 871,
                    'total_time_steps': 168,
                    'poi_dimensions': poi_matrix.shape[1],
                    'embedding_dimension': embeddings.shape[2],
                    'clip_model': 'openai/clip-vit-base-patch32',
                    'alpha': float(self.clip_encoder.alpha.item()),
                    'beta': float(self.clip_encoder.beta.item())
                }
            }
            
            with open(f'{output_path}_descriptions.json', 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            save_pbar.update(1)
        
        print("✅ All files saved successfully!")
    
    def run_complete_pipeline(self, output_path: str = "environment_embeddings") -> np.ndarray:
        """运行完整管道"""
        print("🚀 Starting Environment Embedding Extraction Pipeline")
        print("="*60)
        
        # 总体进度跟踪
        total_steps = 5
        overall_progress = tqdm(total=total_steps, desc="📋 Overall Progress", 
                             colour="magenta", position=0)
        
        try:
            # 步骤1: 加载POI数据
            overall_progress.set_description("📁 Loading POI data...")
            poi_matrix, text_descriptions = self.load_poi_data()
            overall_progress.update(1)
            
            # 步骤2: 创建图像-文本配对
            overall_progress.set_description("🔗 Creating image-text pairs...")
            paired_data = self.create_image_text_pairs(text_descriptions)
            overall_progress.update(1)
            
            # 显示样本
            print("\n" + "="*50)
            print("📝 Sample Descriptions")
            print("="*50)
            for i in range(min(2, 871)):
                print(f"\n👤 User {i + 1} (first 3 points):")
                for j in range(min(3, len(text_descriptions[i]))):
                    print(f"  📍 Point {j + 1}: {text_descriptions[i][j]}")
            
            # 步骤3: 提取嵌入
            overall_progress.set_description("🧠 Extracting embeddings...")
            embeddings = self.extract_embeddings_optimized(paired_data)
            overall_progress.update(1)
            
            # 步骤4: 保存结果
            overall_progress.set_description("💾 Saving results...")
            self.save_results(embeddings, text_descriptions, poi_matrix, output_path)
            overall_progress.update(1)
            
            # 步骤5: 完成
            overall_progress.set_description("✅ Pipeline completed!")
            overall_progress.update(1)
            overall_progress.close()
            
            print("\n🎉 Pipeline completed successfully!")
            print("="*60)
            print("📁 Generated files:")
            print(f"  • {output_path}.npz - Contains embeddings and metadata")
            print(f"  • {output_path}_descriptions.json - Contains text descriptions")
            print(f"📊 Summary:")
            print(f"  • Users processed: {poi_matrix.shape[0]}")
            print(f"  • Time steps per user: {poi_matrix.shape[2]}")
            print(f"  • Total descriptions generated: {871 * 168}")
            print(f"  • Embedding dimension: {embeddings.shape[2]}")
            
            return embeddings
            
        except Exception as e:
            overall_progress.close()
            print(f"❌ Pipeline failed: {e}")
            raise

def main():
    """主函数"""
    print("🚀 Starting Environment Embedding Extraction")
    print("="*60)
    
    # 配置参数
    config = {
        'poi_file_path': 'poi_data.npz',  # 替换为你的POI文件路径
        'image_folder_path': 'user_satellite_images',  # 替换为你的图像文件夹路径
        'llm_type': 'rule_based',  # 选项: 'rule_based', 'huggingface', 'openai', 'claude'
        'llm_model_name': 'distilgpt2',  # 仅当llm_type='huggingface'时使用
        'clip_model_name': 'openai/clip-vit-base-patch32',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 16,  # 根据GPU内存调整
        'output_dim': 128,  # 输出嵌入维度
        'output_path': 'environment_embeddings'
    }
    
    print("⚙️ Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # GPU信息
    if torch.cuda.is_available():
        print(f"🔥 CUDA Device: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        torch.backends.cudnn.benchmark = True
    else:
        print("💻 Using CPU")
    
    # 创建提取器并运行
    try:
        # 检查文件是否存在
        if not os.path.exists(config['poi_file_path']):
            print("⚠️ POI file not found, creating sample data...")
            sample_poi = np.random.rand(871, 17, 168)
            np.savez('poi_data.npz', poi_matrix=sample_poi)
            print("✅ Sample POI data created!")
        
        extractor = EnvironmentEmbeddingExtractor(
            **{k: v for k, v in config.items() if k != 'output_path'}
        )
        
        embeddings = extractor.run_complete_pipeline(config['output_path'])
        
        print(f"\n🎯 Final embeddings shape: {embeddings.shape}")
        return extractor, embeddings
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行主程序
    try:
        extractor, embeddings = main()
        print("\n✅ All tasks completed successfully!")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
