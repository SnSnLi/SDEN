import torch
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from encoder.clip import CLIPWrapperModel
from sden.emergence import EmergenceModel
from sden.dual import DualEmergenceOptimizer
from sden.topology import DynamicTopologyCoupler

def test_sden():
    try:
        # 初始化模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # 加载编码器
        encoder = CLIPWrapperModel(device=device)
        
        # 测试文本编码
        print("\nTesting text encoding...")
        text_feat = encoder.encode_text("A dog playing in the park")
        print(f"Text feature shape: {text_feat.shape}")
        
        # 测试图像编码 (使用项目中的示例图片)
        image_path = os.path.join(root_dir, "data/flickr30k_entities/raw/images/flickr30k_images/1000092795.jpg")
        print("\nTesting image encoding...")
        image_feat = encoder.encode_image(image_path)
        print(f"Image feature shape: {image_feat.shape}")
        
        # 初始化所有SDEN模块
        dim = 512 # 匹配multi_scale输出维度
        
        # 1. Emergence模块
        emergence_model = EmergenceModel(
            dim=dim,
            text_input_dim=text_feat.shape[-1],
            image_input_dim=image_feat.shape[-1]
        ).to(device)
        
        # 2. Dual模块 
        dual_model = DualEmergenceOptimizer(
            dim=dim,
            feature_dim=512 
        ).to(device)
        
        # 3. Topology模块
        topology_model = DynamicTopologyCoupler(dim=dim).to(device)
        
        # 准备输入数据
        text_input = text_feat.unsqueeze(0).unsqueeze(1).to(device)
        image_input = image_feat.unsqueeze(0).unsqueeze(1).to(device)
        
        # 训练模式设置
        emergence_model.is_training = True
        dual_model.is_training = True
        
        # 优化器
        optimizer = torch.optim.Adam(
            list(emergence_model.parameters()) + 
            list(dual_model.parameters()) + 
            list(topology_model.parameters()),
            lr=1e-4
        )
        
        # 训练循环 (增加epoch并添加更多测试样本)
        print("\nStarting full SDEN training...")
        test_texts = [
            "A dog playing in the park",
            "A red car on the street", 
            "People walking in a shopping mall"
        ]
        test_images = [
            os.path.join(root_dir, "data/flickr30k_entities/raw/images/flickr30k_images/1000092795.jpg"),
            os.path.join(root_dir, "data/flickr30k_entities/raw/images/flickr30k_images/10002456.jpg"),
            os.path.join(root_dir, "data/flickr30k_entities/raw/images/flickr30k_images/1000268201.jpg")
        ]
        
        for epoch in range(10):  # 增加训练epoch到10次
            optimizer.zero_grad()
            
            # 阶段1: Emergence处理
            emergence_outputs = emergence_model(
                text_feat=text_input,
                image_feat=image_input
            )
            
            # 调整维度以匹配Dual模块输入要求
            text_emerged = emergence_outputs[0].mean(dim=1, keepdim=True) if emergence_outputs[0] is not None else None
            image_emerged = emergence_outputs[1].mean(dim=1, keepdim=True) if emergence_outputs[1] is not None else None
            
            # 阶段2: Dual优化
            dual_outputs = dual_model(
                text_features=text_emerged,
                image_features=image_emerged
            )
            
            # 阶段3: Topology耦合
            topology_outputs = topology_model(
                text_feat=dual_outputs[0],
                image_feat=dual_outputs[1]
            )
            
            # 计算总损失(确保是标量)
            total_loss = emergence_outputs[4].mean() + dual_outputs[2].mean()
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            print(f"\nEpoch {epoch+1} results:")
            print("[Emergence]")
            print(f"1. Final text: {emergence_outputs[0].shape}")
            print(f"2. Final image: {emergence_outputs[1].shape}")
            print(f"3. Loss: {emergence_outputs[4].item():.4f}")
            
            print("\n[Dual]")
            print(f"1. Optimized text: {dual_outputs[0].shape}")
            print(f"2. Optimized image: {dual_outputs[1].shape}")
            print(f"3. Consistency loss: {dual_outputs[3].item():.4f}")
            
            print("\n[Topology]")
            print(f"1. Final output: {topology_outputs['output'].shape}")
            print(f"2. Adjacency matrix: {topology_outputs['adj_matrix'].shape}")
            
            # 评估涌现语义质量
            final_text = topology_outputs['text_features'] if topology_outputs['text_features'] is not None else dual_outputs[0]
            final_image = topology_outputs['image_features'] if topology_outputs['image_features'] is not None else dual_outputs[1]
            
            # 1. 计算原始特征相似度
            original_sim = F.cosine_similarity(
                text_input.mean(dim=1),
                image_input.mean(dim=1)
            ).item()
            
            # 2. 计算涌现特征相似度
            emerged_sim = F.cosine_similarity(
                final_text.mean(dim=1),
                final_image.mean(dim=1)
            ).item()
            
            # 3. 计算涌现-原始特征关联度
            text_preserve = F.cosine_similarity(
                final_text.mean(dim=1),
                text_input.mean(dim=1)
            ).item()
            
            image_preserve = F.cosine_similarity(
                final_image.mean(dim=1),
                image_input.mean(dim=1)
            ).item()
            
            # 4. 温度调节的语义对齐度
            temperature = 0.1
            text_norm = final_text / (final_text.norm(dim=-1, keepdim=True) + 1e-8)
            image_norm = final_image / (final_image.norm(dim=-1, keepdim=True) + 1e-8)
            sim_matrix = torch.matmul(text_norm, image_norm.transpose(-2, -1)) / temperature
            alignment = F.softmax(sim_matrix, dim=-1).mean().item()
            
            # 3. 分析注意力权重
            attn_weights = topology_outputs['adj_matrix'].mean(dim=(0,3)).cpu().detach().numpy()
            
            print("\n涌现语义质量评估:")
            print(f"1. 原始特征相似度: {original_sim:.4f}")
            print(f"2. 涌现特征相似度: {emerged_sim:.4f} (Δ{emerged_sim-original_sim:+.4f})")
            print(f"3. 文本特征保留度: {text_preserve:.4f}")
            print(f"4. 图像特征保留度: {image_preserve:.4f}") 
            print(f"5. 语义对齐度: {alignment:.4f}")
            print("6. 注意力权重矩阵:")
            print(attn_weights)
            
           
        print("\n✅ Full SDEN training completed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_sden()
