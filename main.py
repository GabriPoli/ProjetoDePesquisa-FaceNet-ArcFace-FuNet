import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import numpy as np
import random
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

################################################################################
# CONFIGURAÇÕES OTIMIZADAS - 30 ÉPOCAS
################################################################################

# Hiperparâmetros Otimizados para 30 Épocas
LEARNING_RATE = 0.0001
BATCH_SIZE = 4
EPOCHS = 30
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.5

# Inicialização da MTCNN
mtcnn = MTCNN(
    image_size=224,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False
)

# Transformação para padronizar a imagem
transform_padronizacao = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Data Augmentation mais agressiva
transform_aumento_forte = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Verifica dispositivo
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {DEVICE}")

# Configurações de performance
if t.cuda.is_available():
    t.backends.cudnn.benchmark = True
    t.backends.cudnn.deterministic = False

################################################################################
# LOSS FUNCTIONS AVANÇADAS - CORRIGIDAS
################################################################################

class FocalLoss(nn.Module):
    """Focal Loss para lidar com desbalanceamento de classes"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = t.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CenterLoss(nn.Module):
    """Center Loss para features mais discriminativas - CORRIGIDO"""
    def __init__(self, num_classes=2, feat_dim=256):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # CORREÇÃO: Usar device correto
        self.centers = nn.Parameter(t.randn(self.num_classes, self.feat_dim).to(DEVICE))
    
    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = t.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  t.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        
        classes = t.arange(self.num_classes).long().to(DEVICE)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss

################################################################################
# IMPLEMENTAÇÃO DO ARCFACE E FACE NET
################################################################################

class ArcFaceModel(nn.Module):
    """Implementação simplificada do ArcFace baseada em ResNet"""
    def __init__(self, embedding_size=512, num_classes=None):
        super(ArcFaceModel, self).__init__()
        # Usar ResNet50 como backbone (similar às implementações ArcFace)
        self.backbone = models.resnet50(pretrained=True)
        
        # Remover a última camada fully connected
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Camada de embedding
        self.embedding = nn.Linear(2048, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        
        # Se fornecido número de classes, adicionar cabeça de classificação ArcFace
        if num_classes:
            self.classifier = ArcFaceHead(embedding_size, num_classes)
        else:
            self.classifier = None
            
    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embedding = self.bn(self.embedding(features))
        
        if self.classifier is not None and labels is not None:
            output = self.classifier(embedding, labels)
            return output, embedding
        else:
            return F.normalize(embedding, p=2, dim=1)

class ArcFaceHead(nn.Module):
    """Cabeça ArcFace para classificação"""
    def __init__(self, embedding_size, num_classes, s=30.0, m=0.5):
        super(ArcFaceHead, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        
        self.W = nn.Parameter(t.Tensor(embedding_size, num_classes))
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, embedding, labels):
        # Normalizar embeddings e pesos
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        W_norm = F.normalize(self.W, p=2, dim=0)
        
        # Calcular cosseno similarity
        cosine = embedding_norm.mm(W_norm)
        
        # Aplicar margem ArcFace
        if labels is not None:
            one_hot = F.one_hot(labels, self.num_classes).float()
            theta = t.acos(t.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            target_theta = theta + self.m
            cosine_modified = t.cos(target_theta)
            output = cosine + one_hot * (cosine_modified - cosine)
            output *= self.s
            return output
        else:
            return cosine * self.s

# Inicialização dos modelos de reconhecimento facial
print("🔄 Carregando modelos de reconhecimento facial...")

# FaceNet (já existente)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ArcFace
arcface_model = ArcFaceModel(embedding_size=512).eval().to(DEVICE)

# Carregar pesos pré-treinados para ArcFace (se disponível)
try:
    # Tentar carregar pesos pré-treinados
    arcface_checkpoint = t.load('arcface_pretrained.pth', map_location=DEVICE)
    arcface_model.load_state_dict(arcface_checkpoint)
    print("✅ ArcFace: Pesos pré-treinados carregados")
except:
    print("⚠️  ArcFace: Usando pesos ImageNet (fine-tuning recomendado)")

def gerar_embedding_facenet(face_tensor):
    """Gera embedding usando FaceNet"""
    with t.no_grad():
        # FaceNet espera entrada 160x160
        if face_tensor.shape[2:] != (160, 160):
            face_resized = F.interpolate(face_tensor, size=(160, 160), mode='bilinear', align_corners=False)
        else:
            face_resized = face_tensor
            
        embedding = facenet_model(face_resized)
        return F.normalize(embedding, p=2, dim=1)

def gerar_embedding_arcface(face_tensor):
    """Gera embedding usando ArcFace"""
    with t.no_grad():
        # ArcFace geralmente usa 112x112
        if face_tensor.shape[2:] != (112, 112):
            face_resized = F.interpolate(face_tensor, size=(112, 112), mode='bilinear', align_corners=False)
        else:
            face_resized = face_tensor
            
        embedding = arcface_model(face_resized)
        return F.normalize(embedding, p=2, dim=1)

################################################################################
# FUNÇÕES DE PRÉ-PROCESSAMENTO E GNN
################################################################################

def extrair_face_mtcnn(imagem_ou_frame):
    """ Módulo 1: Usa a MTCNN para detectar e isolar a face. """
    try:
        face_tensor = mtcnn(imagem_ou_frame, return_prob=False)
        if face_tensor is not None:
            face_array = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            face_pil = Image.fromarray(face_array)
            face_padronizada = transform_padronizacao(face_pil)
            return face_padronizada.unsqueeze(0).to(DEVICE)
        else:
            return None
    except Exception as e:
        return None

def dividir_em_patches(img_tensor):
    """
    Divide a imagem em patches. 
    Entrada: (B, 3, 224, 224). Saída: (B, 49, 3, 32, 32).
    """
    batch_size, channels, height, width = img_tensor.shape
    patch_size = 32
    step = 32
    
    # Para batch processing
    patches = img_tensor.unfold(2, patch_size, step).unfold(3, patch_size, step)
    patches = patches.contiguous().view(batch_size, -1, channels, patch_size, patch_size)
    
    return patches.to(img_tensor.device)

def converter_para_vetores(patches):
    """
    IMPLEMENTAÇÃO REAL - Substitui t.rand() por CNN leve
    Entrada: (B, 49, 3, 32, 32). Saída: (B, 49, 64).
    """
    batch_size, num_patches = patches.shape[0], patches.shape[1]
    
    # CNN leve para extrair features dos patches (definida uma vez)
    if not hasattr(converter_para_vetores, 'patch_cnn'):
        converter_para_vetores.patch_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64)  # Output: 64 features por patch
        ).to(DEVICE)
    
    # Processa todos os patches em paralelo
    patches_flat = patches.view(-1, 3, 32, 32)  # (B*49, 3, 32, 32)
    features = converter_para_vetores.patch_cnn(patches_flat)  # (B*49, 64)
    
    return features.view(batch_size, num_patches, -1)  # (B, 49, 64)

def construir_grafo_knn(nodes):
    """
    IMPLEMENTAÇÃO OTIMIZADA - Calcula similaridade real entre nós
    Entrada: (B, 49, 64). Saída: (B, 49, 49).
    """
    batch_size, num_nodes, feature_dim = nodes.shape
    
    # Normalizar features para cálculo de similaridade
    nodes_norm = F.normalize(nodes, p=2, dim=2)
    
    # Matriz de similaridade cosseno (mais rápido que distância euclidiana)
    similarity = t.bmm(nodes_norm, nodes_norm.transpose(1, 2))  # (B, 49, 49)
    
    # Aplicar limiar e tornar simétrica
    adjacency = (similarity > 0.3).float()
    adjacency = (adjacency + adjacency.transpose(1, 2)) / 2
    
    return adjacency

################################################################################
# DATASET MELHORADO COM EMBEDDINGS E BALANCEAMENTO
################################################################################

class DeepFakeDatasetComEmbeddings(Dataset):
    def __init__(self, pasta_imagens, transform=None, usar_aumento=False, precompute_embeddings=True):
        """
        Dataset melhorado com suporte a embeddings de FaceNet e ArcFace
        """
        self.transform = transform
        self.usar_aumento = usar_aumento
        self.precompute_embeddings = precompute_embeddings
        self.imagens = []
        self.rotulos = []
        self.embeddings_facenet = []
        self.embeddings_arcface = []
        self.identidades = []  # Para rastrear identidades diferentes
        
        # Mapeamento de pastas para rótulos
        pastas = {
            'fake': 0,      # 0 = Fake
            'real': 1       # 1 = Real
        }
        
        # Mapeamento de identidades
        identidade_counter = 0
        identidade_map = {}
        
        for pasta, rotulo in pastas.items():
            caminho_pasta = os.path.join(pasta_imagens, pasta)
            if os.path.exists(caminho_pasta):
                for arquivo in os.listdir(caminho_pasta):
                    if arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                        caminho_completo = os.path.join(caminho_pasta, arquivo)
                        self.imagens.append(caminho_completo)
                        self.rotulos.append(rotulo)
                        
                        # Extrair identidade do nome do arquivo (assumindo formato: identidade_*.jpg)
                        identidade = arquivo.split('_')[0] if '_' in arquivo else arquivo.split('.')[0]
                        if identidade not in identidade_map:
                            identidade_map[identidade] = identidade_counter
                            identidade_counter += 1
                        self.identidades.append(identidade_map[identidade])
        
        print(f"📁 Dataset carregado: {len(self.imagens)} imagens")
        print(f"   Fakes: {sum(1 for r in self.rotulos if r == 0)}, Reais: {sum(1 for r in self.rotulos if r == 1)}")
        print(f"   Identidades únicas: {identidade_counter}")
        
        # Pré-computar embeddings se solicitado
        if precompute_embeddings:
            print("🔄 Pré-computando embeddings...")
            self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pré-computa embeddings para todas as imagens"""
        for i, caminho_imagem in enumerate(self.imagens):
            try:
                imagem = Image.open(caminho_imagem).convert('RGB')
                face_tensor = extrair_face_mtcnn(imagem)
                
                if face_tensor is not None:
                    # Gerar embeddings
                    embedding_facenet = gerar_embedding_facenet(face_tensor)
                    embedding_arcface = gerar_embedding_arcface(face_tensor)
                    
                    self.embeddings_facenet.append(embedding_facenet.cpu())
                    self.embeddings_arcface.append(embedding_arcface.cpu())
                else:
                    # Usar embedding zero se não detectar face
                    self.embeddings_facenet.append(t.zeros(1, 512))
                    self.embeddings_arcface.append(t.zeros(1, 512))
                    
            except Exception as e:
                print(f"Erro ao processar {caminho_imagem}: {e}")
                self.embeddings_facenet.append(t.zeros(1, 512))
                self.embeddings_arcface.append(t.zeros(1, 512))
        
        print("✅ Embeddings pré-computados")
    
    def __len__(self):
        return len(self.imagens)
    
    def __getitem__(self, idx):
        try:
            # Carregar imagem
            imagem = Image.open(self.imagens[idx]).convert('RGB')
            
            # Tentar extrair face
            face_tensor = extrair_face_mtcnn(imagem)
            
            if face_tensor is None:
                # Se não detectar face, usar imagem completa
                if self.usar_aumento and self.transform:
                    imagem_transformada = self.transform(imagem)
                else:
                    imagem_redimensionada = imagem.resize((224, 224))
                    imagem_transformada = transform_padronizacao(imagem_redimensionada)
                face_tensor = imagem_transformada
            else:
                face_tensor = face_tensor.squeeze(0)
            
            rotulo = self.rotulos[idx]
            identidade = self.identidades[idx]
            
            # Retornar embeddings se pré-computados
            if self.precompute_embeddings and len(self.embeddings_facenet) > idx:
                embedding_facenet = self.embeddings_facenet[idx].to(DEVICE)
                embedding_arcface = self.embeddings_arcface[idx].to(DEVICE)
                return face_tensor, rotulo, embedding_facenet, embedding_arcface
            else:
                return face_tensor, rotulo, t.zeros(1, 512), t.zeros(1, 512)
        
        except Exception as e:
            print(f"Erro ao carregar {self.imagens[idx]}: {e}")
            # Retornar dados dummy em caso de erro
            dummy_image = t.rand(3, 224, 224)
            return dummy_image, 0, t.zeros(1, 512), t.zeros(1, 512)

def calcular_pesos_classes(dataset):
    """Calcula pesos para balanceamento de classes"""
    rotulos = dataset.rotulos
    contagem_fake = sum(1 for r in rotulos if r == 0)
    contagem_real = sum(1 for r in rotulos if r == 1)
    
    total = len(rotulos)
    peso_fake = total / (2 * contagem_fake) if contagem_fake > 0 else 1.0
    peso_real = total / (2 * contagem_real) if contagem_real > 0 else 1.0
    
    print(f"📊 Balanceamento - Fakes: {contagem_fake}, Reais: {contagem_real}")
    print(f"   Pesos - Fake: {peso_fake:.4f}, Real: {peso_real:.4f}")
    
    return t.tensor([peso_fake, peso_real]).to(DEVICE)

################################################################################
# FUNET MELHORADA COM SUPORTE A EMBEDDINGS - CORRIGIDA
################################################################################

class MiniGNNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size).to(DEVICE)

    def forward(self, nodes, adjacency_matrix):
        aggregated = t.bmm(adjacency_matrix, nodes) # Agregação de vizinhos
        updated = self.linear(aggregated) # Transformação linear
        updated = F.relu(updated)
        global_features = updated.mean(dim=1)
        return global_features

class FuNetComEmbeddings(nn.Module):
    """FuNet melhorada com suporte a embeddings de reconhecimento facial - CORRIGIDA"""
    def __init__(self, tipo_fusao='FuNet-C', usar_embeddings=True):
        super(FuNetComEmbeddings, self).__init__()
        self.tipo_fusao = tipo_fusao
        self.usar_embeddings = usar_embeddings
        self.CNN_OUT_SIZE = 512
        self.GNN_OUT_SIZE = 512
        self.EMBEDDING_SIZE = 1024 if usar_embeddings else 0  # FaceNet(512) + ArcFace(512)

        # STREAM CNN MAIS PROFUNDA
        self.cnn_stream = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, self.CNN_OUT_SIZE), 
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(DEVICE)

        # STREAM GNN MELHORADA
        self.gnn_stream = nn.Sequential(
            nn.Linear(49 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.GNN_OUT_SIZE),
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(DEVICE)

        # STREAM DE EMBEDDINGS (CORRIGIDA)
        if self.usar_embeddings:
            self.embedding_stream = nn.Sequential(
                nn.Linear(1024, 512),  # CORREÇÃO: 1024 entrada, 512 saída
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),   # CORREÇÃO: Dimensão reduzida
                nn.ReLU(),
                nn.Dropout(0.3)
            ).to(DEVICE)

        # CLASSIFIER MAIS ROBUSTO
        if self.tipo_fusao == 'FuNet-C':
            if self.usar_embeddings:
                fusion_size = self.CNN_OUT_SIZE + self.GNN_OUT_SIZE + 256  # CORREÇÃO: 256 dos embeddings
            else:
                fusion_size = self.CNN_OUT_SIZE + self.GNN_OUT_SIZE
        else:
            fusion_size = self.CNN_OUT_SIZE

        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ).to(DEVICE)
        
        # Registrar features finais para center loss
        self.features_finais = None

    def forward(self, face_padronizada, embedding_facenet=None, embedding_arcface=None):
        # Garantir que tem dimensão de batch
        if face_padronizada.dim() == 3:
            face_padronizada = face_padronizada.unsqueeze(0)
        
        # 1. Pré-processamento para GNN
        patches = dividir_em_patches(face_padronizada)
        nodes = converter_para_vetores(patches)

        # 2. Execução dos Streams
        features_cnn = self.cnn_stream(face_padronizada)
        
        # GNN
        batch_size, num_nodes, feat_dim = nodes.shape
        gnn_input = nodes.contiguous().view(batch_size, -1)
        features_gnn = self.gnn_stream(gnn_input)
        
        # 3. Stream de Embeddings (CORRIGIDA)
        features_embedding = None
        if self.usar_embeddings and embedding_facenet is not None and embedding_arcface is not None:
            # CORREÇÃO: Verificar dimensões e concatenar corretamente
            if embedding_facenet.dim() == 1:
                embedding_facenet = embedding_facenet.unsqueeze(0)
            if embedding_arcface.dim() == 1:
                embedding_arcface = embedding_arcface.unsqueeze(0)
                
            # Concatenar embeddings do FaceNet e ArcFace
            embedding_combined = t.cat([embedding_facenet, embedding_arcface], dim=1)
            
            # Verificar se a dimensão está correta
            if embedding_combined.shape[1] == 1024:
                features_embedding = self.embedding_stream(embedding_combined)
            else:
                print(f"⚠️  Dimensão incorreta dos embeddings: {embedding_combined.shape}")
                features_embedding = t.zeros(batch_size, 256).to(DEVICE)
        
        # 4. Fusão
        if self.tipo_fusao == 'FuNet-C':
            if features_embedding is not None:
                self.features_finais = t.cat([features_cnn, features_gnn, features_embedding], dim=1)
            else:
                self.features_finais = t.cat([features_cnn, features_gnn], dim=1)
        else:
            self.features_finais = features_cnn

        # 5. Classificação
        output = self.classifier(self.features_finais)
        probabilidades = F.softmax(output, dim=1)
        
        return probabilidades

################################################################################
# FUNÇÃO DE TREINAMENTO MELHORADA - COM BALANCEAMENTO
################################################################################

def treinar_modelo_balanceado(modelo: nn.Module, caminho_pesos_saida: str, pasta_dataset):
    """
    Treinamento otimizado com balanceamento de classes e técnicas avançadas
    """
    print("=" * 60)
    print("         TREINAMENTO BALANCEADO - 30 ÉPOCAS")
    print("=" * 60)
    
    # Carregar dataset com embeddings
    dataset = DeepFakeDatasetComEmbeddings(pasta_dataset, transform=transform_aumento_forte, 
                                          usar_aumento=True, precompute_embeddings=True)
    
    if len(dataset) == 0:
        print("❌ Nenhuma imagem encontrada no dataset!")
        return
    
    # Calcular pesos para balanceamento
    class_weights = calcular_pesos_classes(dataset)
    
    # Split estratificado (manter proporção de classes)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, stratify=dataset.rotulos, random_state=42
    )
    
    train_dataset = t.utils.data.Subset(dataset, train_idx)
    val_dataset = t.utils.data.Subset(dataset, val_idx)
    
    print(f"📊 Split Estratificado: Treino={len(train_idx)}, Validação={len(val_idx)}")
    
    # Calcular pesos para amostragem balanceada
    train_labels = [dataset.rotulos[i] for i in train_idx]
    class_counts = np.bincount(train_labels)
    class_weights_samples = 1. / class_counts
    sample_weights = class_weights_samples[train_labels]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Otimizador com learning rates diferenciados
    optimizer = optim.AdamW([
        {'params': modelo.cnn_stream.parameters(), 'lr': LEARNING_RATE},
        {'params': modelo.gnn_stream.parameters(), 'lr': LEARNING_RATE},
        {'params': modelo.embedding_stream.parameters() if hasattr(modelo, 'embedding_stream') else [], 'lr': LEARNING_RATE * 0.1},
        {'params': modelo.classifier.parameters(), 'lr': LEARNING_RATE * 2}
    ], weight_decay=WEIGHT_DECAY)
    
    # Scheduler com OneCycle
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # CORREÇÃO: Usar scheduler mais simples para evitar problemas de import
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE/100)
    
    # Loss functions - CORREÇÃO: Usar apenas Focal Loss inicialmente
    criterion_ce = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Early stopping
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print("\n🚀 Iniciando treinamento balanceado...")
    
    for epoch in range(1, EPOCHS + 1):
        # --- TREINAMENTO ---
        modelo.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 4:  # Com embeddings
                inputs, targets, embedding_facenet, embedding_arcface = batch_data
                embedding_facenet = embedding_facenet.squeeze(1)
                embedding_arcface = embedding_arcface.squeeze(1)
            else:
                inputs, targets = batch_data[0], batch_data[1]
                embedding_facenet, embedding_arcface = None, None
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            if embedding_facenet is not None:
                embedding_facenet = embedding_facenet.to(DEVICE)
                embedding_arcface = embedding_arcface.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            if modelo.usar_embeddings:
                outputs = modelo(inputs, embedding_facenet, embedding_arcface)
            else:
                outputs = modelo(inputs)
            
            # Calcular loss com focal loss
            loss = criterion_ce(outputs, targets)
            
            loss.backward()
            t.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = t.max(outputs, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # --- VALIDAÇÃO ---
        modelo.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with t.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    inputs, targets, embedding_facenet, embedding_arcface = batch_data
                    embedding_facenet = embedding_facenet.squeeze(1)
                    embedding_arcface = embedding_arcface.squeeze(1)
                else:
                    inputs, targets = batch_data[0], batch_data[1]
                    embedding_facenet, embedding_arcface = None, None
                
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                if embedding_facenet is not None:
                    embedding_facenet = embedding_facenet.to(DEVICE)
                    embedding_arcface = embedding_arcface.to(DEVICE)
                
                outputs = modelo(inputs, embedding_facenet, embedding_arcface)
                loss = criterion_ce(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = t.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Atualizar scheduler
        scheduler.step()
        
        # Print progresso
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train: Loss {avg_train_loss:.4f} Acc {train_accuracy:.4f} | "
              f"Val: Loss {avg_val_loss:.4f} Acc {val_accuracy:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Salvar melhor modelo
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            t.save(modelo.state_dict(), f"{caminho_pesos_saida}_best.pth")
            print(f"💾 NOVO MELHOR MODELO! Acc: {val_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping ativado na época {epoch}")
                break
    
    # Carregar melhor modelo para avaliação final
    if os.path.exists(f"{caminho_pesos_saida}_best.pth"):
        modelo.load_state_dict(t.load(f"{caminho_pesos_saida}_best.pth", map_location=DEVICE))
        print("✅ Melhor modelo carregado para avaliação final")
    
    # Avaliação final
    print("\n" + "=" * 50)
    print("          AVALIAÇÃO FINAL DETALHADA")
    print("=" * 50)
    
    modelo.eval()
    all_preds = []
    all_targets = []
    
    with t.no_grad():
        for batch_data in val_loader:
            if len(batch_data) == 4:
                inputs, targets, embedding_facenet, embedding_arcface = batch_data
                embedding_facenet = embedding_facenet.squeeze(1)
                embedding_arcface = embedding_arcface.squeeze(1)
            else:
                inputs, targets = batch_data[0], batch_data[1]
                embedding_facenet, embedding_arcface = None, None
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            if embedding_facenet is not None:
                embedding_facenet = embedding_facenet.to(DEVICE)
                embedding_arcface = embedding_arcface.to(DEVICE)
            
            outputs = modelo(inputs, embedding_facenet, embedding_arcface)
            _, predicted = t.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Métricas finais
    print("\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(all_targets, all_preds, target_names=['Fake', 'Real'], digits=4))
    
    print("🎯 MATRIZ DE CONFUSÃO:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)
    
    # Salvar modelo final
    t.save(modelo.state_dict(), f"{caminho_pesos_saida}_final.pth")
    print(f"\n💾 Modelos salvos:")
    print(f"   Melhor: {caminho_pesos_saida}_best.pth")
    print(f"   Final: {caminho_pesos_saida}_final.pth")
    
    return modelo

################################################################################
# BLOCO DE EXECUÇÃO PRINCIPAL
################################################################################

def executar_treinamento_completo():
    """Executa o treinamento completo com balanceamento"""
    
    # Criar diretórios
    os.makedirs('pesos', exist_ok=True)
    
    # Usar modelo com embeddings
    modelo = FuNetComEmbeddings(tipo_fusao='FuNet-C', usar_embeddings=True).to(DEVICE)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"🔢 Total de parâmetros do modelo: {total_params:,}")
    
    # Caminho do dataset
    pasta_dataset = "dataset"
    
    if not os.path.exists(pasta_dataset):
        print(f"❌ Pasta do dataset não encontrada: {pasta_dataset}")
        return
    
    # Verificar dataset
    fake_path = os.path.join(pasta_dataset, 'fake')
    real_path = os.path.join(pasta_dataset, 'real')
    
    if not os.path.exists(fake_path) or not os.path.exists(real_path):
        print("❌ Estrutura de pastas incorreta!")
        return
    
    fake_images = len([f for f in os.listdir(fake_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    real_images = len([f for f in os.listdir(real_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"📊 Dataset: {fake_images} fakes, {real_images} reais")
    
    if fake_images == 0 or real_images == 0:
        print("❌ Não há imagens suficientes!")
        return
    
    # Treinar
    CAMINHO_PESOS = 'pesos/funet_com_embeddings_balanceado'
    modelo_treinado = treinar_modelo_balanceado(modelo, CAMINHO_PESOS, pasta_dataset)
    
    print("\n🎉 Treinamento completo concluído!")
    print("💡 Use: pesos/funet_com_embeddings_balanceado_best.pth")
    
    return modelo_treinado

if __name__ == '__main__':
    executar_treinamento_completo()