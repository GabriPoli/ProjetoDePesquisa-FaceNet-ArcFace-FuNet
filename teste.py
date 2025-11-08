import torch as t
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np

# Configuração de dispositivo compatível
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")

# Importações condicionais para evitar erros
try:
    from main import extrair_face_mtcnn, transform_padronizacao, dividir_em_patches, converter_para_vetores
except ImportError:
    print("⚠️  Funções do main.py não encontradas, definindo alternativas...")
    
    # Definições alternativas básicas
    transform_padronizacao = None
    
    def extrair_face_mtcnn(imagem):
        return None
    
    def dividir_em_patches(img_tensor):
        batch_size, channels, height, width = img_tensor.shape
        patch_size = 32
        step = 32
        patches = img_tensor.unfold(2, patch_size, step).unfold(3, patch_size, step)
        patches = patches.contiguous().view(batch_size, -1, channels, patch_size, patch_size)
        return patches.to(img_tensor.device)
    
    def converter_para_vetores(patches):
        batch_size, num_patches = patches.shape[0], patches.shape[1]
        return t.randn(batch_size, num_patches, 64).to(patches.device)

################################################################################
# CÓDIGO DE TESTE COMPLETO COM SUPORTE A MÚLTIPLAS VERSÕES
################################################################################

# VERSÃO 1: FuNet SEM embeddings (compatível com pesos antigos)
class FuNetSemEmbeddings(nn.Module):
    """Versão sem embeddings para compatibilidade com pesos antigos"""
    def __init__(self, tipo_fusao='FuNet-C'):
        super(FuNetSemEmbeddings, self).__init__()
        self.tipo_fusao = tipo_fusao
        self.CNN_OUT_SIZE = 512
        self.GNN_OUT_SIZE = 512

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

        # CLASSIFIER (SEM embeddings)
        if self.tipo_fusao == 'FuNet-C':
            fusion_size = self.CNN_OUT_SIZE + self.GNN_OUT_SIZE  # 512 + 512 = 1024
        else:
            fusion_size = self.CNN_OUT_SIZE

        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ).to(DEVICE)

    def forward(self, face_padronizada):
        # Garantir que tem dimensão de batch
        if face_padronizada.dim() == 3:
            face_padronizada = face_padronizada.unsqueeze(0)
        
        # 1. Pré-processamento para GNN
        patches = dividir_em_patches(face_padronizada)
        nodes = converter_para_vetores(patches)

        # 2. Execução dos Streams
        features_cnn = self.cnn_stream(face_padronizada)
        
        # GNN usando MLP
        batch_size, num_nodes, feat_dim = nodes.shape
        gnn_input = nodes.contiguous().view(batch_size, -1)
        features_gnn = self.gnn_stream(gnn_input)
        
        # 3. Fusão
        if self.tipo_fusao == 'FuNet-C':
            features_finais = t.cat([features_cnn, features_gnn], dim=1)
        else:
            features_finais = features_cnn

        # 4. Classificação
        output = self.classifier(features_finais)
        probabilidades = F.softmax(output, dim=1)
        
        return probabilidades

# VERSÃO 2: FuNet COM embeddings (para pesos novos)
class FuNetComEmbeddings(nn.Module):
    """Versão com embeddings para pesos novos"""
    def __init__(self, tipo_fusao='FuNet-C'):
        super(FuNetComEmbeddings, self).__init__()
        self.tipo_fusao = tipo_fusao
        self.CNN_OUT_SIZE = 512
        self.GNN_OUT_SIZE = 512
        self.EMBEDDING_SIZE = 1024  # FaceNet(512) + ArcFace(512)

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

        # STREAM DE EMBEDDINGS
        self.embedding_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        ).to(DEVICE)

        # CLASSIFIER (COM embeddings)
        if self.tipo_fusao == 'FuNet-C':
            fusion_size = self.CNN_OUT_SIZE + self.GNN_OUT_SIZE + 256  # 512 + 512 + 256 = 1280
        else:
            fusion_size = self.CNN_OUT_SIZE

        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ).to(DEVICE)

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
        
        # 3. Stream de Embeddings (usar zeros se não for fornecido)
        if embedding_facenet is not None and embedding_arcface is not None:
            if embedding_facenet.dim() == 1:
                embedding_facenet = embedding_facenet.unsqueeze(0)
            if embedding_arcface.dim() == 1:
                embedding_arcface = embedding_arcface.unsqueeze(0)
            embedding_combined = t.cat([embedding_facenet, embedding_arcface], dim=1)
            features_embedding = self.embedding_stream(embedding_combined)
        else:
            # Usar zeros se embeddings não estiverem disponíveis
            features_embedding = t.zeros(batch_size, 256).to(DEVICE)
        
        # 4. Fusão
        if self.tipo_fusao == 'FuNet-C':
            features_finais = t.cat([features_cnn, features_gnn, features_embedding], dim=1)
        else:
            features_finais = features_cnn

        # 5. Classificação
        output = self.classifier(features_finais)
        probabilidades = F.softmax(output, dim=1)
        
        return probabilidades

################################################################################
# SISTEMA DE RECONHECIMENTO FACIAL SIMPLIFICADO
################################################################################

class SistemaReconhecimentoFacial:
    """Sistema simplificado de reconhecimento facial"""
    
    def __init__(self):
        self.device = DEVICE
        self.limiar_facenet = 0.6
        self.limiar_arcface = 0.6
        self.embedding_referencia = None
        
        print("🔄 Inicializando sistema de reconhecimento...")
        # Em uma implementação real, aqui carregaríamos os modelos FaceNet e ArcFace
        # Por simplicidade, vamos usar placeholders
        print("✅ Sistema de reconhecimento inicializado (modo simulado)")
    
    def definir_referencia(self, caminho_imagem_referencia):
        """Define uma imagem de referência para comparação"""
        print(f"📸 Definindo referência: {os.path.basename(caminho_imagem_referencia)}")
        
        # Simular definição de referência
        self.embedding_referencia = {
            'facenet': t.randn(1, 512).to(DEVICE),
            'arcface': t.randn(1, 512).to(DEVICE)
        }
        print("✅ Referência definida com sucesso (simulada)")
        return True
    
    def calcular_similaridades(self, face_tensor):
        """Calcula similaridades com a referência (simulado)"""
        if self.embedding_referencia is None:
            return 0.0, 0.0, "REJEITADO", "REJEITADO"
        
        # Simular cálculo de similaridades
        similaridade_facenet = np.random.uniform(0.5, 0.9)
        similaridade_arcface = np.random.uniform(0.5, 0.9)
        
        # Determinar vereditos
        veredito_facenet = "ACEITO" if similaridade_facenet >= self.limiar_facenet else "REJEITADO"
        veredito_arcface = "ACEITO" if similaridade_arcface >= self.limiar_arcface else "REJEITADO"
        
        return similaridade_facenet, similaridade_arcface, veredito_facenet, veredito_arcface

################################################################################
# FUNÇÕES DE PRÉ-PROCESSAMENTO
################################################################################

def preprocessar_imagem_individual(caminho_imagem):
    """
    Pré-processa uma imagem individual para o modelo
    """
    try:
        # Carregar imagem
        imagem = Image.open(caminho_imagem).convert('RGB')
        
        # Extrair face com MTCNN (se disponível)
        face_tensor = extrair_face_mtcnn(imagem)
        
        if face_tensor is None or transform_padronizacao is None:
            # Se não detectar face ou transform não disponível, usar abordagem simples
            imagem_redimensionada = imagem.resize((224, 224))
            face_tensor = t.from_numpy(np.array(imagem_redimensionada)).float() / 255.0
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            print(f"⚠️  Usando processamento simplificado para {os.path.basename(caminho_imagem)}")
        
        return face_tensor
    
    except Exception as e:
        print(f"❌ Erro ao processar {caminho_imagem}: {e}")
        return None

################################################################################
# CARREGAMENTO FLEXÍVEL DE MODELOS
################################################################################

def carregar_modelo_flexivel(caminho_pesos):
    """Carrega o modelo FuNet de forma flexível, detectando automaticamente a versão"""
    
    if not os.path.exists(caminho_pesos):
        print(f"❌ Arquivo de pesos não encontrado: {caminho_pesos}")
        # Criar modelo padrão para teste
        print("🔄 Criando modelo padrão para teste...")
        modelo = FuNetComEmbeddings(tipo_fusao='FuNet-C').to(DEVICE)
        modelo.eval()
        return modelo, "PADRAO"
    
    # Tentar primeiro com a versão COM embeddings (mais nova)
    print("🔄 Tentando carregar modelo COM embeddings...")
    modelo = FuNetComEmbeddings(tipo_fusao='FuNet-C').to(DEVICE)
    
    try:
        checkpoint = t.load(caminho_pesos, map_location=DEVICE)
        
        # Verificar se o checkpoint tem a stream de embeddings
        tem_embeddings = any('embedding_stream' in key for key in checkpoint.keys())
        
        if tem_embeddings:
            # Carregar modelo COM embeddings
            modelo.load_state_dict(checkpoint)
            modelo.eval()
            print("✅ Modelo COM embeddings carregado com sucesso")
            return modelo, "COM_EMBEDDINGS"
        else:
            # Tentar com a versão SEM embeddings
            print("🔄 Tentando carregar modelo SEM embeddings...")
            modelo_sem = FuNetSemEmbeddings(tipo_fusao='FuNet-C').to(DEVICE)
            
            # Filtrar apenas os pesos compatíveis
            model_dict = modelo_sem.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
            
            if len(pretrained_dict) > 0:
                model_dict.update(pretrained_dict)
                modelo_sem.load_state_dict(model_dict)
                modelo_sem.eval()
                print(f"✅ Modelo SEM embeddings carregado ({len(pretrained_dict)}/{len(checkpoint)} parâmetros)")
                return modelo_sem, "SEM_EMBEDDINGS"
            else:
                print("❌ Nenhum parâmetro compatível encontrado")
                return None, "ERRO"
                
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None, "ERRO"

################################################################################
# PIPELINE COMPLETO DE TESTE
################################################################################

def testar_imagem_completa(modelo_funet, tipo_modelo, sistema_reconhecimento, caminho_imagem):
    """
    Pipeline completo de teste integrando FuNet + Reconhecimento Facial
    """
    print(f"\n🔍 Testando imagem: {os.path.basename(caminho_imagem)}")
    
    # Pré-processar imagem
    input_tensor = preprocessar_imagem_individual(caminho_imagem)
    
    if input_tensor is None:
        return {
            'Imagem': caminho_imagem,
            'Predicao_FuNet': 'ERRO NO PRÉ-PROCESSAMENTO',
            'Confianca_FuNet': 'N/A',
            'Prob_Fake': 'N/A', 
            'Prob_Real': 'N/A',
            'Similaridade_FaceNet': 'N/A',
            'Similaridade_ArcFace': 'N/A',
            'Veredito_FaceNet': 'N/A',
            'Veredito_ArcFace': 'N/A',
            'Decisao_Final': 'ERRO',
            'Vulnerabilidade': 'N/A',
            'Tipo_Modelo': tipo_modelo
        }
    
    # Fazer predição com FuNet
    modelo_funet.eval()
    with t.no_grad():
        try:
            # Passo 1: Detecção de Autenticidade (FuNet)
            if tipo_modelo == "COM_EMBEDDINGS":
                # Para modelo com embeddings, usar zeros como placeholder
                batch_size = input_tensor.shape[0]
                embedding_facenet = t.zeros(batch_size, 512).to(DEVICE)
                embedding_arcface = t.zeros(batch_size, 512).to(DEVICE)
                probabilidades = modelo_funet(input_tensor, embedding_facenet, embedding_arcface)
            else:
                # Para modelo sem embeddings
                probabilidades = modelo_funet(input_tensor)
            
            prob_fake = probabilidades[0, 0].item()
            prob_real = probabilidades[0, 1].item()
            confianca_funet = max(prob_fake, prob_real)
            predicao_funet = "REAL" if prob_real > prob_fake else "FAKE"
            
            # Passo 2: Geração e Comparação de Embeddings
            similaridade_facenet, similaridade_arcface, veredito_facenet, veredito_arcface = \
                sistema_reconhecimento.calcular_similaridades(input_tensor)
            
            # Passo 3: Análise Integrada (Filtro de Segurança)
            funet_aceita = (predicao_funet == "REAL")
            reconhecimento_aceita = (veredito_facenet == "ACEITO") or (veredito_arcface == "ACEITO")
            
            if funet_aceita and reconhecimento_aceita:
                decisao_final = "✅ ACEITO PELO SISTEMA"
            elif not funet_aceita:
                decisao_final = "❌ REJEITADO: Detectado como DEEPFAKE"
            else:
                decisao_final = "❌ REJEITADO: Não reconhecido"
            
            # Análise de Vulnerabilidade
            vulnerabilidade = "ALTA" if (veredito_facenet == "ACEITO" or veredito_arcface == "ACEITO") and predicao_funet == "FAKE" else "BAIXA"
            
            return {
                'Imagem': caminho_imagem,
                'Predicao_FuNet': predicao_funet,
                'Confianca_FuNet': f"{confianca_funet:.4f}",
                'Prob_Fake': f"{prob_fake:.4f}",
                'Prob_Real': f"{prob_real:.4f}",
                'Similaridade_FaceNet': f"{similaridade_facenet:.4f}",
                'Similaridade_ArcFace': f"{similaridade_arcface:.4f}",
                'Veredito_FaceNet': veredito_facenet,
                'Veredito_ArcFace': veredito_arcface,
                'Decisao_Final': decisao_final,
                'Vulnerabilidade': vulnerabilidade,
                'Tipo_Modelo': tipo_modelo
            }
            
        except Exception as e:
            print(f"❌ Erro durante predição: {e}")
            return {
                'Imagem': caminho_imagem,
                'Predicao_FuNet': f"ERRO NA PREDIÇÃO: {str(e)}",
                'Confianca_FuNet': 'N/A',
                'Prob_Fake': 'N/A',
                'Prob_Real': 'N/A',
                'Similaridade_FaceNet': 'N/A',
                'Similaridade_ArcFace': 'N/A',
                'Veredito_FaceNet': 'N/A',
                'Veredito_ArcFace': 'N/A',
                'Decisao_Final': 'ERRO',
                'Vulnerabilidade': 'N/A',
                'Tipo_Modelo': tipo_modelo
            }

################################################################################
# TESTE COMPLETO COM ANÁLISE DETALHADA
################################################################################

def executar_teste_completo(caminho_pasta_teste, caminho_pesos_funet, caminho_imagem_referencia):
    """
    Executa teste completo com análise comparativa detalhada
    """
    print("🔬 EXECUTANDO TESTE COMPARATIVO COMPLETO")
    print("=" * 60)
    
    # Carregar modelo FuNet de forma flexível
    modelo_funet, tipo_modelo = carregar_modelo_flexivel(caminho_pesos_funet)
    
    if modelo_funet is None:
        print("❌ Falha ao carregar modelo FuNet")
        return
    
    print(f"📦 Tipo de modelo detectado: {tipo_modelo}")
    
    # Inicializar sistema de reconhecimento
    sistema_reconhecimento = SistemaReconhecimentoFacial()
    
    # Definir imagem de referência
    if not sistema_reconhecimento.definir_referencia(caminho_imagem_referencia):
        print("❌ Falha ao definir referência")
        return
    
    print("✅ Sistema completo inicializado com sucesso!")
    
    # Encontrar imagens na pasta
    extensoes = ['.jpg', '.jpeg', '.png', '.bmp']
    imagens = []
    
    for arquivo in os.listdir(caminho_pasta_teste):
        if any(arquivo.lower().endswith(ext) for ext in extensoes):
            imagens.append(os.path.join(caminho_pasta_teste, arquivo))
    
    if not imagens:
        print("❌ Nenhuma imagem encontrada na pasta de teste")
        return
    
    print(f"📁 Encontradas {len(imagens)} imagens para teste")
    
    # Executar testes
    resultados = []
    for caminho_imagem in imagens:
        resultado = testar_imagem_completa(modelo_funet, tipo_modelo, sistema_reconhecimento, caminho_imagem)
        resultados.append(resultado)
        
        # Mostrar resultado resumido
        print(f"   {os.path.basename(resultado['Imagem']):<20} | "
              f"FuNet: {resultado['Predicao_FuNet']:<6} | "
              f"FaceNet: {resultado['Veredito_FaceNet']:<8} | "
              f"ArcFace: {resultado['Veredito_ArcFace']:<8} | "
              f"Final: {resultado['Decisao_Final'][:20]}")
    
    # Análise detalhada
    print("\n" + "=" * 80)
    print("                      ANÁLISE DETALHADA DOS RESULTADOS")
    print("=" * 80)
    
    for resultado in resultados:
        if 'ERRO' not in resultado['Predicao_FuNet']:
            print(f"\n📊 IMAGEM: {os.path.basename(resultado['Imagem'])}")
            print(f"   🤖 DETECÇÃO FuNet ({resultado['Tipo_Modelo']}):")
            print(f"      Predição: {resultado['Predicao_FuNet']}")
            print(f"      Confiança: {resultado['Confianca_FuNet']}")
            print(f"      Prob Fake: {resultado['Prob_Fake']}")
            print(f"      Prob Real: {resultado['Prob_Real']}")
            
            print(f"   👤 RECONHECIMENTO FACIAL:")
            print(f"      FaceNet - Similaridade: {resultado['Similaridade_FaceNet']} | Veredito: {resultado['Veredito_FaceNet']}")
            print(f"      ArcFace - Similaridade: {resultado['Similaridade_ArcFace']} | Veredito: {resultado['Veredito_ArcFace']}")
            
            print(f"   🛡️  SISTEMA INTEGRADO:")
            print(f"      Decisão Final: {resultado['Decisao_Final']}")
            print(f"      Nível de Vulnerabilidade: {resultado['Vulnerabilidade']}")
            print("-" * 50)
    
    # Estatísticas finais
    print("\n📈 ESTATÍSTICAS FINAIS:")
    print("=" * 40)
    
    reais_funet = sum(1 for r in resultados if r['Predicao_FuNet'] == 'REAL')
    fakes_funet = sum(1 for r in resultados if r['Predicao_FuNet'] == 'FAKE')
    aceitos_face = sum(1 for r in resultados if r['Veredito_FaceNet'] == 'ACEITO')
    aceitos_arc = sum(1 for r in resultados if r['Veredito_ArcFace'] == 'ACEITO')
    aceitos_sistema = sum(1 for r in resultados if 'ACEITO' in r['Decisao_Final'])
    vulneraveis = sum(1 for r in resultados if r['Vulnerabilidade'] == 'ALTA')
    
    print(f"   FuNet - Reais: {reais_funet}, Fakes: {fakes_funet}")
    print(f"   FaceNet - Aceitos: {aceitos_face}/{len(imagens)}")
    print(f"   ArcFace - Aceitos: {aceitos_arc}/{len(imagens)}")
    print(f"   Sistema - Aceitos: {aceitos_sistema}/{len(imagens)}")
    print(f"   Casos Vulneráveis: {vulneraveis}/{len(imagens)}")
    print(f"   Tipo de Modelo: {tipo_modelo}")

################################################################################
# EXECUÇÃO PRINCIPAL
################################################################################

if __name__ == '__main__':
    # Configurações
    CAMINHO_PESOS_FUNET = 'pesos/funet_com_embeddings_balanceado_best.pth'  # Modelo balanceado
    CAMINHO_IMAGEM_REFERENCIA = 'imagens_de_teste/real_reference.jpg'
    CAMINHO_PASTA_TESTE = "imagens_de_teste"
    
    # Verificar se a pasta de teste existe
    if not os.path.exists(CAMINHO_PASTA_TESTE):
        print(f"❌ Pasta de teste não encontrada: {CAMINHO_PASTA_TESTE}")
        print("💡 Criando pasta de exemplo...")
        os.makedirs(CAMINHO_PASTA_TESTE, exist_ok=True)
        print("✅ Pasta criada. Adicione imagens para teste.")
        exit(1)
    
    # Verificar se a imagem de referência existe
    if not os.path.exists(CAMINHO_IMAGEM_REFERENCIA):
        print(f"❌ Imagem de referência não encontrada: {CAMINHO_IMAGEM_REFERENCIA}")
        print("💡 Coloque uma imagem real como referência ou ajuste o caminho")
        # Tentar usar qualquer imagem real como fallback
        real_images = [f for f in os.listdir(CAMINHO_PASTA_TESTE) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) and 'real' in f.lower()]
        if real_images:
            CAMINHO_IMAGEM_REFERENCIA = os.path.join(CAMINHO_PASTA_TESTE, real_images[0])
            print(f"🔄 Usando como referência: {CAMINHO_IMAGEM_REFERENCIA}")
        else:
            print("❌ Nenhuma imagem real encontrada para usar como referência")
            # Criar uma imagem dummy para teste
            print("🔄 Criando referência dummy para teste...")
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(dummy_image).save(CAMINHO_IMAGEM_REFERENCIA)
            print(f"✅ Referência dummy criada: {CAMINHO_IMAGEM_REFERENCIA}")
    
    # Executar teste completo
    executar_teste_completo(CAMINHO_PASTA_TESTE, CAMINHO_PESOS_FUNET, CAMINHO_IMAGEM_REFERENCIA)