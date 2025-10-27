import cv2
import numpy as np
from collections import defaultdict

class ContadorVeiculos:
    def __init__(self, video_path):
        """
        Inicializa o contador de veículos
        
        Args:
            video_path: caminho do vídeo
        """
        self.video_path = video_path
        self.contador = 0
        self.linha_contagem = None
        self.veiculos_rastreados = {}
        self.proximo_id = 0
        self.veiculos_contados = set()
        
        # Parâmetros ajustáveis para filtrar apenas carros
        self.AREA_MIN = 2000  # Área mínima (pixels²) - AUMENTADO
        self.AREA_MAX = 50000  # Área máxima para evitar ruídos grandes
        self.ASPECT_RATIO_MIN = 0.4  # Proporção mínima (largura/altura)
        self.ASPECT_RATIO_MAX = 4.0  # Proporção máxima
        self.WIDTH_MIN = 40  # Largura mínima em pixels
        self.HEIGHT_MIN = 40  # Altura mínima em pixels
        self.DISTANCIA_MAX_RASTREAMENTO = 150  # Pixels
        
        # Detector de fundo otimizado
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,  # Histórico reduzido para adaptar mais rápido
            varThreshold=40,  # Mais sensível
            detectShadows=False  # Desabilita detecção de sombras
        )
        
    def definir_linha_contagem(self, frame_height, posicao_percentual=0.5):
        """
        Define a linha de contagem (horizontal no meio do frame por padrão)
        
        Args:
            frame_height: altura do frame
            posicao_percentual: posição da linha (0.0 a 1.0)
        """
        self.linha_contagem = int(frame_height * posicao_percentual)
        print(f"Linha de contagem definida em y={self.linha_contagem}")
        
    def definir_linha_manual(self):
        """
        Permite ao usuário definir a linha de contagem manualmente
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Erro ao ler o vídeo")
            return
        
        print("\nClique para definir a linha de contagem")
        print("Pressione ENTER quando terminar")
        
        linha_y = [frame.shape[0] // 2]  # valor padrão
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                linha_y[0] = y
                
        cv2.namedWindow("Definir Linha de Contagem")
        cv2.setMouseCallback("Definir Linha de Contagem", mouse_callback)
        
        while True:
            frame_temp = frame.copy()
            cv2.line(frame_temp, (0, linha_y[0]), (frame.shape[1], linha_y[0]), 
                    (0, 255, 0), 3)
            cv2.putText(frame_temp, "Clique para ajustar a linha - ENTER para confirmar", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Definir Linha de Contagem", frame_temp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 or key == 10:  # ENTER
                break
                
        cv2.destroyAllWindows()
        self.linha_contagem = linha_y[0]
        print(f"Linha de contagem definida em y={self.linha_contagem}")
    
    def validar_veiculo(self, x, y, w, h, area):
        """
        Valida se a detecção é realmente um veículo baseado em múltiplos critérios
        """
        # Filtro 1: Área
        if area < self.AREA_MIN or area > self.AREA_MAX:
            return False
        
        # Filtro 2: Dimensões mínimas
        if w < self.WIDTH_MIN or h < self.HEIGHT_MIN:
            return False
        
        # Filtro 3: Proporção (aspect ratio)
        aspect_ratio = w / float(h)
        if aspect_ratio < self.ASPECT_RATIO_MIN or aspect_ratio > self.ASPECT_RATIO_MAX:
            return False
        
        # Filtro 4: Densidade (relação entre área do contorno e área do bounding box)
        bbox_area = w * h
        density = area / bbox_area
        if density < 0.3:  # Muito "vazio" - provavelmente ruído
            return False
        
        return True
    
    def detectar_veiculos(self, frame):
        """
        Detecta veículos usando subtração de fundo com filtros aprimorados
        """
        # Pré-processamento: reduz ruído
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Aplica subtração de fundo
        fg_mask = self.bg_subtractor.apply(frame_blur, learningRate=0.001)
        
        # Binarização mais agressiva
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Operações morfológicas otimizadas
        # 1. Remove ruídos pequenos
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        # 2. Fecha buracos dentro dos veículos
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # 3. Dilata para conectar partes do mesmo veículo
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=2)
        
        # Encontra contornos
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        deteccoes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Valida se é um veículo
            if self.validar_veiculo(x, y, w, h, area):
                centro_x = x + w // 2
                centro_y = y + h // 2
                
                deteccoes.append({
                    'bbox': (x, y, w, h),
                    'centro': (centro_x, centro_y),
                    'area': area
                })
        
        return deteccoes, fg_mask
    
    def calcular_iou(self, box1, box2):
        """
        Calcula Intersection over Union entre duas bounding boxes
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Coordenadas da interseção
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def rastrear_e_contar(self, deteccoes, frame_height):
        """
        Rastreia veículos e conta quando cruzam a linha
        Usa distância E IoU para rastreamento mais robusto
        """
        if not deteccoes:
            # Incrementa contador de ausência para todos os veículos
            for vid in list(self.veiculos_rastreados.keys()):
                self.veiculos_rastreados[vid]['frames_ausente'] += 1
                # Remove se ausente por muito tempo
                if self.veiculos_rastreados[vid]['frames_ausente'] > 20:
                    del self.veiculos_rastreados[vid]
            return
        
        # Lista de IDs já usados neste frame
        ids_usados = set()
        novos_veiculos = {}
        
        for deteccao in deteccoes:
            centro = deteccao['centro']
            bbox = deteccao['bbox']
            
            # Procura correspondência com veículos existentes
            melhor_id = None
            melhor_score = 0
            
            for veiculo_id, info in self.veiculos_rastreados.items():
                if veiculo_id in ids_usados:
                    continue
                
                # Calcula distância entre centros
                dist = np.sqrt((centro[0] - info['centro'][0])**2 + 
                             (centro[1] - info['centro'][1])**2)
                
                # Calcula IoU entre bounding boxes
                iou = self.calcular_iou(bbox, info['bbox'])
                
                # Score combinado (50% distância, 50% IoU)
                if dist < self.DISTANCIA_MAX_RASTREAMENTO:
                    score = (1 - dist / self.DISTANCIA_MAX_RASTREAMENTO) * 0.5 + iou * 0.5
                    
                    if score > melhor_score and score > 0.2:  # Threshold mínimo
                        melhor_score = score
                        melhor_id = veiculo_id
            
            if melhor_id is not None:
                # Atualiza veículo existente
                veiculo_id = melhor_id
                info_anterior = self.veiculos_rastreados[melhor_id]
                ids_usados.add(veiculo_id)
            else:
                # Novo veículo
                veiculo_id = self.proximo_id
                self.proximo_id += 1
                info_anterior = {'centro': centro, 'cruzou': False, 'frames_ausente': 0}
            
            # Verifica se cruzou a linha
            y_anterior = info_anterior['centro'][1]
            y_atual = centro[1]
            
            cruzou = info_anterior.get('cruzou', False)
            
            # Detecta cruzamento da linha (apenas de cima para baixo neste caso)
            if not cruzou and self.linha_contagem is not None:
                # Cruzamento de cima para baixo
                if y_anterior < self.linha_contagem <= y_atual:
                    if veiculo_id not in self.veiculos_contados:
                        self.contador += 1
                        self.veiculos_contados.add(veiculo_id)
                        cruzou = True
                        print(f"✓ Veículo #{veiculo_id} contado! Total: {self.contador}")
            
            novos_veiculos[veiculo_id] = {
                'centro': centro,
                'bbox': bbox,
                'cruzou': cruzou,
                'frames_ausente': 0,
                'area': deteccao['area']
            }
        
        # Mantém veículos que ainda não foram substituídos
        for vid, info in self.veiculos_rastreados.items():
            if vid not in novos_veiculos:
                info['frames_ausente'] = info.get('frames_ausente', 0) + 1
                if info['frames_ausente'] < 20:
                    novos_veiculos[vid] = info
        
        self.veiculos_rastreados = novos_veiculos
    
    def processar_video(self, mostrar_video=True, salvar_resultado=False):
        """
        Processa o vídeo e conta os veículos
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Erro ao abrir o vídeo")
            return
        
        # Informações do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Processando vídeo: {width}x{height} @ {fps}fps")
        print(f"Total de frames: {total_frames}")
        print(f"Área mínima: {self.AREA_MIN} | Área máxima: {self.AREA_MAX}")
        print(f"{'='*60}\n")
        
        # Define linha de contagem se não foi definida
        if self.linha_contagem is None:
            self.definir_linha_contagem(height, 0.5)
        
        # Configurar gravação
        if salvar_resultado:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('resultado_contagem.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detecta veículos
            deteccoes, fg_mask = self.detectar_veiculos(frame)
            
            # Rastreia e conta
            self.rastrear_e_contar(deteccoes, height)
            
            # Desenha visualizações
            frame_display = frame.copy()
            
            # Desenha linha de contagem
            cv2.line(frame_display, (0, self.linha_contagem), 
                    (width, self.linha_contagem), (0, 255, 0), 3)
            cv2.putText(frame_display, "LINHA DE CONTAGEM", (width//2 - 100, self.linha_contagem - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Desenha detecções
            for veiculo_id, info in self.veiculos_rastreados.items():
                x, y, w, h = info['bbox']
                centro = info['centro']
                cruzou = info['cruzou']
                area = info.get('area', 0)
                
                # Cor: verde se já contado, vermelho se não
                cor = (0, 255, 0) if cruzou else (0, 0, 255)
                
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), cor, 2)
                cv2.circle(frame_display, centro, 5, cor, -1)
                
                # Label com ID e área
                label = f"#{veiculo_id} A:{int(area)}"
                cv2.putText(frame_display, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
            
            # Painel de informações
            cv2.rectangle(frame_display, (10, 10), (350, 180), (0, 0, 0), -1)
            cv2.rectangle(frame_display, (10, 10), (350, 180), (255, 255, 255), 2)
            
            cv2.putText(frame_display, f"VEICULOS: {self.contador}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Frame: {frame_count}/{total_frames}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_display, f"Rastreados: {len(self.veiculos_rastreados)}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_display, f"Detectados agora: {len(deteccoes)}", 
                       (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_display, f"Progresso: {int(frame_count/total_frames*100)}%", 
                       (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            if salvar_resultado:
                out.write(frame_display)
            
            if mostrar_video:
                # Mostra frame original e máscara
                fg_mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                combined = np.hstack([
                    cv2.resize(frame_display, (640, 480)),
                    cv2.resize(fg_mask_color, (640, 480))
                ])
                cv2.imshow('Contador de Veiculos | Mascara', combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+'):
                    self.AREA_MIN += 500
                    print(f"Área mínima aumentada para: {self.AREA_MIN}")
                elif key == ord('-'):
                    self.AREA_MIN = max(500, self.AREA_MIN - 500)
                    print(f"Área mínima reduzida para: {self.AREA_MIN}")
        
        # Finaliza
        cap.release()
        if salvar_resultado:
            out.release()
        if mostrar_video:
            cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"✓ Processamento concluído!")
        print(f"✓ Total de veículos contados: {self.contador}")
        print(f"{'='*60}")
        
        return self.contador


# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelo caminho do seu vídeo
    video_path = r"C:\Users\vitor.matheus\Music\GIT - Corporativo\PAV\EmbarcadoCCR\channel1_20250.mp4"
    
    # Criar contador
    contador = ContadorVeiculos(video_path)
    
    # Ajuste os parâmetros conforme necessário:
    # contador.AREA_MIN = 3000  # Aumentar para carros maiores
    # contador.AREA_MAX = 40000  # Ajustar para o tamanho máximo esperado
    # contador.WIDTH_MIN = 50  # Largura mínima
    # contador.HEIGHT_MIN = 50  # Altura mínima
    
    # Definir linha de contagem
    contador.definir_linha_manual()
    
    # Processar vídeo
    # Durante a execução:
    # - Pressione '+' para aumentar área mínima
    # - Pressione '-' para diminuir área mínima
    # - Pressione 'q' para sair
    total = contador.processar_video(mostrar_video=True, salvar_resultado=False)
    
    print(f"\nTotal de veículos que passaram: {total}")