import cv2
import numpy as np
from collections import deque

class CancelaDetector:
    def __init__(self, video_path, roi=None):
        """
        Inicializa o detector de cancela
        
        Args:
            video_path: caminho do vídeo
            roi: região de interesse (x, y, w, h) onde a cancela está localizada
        """
        self.video_path = video_path
        self.roi = roi
        self.contador = 0
        self.historico_movimento = deque(maxlen=30)  # histórico dos últimos 30 frames
        self.estado_anterior = "baixo"
        self.limiar_movimento = 15  # ajuste conforme necessário
        
    def processar_video(self, mostrar_video=True, salvar_resultado=False):
        """
        Processa o vídeo e detecta movimentos da cancela
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Erro ao abrir o vídeo")
            return
        
        # Pega informações do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processando vídeo: {width}x{height} @ {fps}fps")
        print(f"Total de frames: {total_frames}")
        
        # Configurar gravação se necessário
        if salvar_resultado:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('resultado_cancela.mp4', fourcc, fps, (width, height))
        
        # Lê o primeiro frame
        ret, frame_anterior = cap.read()
        if not ret:
            print("Erro ao ler o primeiro frame")
            return
        
        # Converte para escala de cinza
        frame_anterior_gray = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
        if self.roi:
            x, y, w, h = self.roi
            frame_anterior_gray = frame_anterior_gray[y:y+h, x:x+w]
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Converte para escala de cinza
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Aplica ROI se definida
            if self.roi:
                x, y, w, h = self.roi
                frame_gray_roi = frame_gray[y:y+h, x:x+w]
            else:
                frame_gray_roi = frame_gray
                x, y = 0, 0
            
            # Calcula diferença entre frames
            diff = cv2.absdiff(frame_anterior_gray, frame_gray_roi)
            
            # Aplica threshold
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Aplica blur para reduzir ruído
            thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
            
            # Calcula a quantidade de movimento (pixels brancos)
            movimento = np.sum(thresh) / 255
            movimento_normalizado = movimento / (thresh.shape[0] * thresh.shape[1]) * 100
            
            # Adiciona ao histórico
            self.historico_movimento.append(movimento_normalizado)
            
            # Detecta movimento significativo
            if len(self.historico_movimento) >= 10:
                media_movimento = np.mean(list(self.historico_movimento)[-10:])
                
                # Detecta quando a cancela começa a subir
                if media_movimento > self.limiar_movimento and self.estado_anterior == "baixo":
                    self.contador += 1
                    self.estado_anterior = "subindo"
                    print(f"Frame {frame_count}: Cancela subindo! (Contagem: {self.contador})")
                
                # Reseta o estado quando o movimento para
                elif media_movimento < 5 and self.estado_anterior == "subindo":
                    self.estado_anterior = "baixo"
            
            # Desenha informações no frame
            frame_display = frame.copy()
            
            # Desenha ROI se definida
            if self.roi:
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Adiciona informações de texto
            cv2.putText(frame_display, f"Contagem: {self.contador}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Movimento: {media_movimento if len(self.historico_movimento) >= 10 else 0:.1f}%", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_display, f"Estado: {self.estado_anterior}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_display, f"Frame: {frame_count}/{total_frames}", (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if salvar_resultado:
                out.write(frame_display)
            
            if mostrar_video:
                # Mostra também a diferença
                diff_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                combined = np.hstack([cv2.resize(frame_display, (640, 480)), 
                                     cv2.resize(diff_display, (640, 480))])
                cv2.imshow('Detector de Cancela | Diferença', combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Atualiza frame anterior
            frame_anterior_gray = frame_gray_roi
        
        # Finaliza
        cap.release()
        if salvar_resultado:
            out.release()
        if mostrar_video:
            cv2.destroyAllWindows()
        
        print(f"\n{'='*50}")
        print(f"Processamento concluído!")
        print(f"Total de vezes que a cancela subiu: {self.contador}")
        print(f"{'='*50}")
        
        return self.contador
    
    def definir_roi_manual(self):
        """
        Permite ao usuário selecionar a ROI manualmente
        """
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Erro ao ler o vídeo para seleção de ROI")
            return None
        
        print("\nSelecione a região onde a cancela está localizada")
        print("Pressione ENTER ou ESPAÇO quando terminar")
        print("Pressione C para cancelar")
        
        roi = cv2.selectROI("Selecione a ROI da Cancela", frame, False, False)
        cv2.destroyAllWindows()
        
        if roi[2] > 0 and roi[3] > 0:
            self.roi = roi
            print(f"ROI definida: {roi}")
            return roi
        else:
            print("ROI não selecionada")
            return None


# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelo caminho do seu vídeo
    video_path = r"C:\Users\vitor.matheus\Music\GIT - Corporativo\PAV\EmbarcadoCCR\channel1_20250.mp4"
    
    # Opção 1: Selecionar ROI manualmente
    detector = CancelaDetector(video_path)
    detector.definir_roi_manual()
    
    # Opção 2: Definir ROI diretamente (x, y, largura, altura)
    # roi = (100, 200, 300, 400)  # ajuste conforme seu vídeo
    # detector = CancelaDetector(video_path, roi=roi)
    
    # Processar o vídeo
    # mostrar_video=True: mostra o vídeo durante processamento
    # salvar_resultado=True: salva o vídeo processado
    total_aberturas = detector.processar_video(mostrar_video=True, salvar_resultado=False)
    
    print(f"\nResultado final: A cancela subiu {total_aberturas} vezes")