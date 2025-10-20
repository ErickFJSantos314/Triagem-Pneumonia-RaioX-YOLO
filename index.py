import streamlit as st
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np
import pandas as pd
import cv2

st.set_page_config(layout="wide")
st.title('🩺 Triagem Rápida de Raio X Toráxico')
st.subheader('Análise Comparativa de Precisão entre YOLOv8m, YOLOv9c e YOLOv10m')

CLINICAL_NAMES = {
    "Bacteria": "Pneumonia Bacteriana",
    "Virus": "Pneumonia Viral",
    "Normal Lung Detected": "Pulmão Normal Detectado",
    "N/A": "N/A"
}

MODEL_PATHS = {
    "YOLOv8m": "YOLOv8m.pt",
    "YOLOv9c": "YOLOv9c.pt",
    "YOLOv10m": "YOLOv10m.pt"
}

MODEL_COLORS = {
    "YOLOv8m": (255, 255, 0),
    "YOLOv9c": (0, 165, 255),
    "YOLOv10m": (0, 0, 255)
}


TREINO_METRICS = {
    "YOLOv8m": {"mAP@.5:.95": 0.782, "Inf_Speed (ms)": 4.6},
    "YOLOv9c": {"mAP@.5:.95": 0.787, "Inf_Speed (ms)": 8.0},
    "YOLOv10m": {"mAP@.5:.95": 0.794, "Inf_Speed (ms)": 5.9}
}

model_load_error = None

@st.cache_resource
def load_yolo_models():
    """Carrega os modelos YOLO uma única vez para otimizar o Streamlit."""
    models = {}
    
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = YOLO(path)
        except Exception as e:
            global model_load_error
            model_load_error = (name, path, e)
            return None
    
    return models

MODELS = load_yolo_models()

if MODELS is None and model_load_error is not None:
    name, path, e = model_load_error
    st.error(f"Erro Crítico ao carregar o modelo {name} em '{path}'. Verifique o caminho do arquivo .pt.")
    st.exception(e)
    st.stop()

def highlight_best_and_worst(s):
    """
    Destaca o melhor (verde) e o pior (vermelho) em cada coluna.
    - mAP: Maior é Melhor (Verde) / Menor é Pior (Vermelho)
    - Inf_Speed: Menor é Melhor (Verde) / Maior é Pior (Vermelho)
    """
    df = s.copy()

    is_max_map = df['mAP@.5:.95'] == df['mAP@.5:.95'].max()
    is_min_map = df['mAP@.5:.95'] == df['mAP@.5:.95'].min()
    
    is_min_speed = df['Inf_Speed (ms)'] == df['Inf_Speed (ms)'].min()
    is_max_speed = df['Inf_Speed (ms)'] == df['Inf_Speed (ms)'].max()
    
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    styles.loc[is_max_map, 'mAP@.5:.95'] = 'background-color: #d4edda; color: #155724; font-weight: bold'
    styles.loc[is_min_speed, 'Inf_Speed (ms)'] = 'background-color: #d4edda; color: #155724; font-weight: bold'

    styles.loc[is_min_map & ~is_max_map, 'mAP@.5:.95'] = 'background-color: #f8d7da; color: #721c24'
    styles.loc[is_max_speed & ~is_min_speed, 'Inf_Speed (ms)'] = 'background-color: #f8d7da; color: #721c24'

    return styles



def run_inference_and_compare(img_bytes):
    """Executa a inferência nos 3 modelos e compara os resultados."""

    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    all_results = {}
    best_detection = {'score': -1.0, 'model': None, 'class': None}
    
    for model_name, model_instance in MODELS.items():
        img_drawn = img_cv2.copy()
        
        results = model_instance(img_np, conf=0.25, iou=0.7, verbose=False, imgsz=320)
        
        local_best_conf = 0.0
        local_best_class_short = "N/A"

        for r in results:
            if r.boxes:
                for box_data in r.boxes:
                    x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
                    conf = float(box_data.conf[0])
                    cls_id = int(box_data.cls[0])
                    class_name_short = model_instance.names[cls_id]
                    
                    class_name_long = CLINICAL_NAMES.get(class_name_short, class_name_short)
                    
                    if conf > local_best_conf:
                        local_best_conf = conf
                        local_best_class_short = class_name_short

                    if conf > best_detection['score']:
                        best_detection['score'] = conf
                        best_detection['model'] = model_name
                        best_detection['class'] = class_name_long
                        best_detection['box'] = [x1, y1, x2, y2]

                    
                    color = MODEL_COLORS[model_name]
                    cv2.rectangle(img_drawn, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name_long} ({conf:.2f})"
                    
                    
                    cv2.putText(img_drawn, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        
        classe_clinica = CLINICAL_NAMES.get(local_best_class_short, local_best_class_short)
        
        all_results[model_name] = {
            "Confianca_Maxima": local_best_conf,
            "Classe_Detectada": classe_clinica,
            "Image_Output": img_drawn
        }
    
    if best_detection['model']:
        winner_name = best_detection['model']
        winner_img = all_results[winner_name]['Image_Output']
        
        x1, y1, x2, y2 = best_detection['box']
        cv2.rectangle(winner_img, (x1-3, y1-3), (x2+3, y2+3), (0, 0, 255), 3) 
        
    
    return all_results, best_detection

col_center, col_rest = st.columns([1, 4])
with col_center:
    uploaded_file = st.file_uploader("🖼️ **Faça o upload de uma imagem de Raio X:**", type=['png', 'jpg', 'jpeg'])

st.markdown("---")


if uploaded_file is not None:
    
    with st.spinner("Analisando imagem com YOLOv8m, YOLOv9c e YOLOv10m..."):
        results_data, best_detection = run_inference_and_compare(uploaded_file.getvalue())

    if results_data and best_detection['score'] > 0:
        
        df_podium = pd.DataFrame.from_dict(TREINO_METRICS, orient='index')
        df_podium['Confiança (Teste)'] = [data['Confianca_Maxima'] for data in results_data.values()]
        df_podium['Classe (Teste)'] = [data['Classe_Detectada'] for data in results_data.values()]
        
        df_podium = df_podium.sort_values(by='Confiança (Teste)', ascending=False)
        
        st.header("🥇 Classificação de Desempenho (Teste da Imagem)")
        
        p_col1, p_col2, p_col3 = st.columns(3)
        
        podium_cols = [p_col1, p_col2, p_col3]
        rank_emojis = ["🥇", "🥈", "🥉"]
        
        ranking = {name: (i + 1, row['Confiança (Teste)']) for i, (name, row) in enumerate(df_podium.iterrows())}

        model_order = ["YOLOv8m", "YOLOv9c", "YOLOv10m"]
        
        winner_name = df_podium.index[0]
        
        for i, model_name in enumerate(model_order):
            
            row_data = df_podium.loc[model_name] 

            rank, score = ranking.get(model_name, (0, 0))
            
            title_text = f"{rank_emojis[rank-1]} {model_name}" if rank > 0 else f"{model_name}"
            
            with podium_cols[i]:
                st.subheader(title_text)
                
                mAP_treino = TREINO_METRICS[model_name]['mAP@.5:.95']
                delta_value = row_data['Confiança (Teste)'] - mAP_treino
                
                st.metric(
                    label=f"Confiança Máxima para {row_data['Classe (Teste)']}",
                    value=f"{row_data['Confiança (Teste)']:.3f}", 
                    delta=f"vs mAP Treino: {delta_value:.3f}",
                    delta_color="normal"
                )
        
        st.markdown("---")

        st.header("👁️ Visualização da Detecção por Modelo")
        st.caption(f"**A detecção mais confiável (Vencedor: {winner_name}) é destacada com uma borda VERMELHA.**")
        
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        v8_img = cv2.cvtColor(results_data["YOLOv8m"]["Image_Output"], cv2.COLOR_BGR2RGB)
        v9_img = cv2.cvtColor(results_data["YOLOv9c"]["Image_Output"], cv2.COLOR_BGR2RGB)
        v10_img = cv2.cvtColor(results_data["YOLOv10m"]["Image_Output"], cv2.COLOR_BGR2RGB)
        
        image_map = {
            "YOLOv8m": v8_img,
            "YOLOv9c": v9_img,
            "YOLOv10m": v10_img
        }

        for i, model_name in enumerate(model_order):
            with [viz_col1, viz_col2, viz_col3][i]:
                 st.image(image_map[model_name], caption=f"{model_name} | {results_data[model_name]['Classe_Detectada']} ({results_data[model_name]['Confianca_Maxima']:.3f})", use_container_width=True)

    else:
        st.warning("Nenhuma detecção encontrada pelos modelos na imagem enviada (Confiança < 25%).")

else:

    st.info("Aguardando o upload de uma imagem de Raio X para iniciar o comparativo de modelos.")

st.markdown("---")
st.markdown("### Visão Geral do Estudo (Dados de Treinamento)")
df_metrics = pd.DataFrame.from_dict(TREINO_METRICS, orient='index')

st.dataframe(df_metrics.style.apply(highlight_best_and_worst, axis=None), use_container_width=True)

st.markdown("""
- **Precisão (mAP@.5:.95):** O valor mais alto (Verde) é o modelo mais preciso.
- **Velocidade (Inf_Speed):** O valor mais baixo (Verde) é o modelo mais rápido.
""")
st.caption("A tabela mostra as métricas fixas obtidas durante o treinamento.")
