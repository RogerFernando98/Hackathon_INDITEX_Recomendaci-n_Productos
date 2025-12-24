import json
import pandas as pd
import os

# Rutas
BASE_PATH = "/Users/fernandoarroyo/Desktop/Inditex/hackathon-inditex-data-recommender"
PREDICTIONS_PATH = os.path.join(BASE_PATH, "predictions/predictions_3.json")
TEST_PATH = os.path.join(BASE_PATH, "data/processed")

def verify_predictions():
    # 1. Cargar predicciones
    print("\n1. Verificando archivo de predicciones...")
    with open(PREDICTIONS_PATH, 'r') as f:
        preds = json.load(f)
    
    # 2. Verificaciones básicas de formato
    print("\n2. Verificaciones básicas:")
    print(f"- Tiene la clave 'target'?: {'target' in preds}")
    print(f"- Número de sesiones: {len(preds['target'])}")
    
    # 3. Verificar formato detallado
    print("\n3. Verificando formato de algunas predicciones:")
    for session_id, recs in list(preds['target'].items())[:3]:
        print(f"\nSession ID {session_id}:")
        print(f"- Tipo de session_id: {type(session_id)}")
        print(f"- Número de recomendaciones: {len(recs)}")
        print(f"- Tipo de productos: {type(recs[0])}")
        print(f"- Productos únicos?: {len(recs) == len(set(recs))}")
        print(f"- Ejemplo de recomendaciones: {recs}")
    
    # 4. Cargar datos de test para verificaciones adicionales
    print("\n4. Verificando cobertura de sesiones:")
    test_files = [f for f in os.listdir(TEST_PATH) if f.startswith('test_prepared')]
    if test_files:
        test_df = pd.read_parquet(os.path.join(TEST_PATH, test_files[-1]))
        test_sessions = set(test_df['session_id'].unique())
        pred_sessions = set(int(x) for x in preds['target'].keys())
        
        print(f"- Sesiones en test: {len(test_sessions)}")
        print(f"- Sesiones predichas: {len(pred_sessions)}")
        print(f"- Cobertura: {len(pred_sessions)/len(test_sessions)*100:.2f}%")
    
    # 5. Verificaciones estadísticas
    print("\n5. Estadísticas de recomendaciones:")
    all_recs = [rec for recs in preds['target'].values() for rec in recs]
    unique_products = len(set(all_recs))
    print(f"- Productos únicos recomendados: {unique_products}")
    print(f"- Total de recomendaciones: {len(all_recs)}")
    print(f"- Promedio de recomendaciones por sesión: {len(all_recs)/len(preds['target']):.2f}")

if __name__ == "__main__":
    verify_predictions()