import json

# Cargar el archivo actual
with open('/Users/fernandoarroyo/Desktop/Inditex/hackathon-inditex-data-recommender/predictions/predictions_3.json', 'r') as f:
    predictions = json.load(f)

# Reformatear las predicciones
reformatted_predictions = {
    "target": {
        str(session_id): [int(prod) for prod in prods]
        for session_id, prods in predictions["target"].items()
    }
}

# Guardar el archivo reformateado
with open('/Users/fernandoarroyo/Desktop/Inditex/hackathon-inditex-data-recommender/predictions/predictions_3.json', 'w') as f:
    json.dump(reformatted_predictions, f, indent=4)