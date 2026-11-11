import os
import json
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image

# Variable globale pour stocker le modèle
model = None

# Mappage des classes vers le format attendu par la Flask app
CLASSES_MAPPING = {
    0: "normal",
    1: "pneumonie"
}

def init():
    """
    Initialise le conteneur Azure ML en chargeant le modèle entraîné.
    """
    global model
    # AZUREML_MODEL_DIR est défini par Azure ML lors du déploiement
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if not model_dir:
        # Fallback pour le développement local
        model_dir = "./artifacts/artifacts/outputs"
        
    print(f"Recherche du modèle dans le répertoire : {model_dir}")
    
    # Recherche récursive du fichier de modèle (.keras ou .h5)
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".keras") or file.endswith(".h5"):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break
            
    if not model_path:
        raise FileNotFoundError(f"Aucun modèle (.keras ou .h5) trouvé dans {model_dir}")
        
    print(f"Modèle trouvé à l'emplacement : {model_path}")
    
    # Chargement du modèle Keras
    model = tf.keras.models.load_model(model_path)
    print("Modèle chargé avec succès et prêt à l'emploi !")

def run(raw_data):
    """
    Traite la requête d'inférence.
    Soutient à la fois le format binaire (octet-stream) et le format JSON (base64).
    """
    print("Requête reçue pour prédiction...")
    try:
        image_bytes = None
        
        # 1. Vérifier si l'entrée est binaire (bytes) - utilisé par votre Flask app
        if isinstance(raw_data, bytes):
            image_bytes = raw_data
        # 2. Si l'entrée est une chaîne (JSON ou base64) - utilisé par Azure Studio Test
        elif isinstance(raw_data, str):
            try:
                # Essayer de parser comme JSON
                data = json.loads(raw_data)
                
                # Support de plusieurs clés JSON courantes
                base64_str = None
                if isinstance(data, dict):
                    if "data" in data:
                        base64_str = data["data"]
                    elif "image" in data:
                        base64_str = data["image"]
                    elif "input_data" in data and isinstance(data["input_data"], dict):
                        # Format standard de test Azure ML Studio
                        input_data = data["input_data"]
                        if "data" in input_data and len(input_data["data"]) > 0:
                            base64_str = input_data["data"][0]
                
                if base64_str:
                    image_bytes = base64.b64decode(base64_str)
                else:
                    return {"error": "Format JSON non supporté. Fournissez l'image base64 sous la clé 'data' ou 'image'."}
            except json.JSONDecodeError:
                # Si ce n'est pas du JSON, essayer de décoder directement comme base64
                try:
                    image_bytes = base64.b64decode(raw_data)
                except Exception:
                    pass
        
        if not image_bytes:
            return {"error": "Impossible de lire les données de l'image. Envoyez des octets bruts ou du base64."}
            
        # Prétraitement de l'image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Redimensionnement aux dimensions d'entraînement (224x224)
        image = image.resize((224, 224))
        
        # Conversion en tableau NumPy pour Keras
        img_array = tf.keras.utils.img_to_array(image)
        # Ajout de la dimension de batch (1, 224, 224, 3)
        input_tensor = np.expand_dims(img_array, axis=0)
        
        # Exécution de la prédiction
        predictions = model.predict(input_tensor, verbose=0)[0]
        
        # Récupération du diagnostic le plus probable
        predicted_idx = int(np.argmax(predictions))
        predicted_label = CLASSES_MAPPING[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Bilan détaillé pour logs
        print(f"Prédiction : {predicted_label} | Confiance : {confidence:.4f}")
        
        return {
            "label": predicted_label,
            "confidence": confidence
        }
        
    except Exception as e:
        import traceback
        err_msg = f"Erreur lors de l'inférence : {str(e)}\n{traceback.format_exc()}"
        print(err_msg)
        return {"error": str(e)}
