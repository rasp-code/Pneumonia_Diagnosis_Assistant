import os

# --- LIGNE MAGIQUE ---
# Doit être placée AVANT l'import de TensorFlow.
# Cela rend le GPU totalement invisible pour le script.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

# --- PARAMÈTRES ---
CHEMIN_MODELE = "./artifacts/artifacts/outputs/pneumonia_model.keras" 
DOSSIER_TEST = "./datasets/ml/test" 
CLASSES = ["NORMAL", "PNEUMONIA"]

# 1. CHARGEMENT DU MODÈLE
print("Chargement du modèle Keras en mode CPU...")
try:
    model = tf.keras.models.load_model(CHEMIN_MODELE)
    print("Modèle prêt pour l'évaluation en masse !\n")
except Exception as e:
    print(f"[ERREUR] Problème lors du chargement du modèle : {e}")
    exit()

# Variables pour les statistiques
total_images = 0
predictions_correctes = 0

print("==========================================")
print(" DÉBUT DE L'ANALYSE DU DATASET DE TEST")
print("==========================================")

# 2. PARCOURS AUTOMATIQUE DES DOSSIERS
for classe_reelle in CLASSES:
    chemin_dossier_classe = os.path.join(DOSSIER_TEST, classe_reelle)
    
    if not os.path.exists(chemin_dossier_classe):
        print(f"⚠️  Dossier introuvable : {chemin_dossier_classe}")
        continue
        
    fichiers = os.listdir(chemin_dossier_classe)
    print(f"\n--- Analyse du dossier : {classe_reelle} ({len(fichiers)} fichiers) ---")
    
    for fichier in fichiers:
        if not fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        chemin_image = os.path.join(chemin_dossier_classe, fichier)
        
        try:
            # Traitement de l'image
            img = tf.keras.utils.load_img(chemin_image, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            input_tensor = np.expand_dims(img_array, axis=0) 
            
            # Prédiction
            output = model.predict(input_tensor, verbose=0)
            
            index_prediction = np.argmax(output[0])
            prediction_ia = CLASSES[index_prediction]
            
            # Vérification
            total_images += 1
            if prediction_ia == classe_reelle:
                predictions_correctes += 1
                statut = "✅ SUCCÈS"
            else:
                statut = "❌ ÉCHEC "
                
            print(f"{statut} | Fichier: {fichier} | Prédiction: {prediction_ia}")
            
        except Exception as e:
            print(f"⚠️ Erreur de lecture sur {fichier}: {e}")

# 3. BILAN FINAL
print("\n==========================================")
print(" BILAN DE L'ÉVALUATION")
print("==========================================")
if total_images > 0:
    precision = (predictions_correctes / total_images) * 100
    print(f"Images analysées    : {total_images}")
    print(f"Bonnes prédictions  : {predictions_correctes}")
    print(f"Précision (Accuracy): {precision:.2f}%")
else:
    print("Aucune image n'a été trouvée ou analysée. Vérifiez le chemin DOSSIER_TEST.")
print("==========================================")