import os
import asyncio
import requests
from flask import Flask, request, render_template
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

app = Flask(__name__)

MODE = os.getenv("MODE")

# ── Configuration CLOUD ──────────────────────────────────────────
ACV_ENDPOINT_URL = os.getenv("ACV_ENDPOINT_URL")
ACV_PREDICTION_KEY = os.getenv("ACV_PREDICTION_KEY")
AML_ENDPOINT_URL = os.getenv("AML_ENDPOINT_URL")
AML_API_KEY = os.getenv("AML_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Configuration EDGE ───────────────────────────────────────────
ACV_LOCAL_URL = os.getenv("ACV_LOCAL_URL")
AML_MODEL_PATH = os.getenv("AML_MODEL_PATH")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# ── Chargement du modèle Keras une seule fois au démarrage ───────
aml_model = None
if MODE == "edge":
    import tensorflow as tf
    aml_model = tf.keras.models.load_model(AML_MODEL_PATH)
    print(f"Modèle AML chargé depuis {AML_MODEL_PATH}")

ACV_THRESHOLD = 0.85
CONFIDENCE_THRESHOLD = 95  # en dessous : validation humaine recommandée


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 : Azure Computer Vision (Custom Vision) — validation image
# ══════════════════════════════════════════════════════════════════
def call_custom_vision(image_bytes: bytes) -> dict:
    """Retourne {'is_valide': bool, 'confidence': float}"""
    if MODE == "cloud":
        url = ACV_ENDPOINT_URL
        headers = {"Prediction-Key": ACV_PREDICTION_KEY, "Content-Type": "application/octet-stream"}
    else:
        url = ACV_LOCAL_URL
        headers = {"Content-Type": "application/octet-stream"}

    response = requests.post(url, headers=headers, data=image_bytes)
    response.raise_for_status()
    predictions = response.json().get("predictions", [])
    valide_pred = next(p for p in predictions if p["tagName"] == "valide")
    probability = valide_pred["probability"]
    return {
        "is_valide": probability > ACV_THRESHOLD,
        "confidence": probability,
    }


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 : Azure ML — diagnostic pneumonie/normal
# ══════════════════════════════════════════════════════════════════
def call_aml_model(image_bytes: bytes) -> dict:
    if MODE == "cloud":
        import base64
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        headers = {
            "Authorization": f"Bearer {AML_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(AML_ENDPOINT_URL, headers=headers, json={"data": encoded}, timeout=30)
        response.raise_for_status()
        result = response.json()
        return {"label": result["label"], "confidence": result["confidence"]}
    else:
        return run_local_aml_model(image_bytes)


def run_local_aml_model(image_bytes: bytes) -> dict:
    import numpy as np
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = tf.keras.utils.img_to_array(image)   # valeurs 0-255, comme le cloud
    arr = np.expand_dims(arr, axis=0)

    probs = aml_model.predict(arr, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    classes_mapping = {0: "negatif", 1: "positif"}
    label = classes_mapping[predicted_idx]
    confidence = float(probs[predicted_idx])

    return {"label": label, "confidence": confidence}


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 : Agents Microsoft Agent Framework — génération du rapport
# ══════════════════════════════════════════════════════════════════
def get_chat_client() -> OpenAIChatClient:
    """
    Retourne un client de chat compatible OpenAI, pointé soit vers Groq (cloud),
    soit vers Ollama (edge, servi par le conteneur docker-compose).
    """
    if MODE == "cloud":
        return OpenAIChatClient(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
        )
    else:
        return OpenAIChatClient(
            api_key="ollama",  # placeholder, Ollama n'exige pas de clé
            base_url=f"{OLLAMA_HOST}/v1",
            model=OLLAMA_MODEL,
        )


async def _generate_report_async(acv_result: dict, aml_result: dict) -> str:
    client = get_chat_client()
    confidence = aml_result["confidence"]
    label = aml_result["label"]

    # ── Agent Triage ────────────────────────────────────────────
    # Pilote la pipeline et synthétise les données brutes des deux modèles
    agent_triage = Agent(
        client=client,
        name="Agent_Triage",
        instructions=(
            "Tu es l'Agent Triage d'un système d'analyse de radiographies pulmonaires par IA. "
            "Tu reçois les résultats bruts des deux modèles d'analyse (Computer Vision et Azure ML) "
            "et tu synthétises ces données en un résumé technique structuré pour le radiologue. "
            "Présente les données clairement : résultat de validation de l'image, "
            "diagnostic IA, niveau de confiance "
            "Sois factuel et concis, sans interprétation médicale."
        ),
    )

    # ── Agent Radiologue ────────────────────────────────────────
    # Traduit le score brut en pré-rapport médical structuré
    agent_radiologue = Agent(
        client=client,
        name="Agent_Radiologue",
        instructions=(
            "Tu es l'Agent Radiologue d'un système d'analyse IA. "
            "Tu reçois un résumé technique de l'Agent Triage et tu le traduis "
            "en un pré-rapport médical structuré destiné au patient. "
            "Commence par préciser que ceci est une analyse par IA. "
            "Annonce clairement si le test de pneumonie est positif ou négatif. "
            "Si le résultat est positif, adapte le ton pour ne pas brusquer le patient. "
            "Rédige en français, moins de 8 lignes, sans inventer d'informations."
        ),
    )

    # ── Agent Critique ──────────────────────────────────────────
    # Vérifie la cohérence et exige une validation humaine si nécessaire
    agent_critique = Agent(
        client=client,
        name="Agent_Critique",
        instructions=(
            "Tu es l'Agent Critique d'un système d'analyse IA médicale. "
            "Tu reçois le pré-rapport de l'Agent Radiologue et tu vérifies sa cohérence. "
            f"Si le niveau de confiance du modèle est inférieur à {CONFIDENCE_THRESHOLD}%, tu dois "
            "OBLIGATOIREMENT ajouter en fin de rapport une mention explicite "
            f"disant que le niveau de confiance est inférieur à {CONFIDENCE_THRESHOLD}% et donc demandant "
            "une validation par un médecin spécialiste, avec des coordonnées fictives entre crochets. "
            f"Si le niveau de confiance est supérieur ou égal à {CONFIDENCE_THRESHOLD}%, tu valides le "
            "rapport et indiques que les résultats sont clairs. "
            "Retourne uniquement le rapport final corrigé et complété, sans commentaire "
            "sur ton propre travail."
        ),
    )

    # ── Tour 1 : Triage synthétise les données brutes ───────────
    message_triage = (
        f"Données brutes à synthétiser :\n"
        f"- Validation image (Custom Vision) : {acv_result['confidence']*100:.1f}% de confiance — image valide\n"
        f"- Diagnostic pneumonie (Azure ML) : {label} ({confidence*100:.1f}% de confiance)\n"
        f"Produis le résumé technique pour l'Agent Radiologue."
    )
    result_triage = await agent_triage.run(message_triage)
    resume_triage = str(result_triage)

    # ── Tour 2 : Radiologue rédige le pré-rapport ───────────────
    result_radiologue = await agent_radiologue.run(resume_triage)
    pre_rapport = str(result_radiologue)

    # ── Tour 3 : Critique valide et complète le rapport ─────────
    message_critique = (
        f"{pre_rapport}\n\n"
        f"[Métadonnées pour vérification — niveau de confiance du modèle : {confidence*100:.1f}%]"
    )
    result_critique = await agent_critique.run(message_critique)
    rapport_final = str(result_critique)

    return rapport_final


def generate_report(acv_result: dict, aml_result: dict) -> str:
    """Wrapper synchrone : Flask est sync, agent_framework est async."""
    return asyncio.run(_generate_report_async(acv_result, aml_result))


# ══════════════════════════════════════════════════════════════════
# ROUTES FLASK
# ══════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", mode=MODE)


@app.route("/analyse", methods=["POST"])
def analyse():
    if "image" not in request.files:
        return render_template("index.html", error="Aucune image fournie.", mode=MODE)

    file = request.files["image"]
    image_bytes = file.read()

    # Étape 1 : validation Custom Vision
    try:
        acv_result = call_custom_vision(image_bytes)
    except Exception as e:
        return render_template("index.html", error=f"Erreur Custom Vision : {e}", mode=MODE)

    if not acv_result["is_valide"]:
        return render_template(
            "index.html",
            error=(
                f"L'image fournie ne semble pas être une radio thoracique "
                f"(confiance : {acv_result['confidence']*100:.1f}%)."
            ),
            mode=MODE,
        )

    # Étape 2 : diagnostic Azure ML
    try:
        aml_result = call_aml_model(image_bytes)
    except Exception as e:
        return render_template("index.html", error=f"Erreur Azure ML : {e}", mode=MODE)

    # Étape 3 : pipeline des 3 agents (Microsoft Agent Framework)
    try:
        rapport = generate_report(acv_result, aml_result)
    except Exception as e:
        return render_template("index.html", error=f"Erreur agent : {e}", mode=MODE)

    return render_template(
        "index.html",
        rapport=rapport,
        acv_confidence=f"{acv_result['confidence']*100:.1f}",
        aml_label=aml_result["label"],
        aml_confidence=f"{aml_result['confidence']*100:.1f}",
        mode=MODE,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)