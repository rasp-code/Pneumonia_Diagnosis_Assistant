from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.constants import AssetTypes
# IMPORTATION SUPPLÉMENTAIRE ICI
from azure.ai.ml.entities import Environment 

# 1. Connexion à votre Azure ML EPITA
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="355e05d1-b680-4c75-83a3-0a115ebaebbd",
    resource_group_name="projet-epita",
    workspace_name="projet-epita-machine-learning"
)

# 2. Récupérer le Dataset
dataset_dossier = ml_client.data.get(name="pneumonia_images_dataset", version="1.0")

# 3. Configuration de l'environnement à la volée avec l'image Docker
custom_env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    name="docker-image-cpu-env",
    description="Environnement Docker de base pour calcul CPU"
)

# 4. Configuration du Job révisée
job = command(
    code="./src",  
    command="pip install tensorflow && python train.py ${{inputs.images}}", 
    inputs={"images": Input(type=AssetTypes.URI_FOLDER, path=dataset_dossier.id)},
    
    # On passe l'objet Environment créé juste au-dessus
    environment=custom_env, 
    
    compute="compute-classification", 
    experiment_name="epita-pneumonia-cpu"
)

# 5. Envoi du Job
print("Envoi du script d'entraînement sur votre machine CPU...")
returned_job = ml_client.jobs.create_or_update(job)

print("\n--- JOB ENVOYÉ AVEC SUCCÈS ---")
print(f"Suivez l'entraînement ici : {returned_job.studio_url}")