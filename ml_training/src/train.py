import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, Dropout
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Chemin du dataset fourni par Azure")
    args = parser.parse_args()

    print(f"Le chemin du dataset reçu d'Azure est : {args.data_dir}")

    # --- PARAMÈTRES ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32  # Batch size augmenté pour une meilleure stabilité de gradient
    EPOCHS_PHASE1 = 3
    EPOCHS_PHASE2 = 7

    # 1. CHARGEMENT DES DONNÉES
    print("Chargement du dataset d'entraînement (avec division de validation à 20%)...")
    # Division train/val sur le dossier d'entraînement pour obtenir une validation robuste (~1043 images)
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(args.data_dir, 'train'),
        validation_split=0.2,
        subset="training",
        seed=123,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='categorical'
    )

    val_dataset_split = tf.keras.utils.image_dataset_from_directory(
        os.path.join(args.data_dir, 'train'),
        validation_split=0.2,
        subset="validation",
        seed=123,
        shuffle=False,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='categorical'
    )

    print("Chargement du dataset de validation original (pour suivi)...")
    val_dataset_original = tf.keras.utils.image_dataset_from_directory(
        os.path.join(args.data_dir, 'val'),
        shuffle=False,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='categorical'
    )

    class_names = train_dataset.class_names
    print(f"Classes détectées : {class_names}")

    # Calcul dynamique des poids de classe pour gérer le déséquilibre du dataset
    normal_dir = os.path.join(args.data_dir, 'train', 'NORMAL')
    pneumonia_dir = os.path.join(args.data_dir, 'train', 'PNEUMONIA')
    
    num_normal = len([f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(normal_dir) else 1341
    num_pneumonia = len([f for f in os.listdir(pneumonia_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(pneumonia_dir) else 3875
    total_train = num_normal + num_pneumonia

    weight_for_0 = (1 / num_normal) * (total_train / 2.0)
    weight_for_1 = (1 / num_pneumonia) * (total_train / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Poids de classes calculés : {class_weight} (Normal: {num_normal}, Pneumonia: {num_pneumonia})")

    # Optimisation du chargement en mémoire
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset_split = val_dataset_split.prefetch(buffer_size=AUTOTUNE)
    val_dataset_original = val_dataset_original.prefetch(buffer_size=AUTOTUNE)

    # 2. CRÉATION DU MODÈLE (DenseNet121)
    print("\nCréation du modèle DenseNet121 avec augmentation de données...")
    
    # Couches d'augmentation de données intégrées
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.15),
        RandomZoom(0.15),
    ])

    preprocess_input = tf.keras.applications.densenet.preprocess_input

    # Chargement de la base pré-entraînée (gelée par défaut)
    base_model = DenseNet121(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False

    # Assemblage du réseau
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)  # Maintient BatchNormalization en mode inférence
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(class_names), activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)

    # Vérification GPU
    print("\nVérification GPU :", "Actif 🟢" if tf.config.list_physical_devices('GPU') else "Inactif 🔴 (CPU)")

    # 3. PHASE 1 : Entraînement de la tête (Warm-up)
    print(f"\n--- PHASE 1 : Entraînement de la tête (Warm-up) sur {EPOCHS_PHASE1} époques ---")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_dataset,
        validation_data=val_dataset_split,
        epochs=EPOCHS_PHASE1,
        class_weight=class_weight
    )

    # 4. PHASE 2 : Fine-tuning global
    print(f"\n--- PHASE 2 : Fine-tuning global sur {EPOCHS_PHASE2} époques ---")
    base_model.trainable = True  # Dégel du modèle de base

    # Recompilation avec un taux d'apprentissage très faible pour ne pas détruire les poids pré-entraînés
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callback pour réduire le taux d'apprentissage en cas de stagnation
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset_split,
        epochs=EPOCHS_PHASE2,
        class_weight=class_weight,
        callbacks=[reduce_lr]
    )

    # 5. ÉVALUATIONS FINALES
    print("\n--- ÉVALUATIONS FINALES ---")
    val_loss, val_acc = model.evaluate(val_dataset_original, verbose=0)
    print(f"Dataset de validation original (16 images) -> Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")

    # 6. SAUVEGARDE POUR AZURE ML
    os.makedirs('outputs', exist_ok=True)
    save_path = 'outputs/pneumonia_model.keras'
    model.save(save_path)
    print(f"\nModèle sauvegardé avec succès sous : {save_path}")