import os
import json
import math
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# ==============================
# CONFIG
# ==============================
DATASET_DIR_PRIORITY = ["dataset_aug", "dataset"]  # usa dataset_aug se existir
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 1337

MODEL_OUT_KERAS = "modelo_molas.keras"  # recomendado (Keras v3)
MODEL_OUT_H5 = "modelo_molas.h5"        # opcional (compat)
LABELS_OUT = "labels_molas.json"

# split
VAL_SPLIT = 0.2  # 20%


def pick_dataset_dir():
    for d in DATASET_DIR_PRIORITY:
        if os.path.isdir(d):
            return d
    return None


def count_images_per_class(root_dir):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    counts = {}
    total = 0
    for c in classes:
        cdir = os.path.join(root_dir, c)
        n = 0
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                n += 1
        counts[c] = n
        total += n
    return classes, counts, total


def ensure_non_empty_split(total_images):
    """
    Evita o problema clássico: poucas imagens + validation_split => val vazio.
    """
    if total_images < 10:
        print("\n[AVISO] Poucas imagens no dataset. Recomendo gerar mais (dataset_aug) antes de treinar.\n")

    # garante pelo menos 1 batch no treino e no val
    # (image_dataset_from_directory já cuida, mas o dataset pode ficar minúsculo)
    if total_images == 0:
        raise RuntimeError("Dataset vazio: nenhuma imagem encontrada.")


def save_labels_json(class_names):
    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"[OK] Labels salvos em: {LABELS_OUT} -> {class_names}")


def main():
    ds_dir = pick_dataset_dir()
    if ds_dir is None:
        raise RuntimeError("Nenhuma pasta de dataset encontrada. Crie 'dataset' ou 'dataset_aug'.")

    class_names, counts, total = count_images_per_class(ds_dir)
    print(f"\n[INFO] Dataset escolhido: {ds_dir}")
    print(f"[INFO] Classes encontradas: {class_names}")
    print(f"[INFO] Contagem por classe: {counts}")
    print(f"[INFO] Total imagens: {total}\n")

    if len(class_names) < 2:
        raise RuntimeError("Precisa de pelo menos 2 classes. Verifique as pastas dentro do dataset.")

    ensure_non_empty_split(total)

    tf.random.set_seed(SEED)

    # Carrega datasets com split fixo e seed
    train_ds = tf.keras.utils.image_dataset_from_directory(
        ds_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        ds_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="validation",
    )

    # Confere se não ficou vazio (blindagem extra)
    train_batches = tf.data.experimental.cardinality(train_ds).numpy()
    val_batches = tf.data.experimental.cardinality(val_ds).numpy()
    print(f"[INFO] Batches treino: {train_batches} | Batches validação: {val_batches}")
    if train_batches <= 0 or val_batches <= 0:
        raise RuntimeError(
            "Split gerou dataset vazio (treino ou validação). "
            "Gere mais imagens no dataset_aug ou reduza o VAL_SPLIT."
        )

    class_names = train_ds.class_names
    save_labels_json(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data augmentation (leve) dentro do modelo (mais robusto)
    data_aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.08),
        ],
        name="data_augmentation",
    )

    # Base MobileNetV2
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # primeiro treina só o topo

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_aug(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_OUT_KERAS,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n[INFO] Treinando (fase 1 - topo)...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Fine-tuning leve (opcional) – libera um pedaço do backbone
    print("\n[INFO] Fine-tuning (fase 2 - ajustando últimas camadas)...\n")
    base.trainable = True
    # congela as camadas iniciais e libera só o final
    fine_tune_at = int(len(base.layers) * 0.75)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ft_callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_OUT_KERAS,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(5, EPOCHS // 2),
        callbacks=ft_callbacks,
        verbose=1,
    )

    # garante que o arquivo final exista e seja válido
    model.save(MODEL_OUT_KERAS)
    print(f"\n[OK] Modelo salvo (Keras): {MODEL_OUT_KERAS}")

    # opcional: exporta .h5 (compat)
    try:
        model.save(MODEL_OUT_H5)
        print(f"[OK] Modelo salvo (H5): {MODEL_OUT_H5}")
    except Exception as e:
        print(f"[AVISO] Não consegui salvar H5: {e}")

    print("\n✅ Treinamento finalizado.\n")


if __name__ == "__main__":
    main()
MODEL_PATH = "modelo_molas.keras"
model.save(MODEL_PATH)
