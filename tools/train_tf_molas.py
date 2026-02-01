# train_tf_molas.py
from pathlib import Path
import json
import os
import numpy as np
import tensorflow as tf

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

# Dataset splitado (train/val/test)
DATASET_DIR = BASE_DIR / "dataset_mola_roi_split"  # <-- ajuste se o nome for outro
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR   = DATASET_DIR / "val"
TEST_DIR  = DATASET_DIR / "test"

# 2 classes (ordem fixa!)
CLASSES = ["mola_ausente", "mola_presente"]

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

MODEL_OUT  = BASE_DIR / "modelo_molas.keras"
LABELS_OUT = BASE_DIR / "labels.json"
CM_PNG_OUT = BASE_DIR / "confusion_matrix_test.png"

# =========================
# HELPERS
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(root: Path, classes):
    counts = {}
    total = 0
    for c in classes:
        p = root / c
        n = 0
        if p.exists():
            n = sum(1 for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
        counts[c] = n
        total += n
    return counts, total

def make_ds(dir_path: Path, shuffle: bool):
    ds = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        labels="inferred",
        class_names=CLASSES,     # garante mapeamento correto
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    return ds

def build_model(num_classes: int):
    # Augmentation leve (robustez de linha)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomBrightness(factor=0.10),
        tf.keras.layers.RandomContrast(factor=0.10),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base

def save_confusion_matrix_png(cm: np.ndarray, labels, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(111)
    ax.imshow(cm)
    ax.set_title("Confusion Matrix (TEST)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# =========================
# MAIN
# =========================
def main():
    print("BASE_DIR:", BASE_DIR)
    print("DATASET_DIR:", DATASET_DIR)

    for name, p in [("TRAIN_DIR", TRAIN_DIR), ("VAL_DIR", VAL_DIR), ("TEST_DIR", TEST_DIR)]:
        if not p.exists():
            raise FileNotFoundError(f"Pasta não existe: {p} (esperado {name})")

    tr_counts, tr_total = count_images(TRAIN_DIR, CLASSES)
    va_counts, va_total = count_images(VAL_DIR, CLASSES)
    te_counts, te_total = count_images(TEST_DIR, CLASSES)

    print("\n== CONTAGEM ==")
    print("train:", tr_counts, "total:", tr_total)
    print("val  :", va_counts, "total:", va_total)
    print("test :", te_counts, "total:", te_total)

    if tr_total == 0 or va_total == 0 or te_total == 0:
        raise RuntimeError("Tem split com zero imagens. Confere as pastas train/val/test e as classes.")

    train_ds = make_ds(TRAIN_DIR, shuffle=True)
    val_ds   = make_ds(VAL_DIR, shuffle=False)
    test_ds  = make_ds(TEST_DIR, shuffle=False)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    model, base = build_model(num_classes=len(CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # class_weight = total / (num_classes * count_da_classe)
    class_weight = {}
    for i, c in enumerate(CLASSES):
        n = tr_counts[c]
        class_weight[i] = (tr_total / (len(CLASSES) * max(1, n)))
    print("\nclass_weight:", class_weight)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    print("\n== TREINO (BASE CONGELADA) ==")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # Fine-tuning leve (opcional, mas recomendo)
    print("\n== FINE-TUNING (últimas camadas) ==")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        verbose=1
    )

    model.save(MODEL_OUT)
    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo salvo em: {MODEL_OUT}")
    print(f"✅ Labels salvos em: {LABELS_OUT}")

    print("\n== AVALIANDO NO TEST ==")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    y_true, y_pred = [], []
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(batch_y.numpy().tolist())
        y_pred.extend(preds.tolist())

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(CLASSES)).numpy()
    print("\nConfusion matrix (TEST):")
    print(cm)

    save_confusion_matrix_png(cm, CLASSES, CM_PNG_OUT)
    print(f"\n✅ Confusion matrix PNG salva em: {CM_PNG_OUT}")

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    main()
