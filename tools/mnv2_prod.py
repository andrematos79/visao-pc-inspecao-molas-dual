import os
import json
import numpy as np
import tensorflow as tf

def load_production_package(model_dir: str):
    pkg_path = os.path.join(model_dir, "production_package.json")
    if not os.path.isfile(pkg_path):
        raise FileNotFoundError(f"production_package.json não encontrado em: {pkg_path}")

    with open(pkg_path, "r", encoding="utf-8") as f:
        pkg = json.load(f)

    class_names = pkg["class_names"]
    pos_name = pkg["pos_class_name"]
    pos_idx = int(pkg["pos_class_index"])
    thr_ng = float(pkg["best_threshold_ng"]["thr"])
    img_size = tuple(pkg.get("img_size", [224, 224]))

    return class_names, pos_name, pos_idx, thr_ng, img_size


def load_model_prod(model_dir: str):
    model_path = os.path.join(model_dir, "model_final.keras")
    if not os.path.isfile(model_path):
        alt = os.path.join(model_dir, "best_model.keras")
        if os.path.isfile(alt):
            model_path = alt
        else:
            raise FileNotFoundError(
                f"Modelo não encontrado em {model_dir}. Esperado model_final.keras (ou best_model.keras)."
            )
    model = tf.keras.models.load_model(model_path)
    return model, model_path


def infer_prod(bgr_img: np.ndarray,
              model: tf.keras.Model,
              class_names: list,
              pos_idx: int,
              thr_ng: float,
              img_size=(224, 224)):
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Imagem vazia na inferência.")

    # BGR -> RGB
    rgb = bgr_img[..., ::-1]

    # Resize
    x = tf.image.resize(rgb, img_size, method="bilinear")
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, axis=0)  # (1,H,W,3)

    # IMPORTANTE: não aplicar preprocess_input aqui
    # porque no treino ele já foi embutido no modelo.
    p = model.predict(x, verbose=0)[0]
    p = np.asarray(p, dtype=np.float32)

    prob_ng = float(p[pos_idx])
    probs = {class_names[i]: float(p[i]) for i in range(len(class_names))}
    pred_label = "NG_MISALIGNED" if prob_ng >= thr_ng else "OK"
    return pred_label, prob_ng, probs