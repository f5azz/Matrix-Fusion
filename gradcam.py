import numpy as np
import tensorflow as tf
import cv2


def get_gradcam(model, img_array):

    # ✅ Get base model (MobileNetV2)
    base_model = model.layers[0]

    # ✅ Forward pass through base model
    conv_outputs = base_model(img_array)

    # ✅ Convert to numpy
    conv_outputs = conv_outputs.numpy()[0]

    # ✅ Create simple heatmap (channel-wise importance)
    heatmap = np.mean(conv_outputs, axis=-1)

    # ✅ Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap


def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 🔥 FIX: ensure proper scaling
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed