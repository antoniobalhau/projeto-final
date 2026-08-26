import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from patchify import patchify
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

print("GPUs disponíveis:", tf.config.list_physical_devices("GPU"))
# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path("treino")

IMAGES_DIR = BASE_DIR / "imagens"
MASKS_DIR = BASE_DIR / "mascaras"

MODELS_DIR = Path("modelos")
MODELS_DIR.mkdir(exist_ok=True)

PATCH_SIZE = 256

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ============================================================
# CLASSES DA SEGMENTAÇÃO
# ============================================================

CLASS_COLORS = {
    "background": (0, 0, 0),
    "campo_agricola": (210, 210, 24),
    "floresta": (36, 179, 83),
    "agua": (45, 12, 212),
    "estrada": (255, 255, 255),
    "edificio": (245, 147, 49),
}

CLASS_NAMES = list(CLASS_COLORS.keys())
NUMBER_OF_CLASSES = len(CLASS_NAMES)

# Classe ignorada nas métricas e na loss
IGNORE_CLASS_ID = 0

SEGMENTATION_CMAP = ListedColormap(
    [
        tuple(channel / 255.0 for channel in CLASS_COLORS[name])
        for name in CLASS_NAMES
    ]
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def list_files(folder: Path, valid_extensions: set[str]) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"A pasta não existe: {folder}")

    files = [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions
    ]

    return sorted(files)


def create_file_map(files: list[Path]) -> dict[str, Path]:
    return {file.stem: file for file in files}


def find_image_mask_pairs() -> list[tuple[Path, Path]]:
    image_files = list_files(IMAGES_DIR, IMAGE_EXTENSIONS)
    mask_files = list_files(MASKS_DIR, MASK_EXTENSIONS)

    image_map = create_file_map(image_files)
    mask_map = create_file_map(mask_files)

    common_names = sorted(set(image_map) & set(mask_map))

    images_without_mask = sorted(set(image_map) - set(mask_map))
    masks_without_image = sorted(set(mask_map) - set(image_map))

    if images_without_mask:
        print("\nImagens sem máscara correspondente:")
        for name in images_without_mask:
            print(f"  - {image_map[name].name}")

    if masks_without_image:
        print("\nMáscaras sem imagem correspondente:")
        for name in masks_without_image:
            print(f"  - {mask_map[name].name}")

    if not common_names:
        raise RuntimeError(
            "Não foi encontrado nenhum par imagem/máscara.\n"
            "Confirma se os ficheiros têm o mesmo nome-base."
        )

    return [(image_map[name], mask_map[name]) for name in common_names]


def load_rgb_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"Não foi possível abrir a imagem: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_rgb_mask(mask_path: Path) -> np.ndarray:
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)

    if mask_bgr is None:
        raise ValueError(f"Não foi possível abrir a máscara: {mask_path}")

    return cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)


def crop_to_patch_size(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "A imagem e a máscara têm dimensões diferentes: "
            f"imagem={image.shape[:2]}, máscara={mask.shape[:2]}"
        )

    height, width = image.shape[:2]

    cropped_height = (height // patch_size) * patch_size
    cropped_width = (width // patch_size) * patch_size

    if cropped_height == 0 or cropped_width == 0:
        raise ValueError(
            f"A imagem tem dimensões inferiores a {patch_size}x{patch_size}: "
            f"{width}x{height}"
        )

    cropped_image = image[:cropped_height, :cropped_width]
    cropped_mask = mask[:cropped_height, :cropped_width]

    return cropped_image, cropped_mask


def rgb_mask_to_class_mask(mask_rgb: np.ndarray) -> np.ndarray:
    class_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    recognized_pixels = np.zeros(mask_rgb.shape[:2], dtype=bool)

    for class_id, class_name in enumerate(CLASS_NAMES):
        class_color = np.array(CLASS_COLORS[class_name], dtype=np.uint8)

        pixels_of_class = np.all(mask_rgb == class_color, axis=-1)

        class_mask[pixels_of_class] = class_id
        recognized_pixels |= pixels_of_class

    unrecognized_count = np.count_nonzero(~recognized_pixels)

    if unrecognized_count > 0:
        unique_unknown_colors = np.unique(
            mask_rgb[~recognized_pixels].reshape(-1, 3),
            axis=0,
        )

        print(
            f"\nAviso: foram encontrados {unrecognized_count} píxeis "
            "com cores não definidas."
        )

        print("Algumas cores RGB não reconhecidas:")

        for color in unique_unknown_colors[:10]:
            print(f"  - {tuple(int(value) for value in color)}")

    return class_mask


def create_patches(
    image: np.ndarray,
    class_mask: np.ndarray,
    patch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    image_patches = patchify(
        image,
        (patch_size, patch_size, 3),
        step=patch_size,
    )

    mask_patches = patchify(
        class_mask,
        (patch_size, patch_size),
        step=patch_size,
    )

    image_patch_list = []
    mask_patch_list = []

    number_of_rows = image_patches.shape[0]
    number_of_columns = image_patches.shape[1]

    for row in range(number_of_rows):
        for column in range(number_of_columns):
            image_patch = image_patches[row, column, 0]
            mask_patch = mask_patches[row, column]

            image_patch = image_patch.astype(np.float32) / 255.0
            mask_patch = mask_patch.astype(np.uint8)

            image_patch_list.append(image_patch)
            mask_patch_list.append(mask_patch)

    return image_patch_list, mask_patch_list


def split_image_mask_pairs(
    pairs,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
):
    if not np.isclose(
        train_ratio + validation_ratio + test_ratio,
        1.0,
    ):
        raise ValueError(
            "As percentagens de treino, validação e teste devem somar 1."
        )

    train_pairs, temporary_pairs = train_test_split(
        pairs,
        test_size=validation_ratio + test_ratio,
        random_state=random_state,
        shuffle=True,
    )

    test_fraction_of_temporary = (
        test_ratio / (validation_ratio + test_ratio)
    )

    validation_pairs, test_pairs = train_test_split(
        temporary_pairs,
        test_size=test_fraction_of_temporary,
        random_state=random_state,
        shuffle=True,
    )

    print("\nDivisão do dataset:")
    print(f"Treino: {len(train_pairs)} imagens")
    print(f"Validação: {len(validation_pairs)} imagens")
    print(f"Teste: {len(test_pairs)} imagens")

    return train_pairs, validation_pairs, test_pairs


def prepare_dataset_from_pairs(pairs, dataset_name):
    image_dataset = []
    mask_dataset = []

    print(f"\nA preparar o conjunto de {dataset_name}...")

    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        print(
            f"[{index}/{len(pairs)}] "
            f"{image_path.name} | {mask_path.name}"
        )

        image = load_rgb_image(image_path)
        mask_rgb = load_rgb_mask(mask_path)

        image, mask_rgb = crop_to_patch_size(
            image,
            mask_rgb,
            PATCH_SIZE,
        )

        class_mask = rgb_mask_to_class_mask(mask_rgb)

        image_patches, mask_patches = create_patches(
            image,
            class_mask,
            PATCH_SIZE,
        )

        image_dataset.extend(image_patches)
        mask_dataset.extend(mask_patches)

    images_array = np.asarray(image_dataset, dtype=np.float32)
    masks_array = np.asarray(mask_dataset, dtype=np.uint8)

    print(f"{dataset_name}: {len(images_array)} patches criados.")
    print(f"Formato das imagens: {images_array.shape}")
    print(f"Formato das máscaras: {masks_array.shape}")
    print(f"Classes encontradas: {np.unique(masks_array)}")

    return images_array, masks_array


def add_class_legend():
    legend_handles = [
        Patch(
            facecolor=tuple(channel / 255.0 for channel in CLASS_COLORS[name]),
            edgecolor="black",
            label=name,
        )
        for name in CLASS_NAMES
    ]

    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize="small",
    )


def show_random_examples(images, masks, number_of_examples=15):
    number_of_examples = min(number_of_examples, len(images))

    selected_indexes = random.sample(
        range(len(images)),
        number_of_examples,
    )

    for index in selected_indexes:
        plt.figure(figsize=(11, 5))

        plt.subplot(1, 2, 1)
        plt.title(f"Imagem - patch {index}")
        plt.imshow(images[index])
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Máscara - patch {index}")
        plt.imshow(
            masks[index],
            cmap=SEGMENTATION_CMAP,
            vmin=0,
            vmax=NUMBER_OF_CLASSES - 1,
        )
        plt.axis("off")
        add_class_legend()

        plt.tight_layout()
        plt.show()


# ============================================================
# LOSS E MÉTRICAS SEM BACKGROUND
# ============================================================

def masked_categorical_crossentropy(y_true, y_pred):
    y_true_ids = tf.argmax(y_true, axis=-1)

    valid_mask = tf.not_equal(y_true_ids, IGNORE_CLASS_ID)
    valid_mask = tf.cast(valid_mask, tf.float32)

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    loss = loss * valid_mask

    return tf.reduce_sum(loss) / (tf.reduce_sum(valid_mask) + K.epsilon())


def masked_accuracy(y_true, y_pred):
    y_true_ids = tf.argmax(y_true, axis=-1)
    y_pred_ids = tf.argmax(y_pred, axis=-1)

    valid_mask = tf.not_equal(y_true_ids, IGNORE_CLASS_ID)

    correct_predictions = tf.equal(y_true_ids, y_pred_ids)
    correct_predictions = tf.logical_and(correct_predictions, valid_mask)

    correct_predictions = tf.cast(correct_predictions, tf.float32)
    valid_mask = tf.cast(valid_mask, tf.float32)

    return tf.reduce_sum(correct_predictions) / (
        tf.reduce_sum(valid_mask) + K.epsilon()
    )


def masked_jaccard_coef(y_true, y_pred):
    y_true_ids = tf.argmax(y_true, axis=-1)
    y_pred_ids = tf.argmax(y_pred, axis=-1)

    iou_values = []

    for class_id in range(1, NUMBER_OF_CLASSES):
        true_class = tf.equal(y_true_ids, class_id)
        pred_class = tf.equal(y_pred_ids, class_id)

        intersection = tf.logical_and(true_class, pred_class)
        union = tf.logical_or(true_class, pred_class)

        intersection = tf.reduce_sum(tf.cast(intersection, tf.float32))
        union = tf.reduce_sum(tf.cast(union, tf.float32))

        iou = intersection / (union + K.epsilon())

        iou_values.append(iou)

    return tf.reduce_mean(tf.stack(iou_values))


# ============================================================
# MODELO DEEPLABV3+ SIMPLIFICADO
# ============================================================

def convolution_block(
    inputs,
    filters,
    kernel_size=3,
    dilation_rate=1,
):
    x = Conv2D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        activation="relu",
    )(inputs)

    x = Conv2D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)

    return x


def aspp_block(inputs, filters):
    """
    ASPP - Atrous Spatial Pyramid Pooling.
    Usa convoluções com diferentes dilation rates para captar contexto
    em várias escalas.
    """

    conv_1 = Conv2D(
        filters,
        1,
        padding="same",
        activation="relu",
    )(inputs)

    conv_6 = Conv2D(
        filters,
        3,
        padding="same",
        dilation_rate=6,
        activation="relu",
    )(inputs)

    conv_12 = Conv2D(
        filters,
        3,
        padding="same",
        dilation_rate=12,
        activation="relu",
    )(inputs)

    conv_18 = Conv2D(
        filters,
        3,
        padding="same",
        dilation_rate=18,
        activation="relu",
    )(inputs)

    x = concatenate([conv_1, conv_6, conv_12, conv_18])

    x = Conv2D(
        filters,
        1,
        padding="same",
        activation="relu",
    )(x)

    return x


def build_deeplabv3plus_model(
    image_height: int,
    image_width: int,
    image_channels: int,
    number_of_classes: int,
) -> Model:
    inputs = Input((image_height, image_width, image_channels))

    # ========================================================
    # Encoder simples
    # ========================================================

    # 256x256
    c1 = convolution_block(inputs, 32)
    low_level_features = c1

    # 128x128
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = convolution_block(p1, 64)

    # 64x64
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = convolution_block(p2, 128)
    c3 = Dropout(0.2)(c3)

    # 32x32
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = convolution_block(p3, 256)
    c4 = Dropout(0.3)(c4)

    # ========================================================
    # ASPP
    # ========================================================

    x = aspp_block(c4, 256)

    # ========================================================
    # Decoder
    # ========================================================

    # 32x32 -> 64x64
    x = Conv2DTranspose(
        128,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(x)

    x = convolution_block(x, 128)

    # 64x64 -> 128x128
    x = Conv2DTranspose(
        64,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(x)

    x = convolution_block(x, 64)

    # 128x128 -> 256x256
    x = Conv2DTranspose(
        32,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(x)

    # Features de baixo nível para recuperar detalhe espacial
    low_level_features = Conv2D(
        32,
        1,
        padding="same",
        activation="relu",
    )(low_level_features)

    x = concatenate([x, low_level_features])

    x = convolution_block(x, 64)

    outputs = Conv2D(
        number_of_classes,
        1,
        padding="same",
        activation="softmax",
    )(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# ============================================================
# TREINO
# ============================================================

def train_deeplabv3plus_model(
    train_images,
    train_masks,
    validation_images,
    validation_masks,
    epochs=100,
    batch_size=4,
):
    train_masks_cat = to_categorical(
        train_masks,
        num_classes=NUMBER_OF_CLASSES,
    )

    validation_masks_cat = to_categorical(
        validation_masks,
        num_classes=NUMBER_OF_CLASSES,
    )

    image_height = train_images.shape[1]
    image_width = train_images.shape[2]
    image_channels = train_images.shape[3]

    model = build_deeplabv3plus_model(
        image_height=image_height,
        image_width=image_width,
        image_channels=image_channels,
        number_of_classes=NUMBER_OF_CLASSES,
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=masked_categorical_crossentropy,
        metrics=[masked_accuracy, masked_jaccard_coef],
    )

    model.summary()

    checkpoint = ModelCheckpoint(
        filepath=MODELS_DIR / "melhor_modelo_deeplabv3plus.keras",
        monitor="val_masked_jaccard_coef",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    history = model.fit(
        train_images,
        train_masks_cat,
        validation_data=(validation_images, validation_masks_cat),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint],
    )

    return model, history


# ============================================================
# GRÁFICOS E RESULTADOS
# ============================================================

def plot_training_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Loss treino")
    plt.plot(history.history["val_loss"], label="Loss validação")
    plt.title("DeepLabV3+ - Training and validation loss")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history.history["masked_accuracy"],
        label="Accuracy treino sem background",
    )
    plt.plot(
        history.history["val_masked_accuracy"],
        label="Accuracy validação sem background",
    )
    plt.title("DeepLabV3+ - Accuracy sem background")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history.history["masked_jaccard_coef"],
        label="IoU/Jaccard treino sem background",
    )
    plt.plot(
        history.history["val_masked_jaccard_coef"],
        label="IoU/Jaccard validação sem background",
    )
    plt.title("DeepLabV3+ - IoU/Jaccard sem background")
    plt.xlabel("Épocas")
    plt.ylabel("IoU/Jaccard")
    plt.legend()
    plt.show()


def save_training_results(history):
    final_train_accuracy = history.history["masked_accuracy"][-1]
    final_val_accuracy = history.history["val_masked_accuracy"][-1]
    final_train_iou = history.history["masked_jaccard_coef"][-1]
    final_val_iou = history.history["val_masked_jaccard_coef"][-1]

    best_val_accuracy = max(history.history["val_masked_accuracy"])
    best_val_iou = max(history.history["val_masked_jaccard_coef"])

    best_val_accuracy_epoch = (
        history.history["val_masked_accuracy"].index(best_val_accuracy) + 1
    )

    best_val_iou_epoch = (
        history.history["val_masked_jaccard_coef"].index(best_val_iou) + 1
    )

    print("\n===== RESULTADOS FINAIS DEEPLABV3+ SEM BACKGROUND =====")
    print(f"Accuracy treino final: {final_train_accuracy * 100:.2f}%")
    print(f"Accuracy validação final: {final_val_accuracy * 100:.2f}%")
    print(f"IoU/Jaccard treino final: {final_train_iou * 100:.2f}%")
    print(f"IoU/Jaccard validação final: {final_val_iou * 100:.2f}%")

    print("\n===== MELHORES RESULTADOS DEEPLABV3+ SEM BACKGROUND =====")
    print(
        f"Melhor accuracy validação: "
        f"{best_val_accuracy * 100:.2f}% "
        f"na época {best_val_accuracy_epoch}"
    )
    print(
        f"Melhor IoU/Jaccard validação: "
        f"{best_val_iou * 100:.2f}% "
        f"na época {best_val_iou_epoch}"
    )

    with open(
        "resultados_treino_deeplabv3plus.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("===== RESULTADOS FINAIS DEEPLABV3+ SEM BACKGROUND =====\n")
        file.write(
            f"Accuracy treino final: "
            f"{final_train_accuracy * 100:.2f}%\n"
        )
        file.write(
            f"Accuracy validação final: "
            f"{final_val_accuracy * 100:.2f}%\n"
        )
        file.write(
            f"IoU/Jaccard treino final: "
            f"{final_train_iou * 100:.2f}%\n"
        )
        file.write(
            f"IoU/Jaccard validação final: "
            f"{final_val_iou * 100:.2f}%\n\n"
        )

        file.write("===== MELHORES RESULTADOS DEEPLABV3+ SEM BACKGROUND =====\n")
        file.write(
            f"Melhor accuracy validação: "
            f"{best_val_accuracy * 100:.2f}% "
            f"na época {best_val_accuracy_epoch}\n"
        )
        file.write(
            f"Melhor IoU/Jaccard validação: "
            f"{best_val_iou * 100:.2f}% "
            f"na época {best_val_iou_epoch}\n"
        )

    print("\nResultados guardados em: resultados_treino_deeplabv3plus.txt")


# ============================================================
# AVALIAÇÃO
# ============================================================

def evaluate_model_with_mean_iou(
    model,
    test_images,
    test_masks,
):
    predictions = model.predict(test_images)
    predicted_masks = np.argmax(predictions, axis=-1)

    print("\nClasses reais no teste:", np.unique(test_masks))
    print("Classes previstas pelo modelo:", np.unique(predicted_masks))

    mean_iou_all = MeanIoU(num_classes=NUMBER_OF_CLASSES)
    mean_iou_all.update_state(test_masks, predicted_masks)

    print(
        "\nMean IoU no conjunto de teste incluindo background:",
        mean_iou_all.result().numpy(),
    )

    iou_values = []

    for class_id in range(1, NUMBER_OF_CLASSES):
        true_class = test_masks == class_id
        pred_class = predicted_masks == class_id

        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()

        if union > 0:
            iou = intersection / union
            iou_values.append(iou)

            print(
                f"IoU classe {class_id} ({CLASS_NAMES[class_id]}): "
                f"{iou * 100:.2f}%"
            )
        else:
            print(
                f"IoU classe {class_id} ({CLASS_NAMES[class_id]}): "
                "classe ausente no teste"
            )

    if iou_values:
        mean_iou_no_background = np.mean(iou_values)

        print(
            "\nMean IoU no conjunto de teste sem background:",
            mean_iou_no_background,
        )

    return predicted_masks


def show_prediction_examples(
    test_images,
    test_masks,
    predicted_masks,
    number_of_examples=4,
):
    number_of_examples = min(number_of_examples, len(test_images))

    selected_indexes = random.sample(
        range(len(test_images)),
        number_of_examples,
    )

    for index in selected_indexes:
        plt.figure(figsize=(16, 5))

        plt.subplot(1, 3, 1)
        plt.title(f"Imagem original - teste {index}")
        plt.imshow(test_images[index])
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Máscara correta/anotada")
        plt.imshow(
            test_masks[index],
            cmap=SEGMENTATION_CMAP,
            vmin=0,
            vmax=NUMBER_OF_CLASSES - 1,
        )
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Previsão DeepLabV3+")
        plt.imshow(
            predicted_masks[index],
            cmap=SEGMENTATION_CMAP,
            vmin=0,
            vmax=NUMBER_OF_CLASSES - 1,
        )
        plt.axis("off")
        add_class_legend()

        plt.tight_layout()
        plt.show()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    print("Modelo: DeepLabV3+ simplificado")
    print("Pasta de imagens:", IMAGES_DIR)
    print("Pasta de máscaras:", MASKS_DIR)

    pairs = find_image_mask_pairs()

    train_pairs, validation_pairs, test_pairs = split_image_mask_pairs(
        pairs,
    )

    train_images, train_masks = prepare_dataset_from_pairs(
        train_pairs,
        "treino",
    )

    validation_images, validation_masks = prepare_dataset_from_pairs(
        validation_pairs,
        "validação",
    )

    test_images, test_masks = prepare_dataset_from_pairs(
        test_pairs,
        "teste",
    )

    print("\nResumo final:")

    print(
        f"Treino: {train_images.shape}, {train_masks.shape}"
    )

    print(
        f"Validação: {validation_images.shape}, "
        f"{validation_masks.shape}"
    )

    print(
        f"Teste: {test_images.shape}, {test_masks.shape}"
    )

    print("\nClasses no treino:", np.unique(train_masks))
    print("Classes na validação:", np.unique(validation_masks))
    print("Classes no teste:", np.unique(test_masks))

    show_random_examples(
        train_images,
        train_masks,
        number_of_examples=15,
    )

    # ========================================================
    # TREINO DO DEEPLABV3+
    # ========================================================

    model, history = train_deeplabv3plus_model(
        train_images=train_images,
        train_masks=train_masks,
        validation_images=validation_images,
        validation_masks=validation_masks,
        epochs=100,
        batch_size=4,
    )

    plot_training_history(history)

    save_training_results(history)

    model.save(MODELS_DIR / "ultimo_modelo_deeplabv3plus.keras")

    print(
        "\nÚltimo modelo guardado como: "
        "modelos/ultimo_modelo_deeplabv3plus.keras"
    )
    print(
        "Melhor modelo guardado como: "
        "modelos/melhor_modelo_deeplabv3plus.keras"
    )

    # ========================================================
    # AVALIAÇÃO COM O MELHOR MODELO
    # ========================================================

    best_model = load_model(
        MODELS_DIR / "melhor_modelo_deeplabv3plus.keras",
        custom_objects={
            "masked_categorical_crossentropy": masked_categorical_crossentropy,
            "masked_accuracy": masked_accuracy,
            "masked_jaccard_coef": masked_jaccard_coef,
        },
    )

    predicted_test_masks = evaluate_model_with_mean_iou(
        model=best_model,
        test_images=test_images,
        test_masks=test_masks,
    )

    show_prediction_examples(
        test_images=test_images,
        test_masks=test_masks,
        predicted_masks=predicted_test_masks,
        number_of_examples=4,
    )