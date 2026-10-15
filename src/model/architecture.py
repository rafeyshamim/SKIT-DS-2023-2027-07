"""
3D CNN Architecture for volumetric medical image classification.

Sprint 2 — 3D CNN Architecture Design & Implementation.

Architecture
------------
Input tensor  :  (batch, D, H, W, C)
                  e.g. (None, 64, 64, 64, 1) from config.yaml

Feature extraction backbone (one block per entry in ``model.filters``):
    Conv3D(f, kernel, padding='same')  →  No bias (absorbed by BN)
    BatchNormalization
    ReLU
    MaxPooling3D(2, 2, 2)   ← omitted on the *last* block to avoid
                               over-downsampling small feature maps

Global aggregation:
    GlobalAveragePooling3D  →  vector of length = last filter count

Classification head:
    Dense(dense_units[0]) + ReLU + Dropout
    Dense(dense_units[1]) + ReLU + Dropout
    Dense(num_classes,  activation='softmax')

All parameters are driven from ``config/config.yaml`` via
:func:`get_model_from_config`.
"""
from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def conv3d_block(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int, int] = (3, 3, 3),
    l2_reg: float = 0.001,
    pool: bool = True,
    name_prefix: str = "block",
) -> tf.Tensor:
    """
    One 3-D convolutional block: Conv3D → BatchNorm → ReLU → (MaxPool3D).

    Args:
        inputs:      Input tensor.
        filters:     Number of Conv3D output filters.
        kernel_size: Convolution kernel dimensions.
        l2_reg:      L2 regularization factor applied to the Conv3D kernel.
        pool:        Whether to append a 2×2×2 MaxPooling3D layer.
        name_prefix: Layer-name prefix for readability in model summaries.

    Returns:
        Output tensor after this block.
    """
    x = layers.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,                   # BN has its own bias term
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name_prefix}_conv",
    )(inputs)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu")(x)

    if pool:
        x = layers.MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            name=f"{name_prefix}_pool",
        )(x)
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def build_3d_cnn(
    input_shape: Tuple[int, int, int, int],
    num_classes: int,
    filters: Optional[List[int]] = None,
    kernel_size: Tuple[int, int, int] = (3, 3, 3),
    dense_units: Optional[List[int]] = None,
    dropout_rate: float = 0.5,
    l2_reg: float = 0.001,
) -> keras.Model:
    """
    Construct and return the 3-D CNN model.

    Spatial downsampling schedule (example with input 64³ and filters [32,64,128,256]):

    +---------+-----------------------------+-------------------+
    | Block   | Operation                   | Output shape      |
    +=========+=============================+===================+
    | Input   | —                           | (64, 64, 64, 1)   |
    +---------+-----------------------------+-------------------+
    | Block 1 | Conv3D(32) + MaxPool3D(2)   | (32, 32, 32, 32)  |
    +---------+-----------------------------+-------------------+
    | Block 2 | Conv3D(64) + MaxPool3D(2)   | (16, 16, 16, 64)  |
    +---------+-----------------------------+-------------------+
    | Block 3 | Conv3D(128) + MaxPool3D(2)  | (8,  8,  8,  128) |
    +---------+-----------------------------+-------------------+
    | Block 4 | Conv3D(256)  [no pool]      | (8,  8,  8,  256) |
    +---------+-----------------------------+-------------------+
    | GAP     | GlobalAveragePooling3D      | (256,)            |
    +---------+-----------------------------+-------------------+
    | Dense 1 | Dense(512) + Dropout        | (512,)            |
    +---------+-----------------------------+-------------------+
    | Dense 2 | Dense(256) + Dropout        | (256,)            |
    +---------+-----------------------------+-------------------+
    | Output  | Dense(num_classes, softmax) | (num_classes,)    |
    +---------+-----------------------------+-------------------+

    Args:
        input_shape: ``(D, H, W, C)`` — spatial + channel dims.
        num_classes: Number of output disease/class categories.
        filters:     Filter counts per Conv3D block.
                     Defaults to ``[32, 64, 128, 256]``.
        kernel_size: Conv3D kernel dimensions.
        dense_units: Dense layer sizes in the classification head.
                     Defaults to ``[512, 256]``.
        dropout_rate: Dropout probability in the classification head.
        l2_reg:       L2 regularization coefficient on Conv3D kernels.

    Returns:
        Un-compiled :class:`keras.Model`.  Call
        :func:`src.model.trainer.ModelTrainer._compile_model` to compile.
    """
    if filters is None:
        filters = [32, 64, 128, 256]
    if dense_units is None:
        dense_units = [512, 256]

    logger.info(
        "Building 3D CNN | input_shape=%s | num_classes=%d | "
        "filters=%s | dense_units=%s",
        input_shape, num_classes, filters, dense_units,
    )

    inputs = keras.Input(shape=input_shape, name="volumetric_input")
    x = inputs

    # --- Feature extraction blocks -------------------------------------------
    for i, f in enumerate(filters):
        # The last block does NOT pool to avoid making feature maps too small.
        apply_pool = i < (len(filters) - 1)
        x = conv3d_block(
            x,
            filters=f,
            kernel_size=kernel_size,
            l2_reg=l2_reg,
            pool=apply_pool,
            name_prefix=f"block{i + 1}",
        )

    # --- Global spatial pooling -----------------------------------------------
    x = layers.GlobalAveragePooling3D(name="global_avg_pool")(x)

    # --- Classification head --------------------------------------------------
    for j, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"dense_{j + 1}",
        )(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{j + 1}")(x)

    # --- Output layer (softmax for multi-class classification) ----------------
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="3D_CNN_Medical")

    total_params = model.count_params()
    logger.info(
        "Model built successfully | Total parameters: {:,}".format(total_params)
    )
    return model


# ---------------------------------------------------------------------------
# Config-driven factory
# ---------------------------------------------------------------------------

def get_model_from_config(config: dict) -> keras.Model:
    """
    Instantiate the 3-D CNN model using parameters from the config dict.

    This is the recommended way to build the model in all scripts and tests.

    Args:
        config: Configuration dict returned by
                :func:`~src.utils.config_loader.load_config`.

    Returns:
        Un-compiled :class:`keras.Model`.
    """
    model_cfg = config["model"]
    data_cfg  = config["data"]

    input_shape  = tuple(model_cfg["input_shape"])          # (D, H, W, C)
    num_classes  = int(data_cfg["num_classes"])
    filters      = list(model_cfg.get("filters",      [32, 64, 128, 256]))
    kernel_size  = tuple(model_cfg.get("kernel_size", [3, 3, 3]))
    dense_units  = list(model_cfg.get("dense_units",  [512, 256]))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.5))
    l2_reg       = float(model_cfg.get("l2_reg",       0.001))

    return build_3d_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=filters,
        kernel_size=kernel_size,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
    )
