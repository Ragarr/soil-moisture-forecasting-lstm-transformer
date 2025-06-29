import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from models import (
    LSTM,
    Transformer,
    DummyModel,
    load_and_preprocess_data,
    generate_sequences,
    device
)

# =========================
# Helper functions
# =========================
def calculate_metrics(y_true, y_pred):
    """Calcula RMSE y R² entre y_true y y_pred."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def denormalize(y, max_val, min_val):
    """Convierte de rango normalizado [-1,1] a escala original."""
    return ((y + 1) / 2) * (max_val - min_val) + min_val

def predict_in_blocks(model, x_test, horizon, max_val, min_val):
    """
    Realiza predicciones en bloques de tamaño `horizon` sobre el conjunto x_test.
    Devuelve una lista de tuplas (start_idx, pred_block_denormalizado).
    """
    blocks = []
    for i in range(0, len(x_test) - horizon, horizon):
        with torch.no_grad():
            pred = model(x_test[i:i+1], horizon=horizon).cpu().numpy().flatten()
        pred = denormalize(pred, max_val, min_val)
        blocks.append((i, pred))
    return blocks

def plot_block_predictions(
    y_true,
    y_preds_blocks,
    horizon,
    title,
    save_path,
    rmse=None,
    r2=None
):
    """
    Dibuja y guarda la curva real y las predicciones por bloques.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True', color='black')
    colors = plt.cm.tab20(np.linspace(0, 1, len(y_preds_blocks)))

    for idx, (start_idx, pred_block) in enumerate(y_preds_blocks):
        plt.plot(
            range(start_idx, start_idx + horizon),
            pred_block,
            label=f'Block {idx + 1}',
            color=colors[idx % len(colors)]
        )

    if rmse is not None and r2 is not None:
        text = f"RMSE: {rmse:.2f}  •  R²: {r2:.2f}"
        plt.text(
            0.01, 0.99, text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    plt.ylim(y_true.min() - 5, y_true.max() + 5)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Moisture (%)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =========================
# Main script
# =========================
if __name__ == "__main__":
    # 1. Data path y detección dinámica de estaciones (archivos CSV)
    data_path = "intervals"
    depth = "0.5000"

    # Obtenemos la lista de archivos .csv en data_path (sin extensión)
    try:
        all_entries = os.listdir(data_path)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró la carpeta '{data_path}'.")
        exit(1)

    stations = [
        os.path.splitext(entry)[0]
        for entry in all_entries
        if entry.lower().endswith(".csv")
    ]

    if not stations:
        print(f"[ERROR] No se encontraron archivos '.csv' dentro de '{data_path}'.")
        exit(1)

    print(f"[INFO] Se detectaron {len(stations)} estaciones (archivos CSV):")
    for st in stations:
        print(f"       - {st}")

    # 2. Buscar todas las carpetas que empiecen por "experiment-"
    experiment_dirs = [
        d for d in glob.glob("experiment-*-*")
        if os.path.isdir(d)
    ]

    if not experiment_dirs:
        print("[ERROR] No se encontraron carpetas con el patrón 'experiment-*-*'.")
        exit(1)

    # 3. Procesar cada experimento
    for exp_dir in experiment_dirs:
        print(f"[INFO] Processing experiment folder: {exp_dir}")

        # Directorios dentro del experimento
        model_dir = os.path.join(exp_dir, "models")
        plot_dir = os.path.join(exp_dir, "plots", "best_trial")
        metrics_dir = os.path.join(exp_dir, "metrics")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        # Ruta al checkpoint "best_model.pt"
        ckpt_path = os.path.join(model_dir, "best_model.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[WARNING] No se encontró '{ckpt_path}'. Skipping this experiment.")
            continue

        # Cargar checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        params   = ckpt["params"]
        lookback = ckpt["lookback"]
        horizon  = ckpt["horizon"]
        max_val  = ckpt["max_val"]
        min_val  = ckpt["min_val"]

        # Detectar tipo de modelo
        state_keys = ckpt["model_state_dict"].keys()
        if any("transformer_encoder" in k for k in state_keys):
            model_type = "transformer"
        elif any("lstm" in k for k in state_keys):
            model_type = "lstm"
        elif len(state_keys) == 0:
            # Si no hay claves en el state_dict, asumimos DummyModel
            model_type = "dummy"
        else:
            print(f"[ERROR] Cannot determine model type for checkpoint in {exp_dir}.")
            continue

        # Reconstrucción del modelo
        if model_type == "transformer":
            model = Transformer(
                input_size=8,
                d_model=params["d_model"],
                nhead=params["nhead"],
                d_ff=params["d_ff"],
                num_layers=params["num_layers"],
                dropout=params.get("DROPOUT_RATE", 0.0)
            ).to(device)

        elif model_type == "lstm":
            model = LSTM(
                input_size=8,
                dropout=params.get("DROPOUT_RATE", 0.0)
            ).to(device)

        else:
            # DummyModel no tiene parámetros entrenables
            model = DummyModel().to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Listas para acumular métricas de cada estación
        all_rmses = []
        all_r2s = []

        # 4. Para cada estación detectada dinámicamente:
        for station in stations:
            print(f"    [INFO] Processing station: {station}")

            # Intentar cargar datos y generar secuencias de prueba
            try:
                x, y, _, _ = load_and_preprocess_data(data_path, station, depth)
            except Exception as e:
                print(f"    [WARNING] Failed to load data for station {station}: {e}")
                continue

            split = int(len(x) * 0.8)
            x_test, y_test = generate_sequences(x[split:], y[split:], lookback, horizon)
            if x_test.shape[0] == 0:
                print(f"    [WARNING] No se generaron muestras de test para station {station}.")
                continue

            y_true = denormalize(y_test.cpu().numpy()[:, 0], max_val, min_val)

            # 5. Predicción en bloques
            y_preds_blocks = predict_in_blocks(model, x_test, horizon, max_val, min_val)
            if not y_preds_blocks:
                print(f"    [WARNING] No predictions generated for station {station}.")
                continue

            # 6. Recortar y_true para graficar según el último bloque
            last_block_end = y_preds_blocks[-1][0] + horizon
            y_true_plot = y_true[:last_block_end].flatten()

            # 7. Calcular métricas RMSE y R² concatenando bloques
            y_pred_concat = np.zeros_like(y_true_plot)
            for start_idx, pred_block in y_preds_blocks:
                y_pred_concat[start_idx:start_idx + horizon] = pred_block.flatten()

            rmse, r2 = calculate_metrics(y_true_plot, y_pred_concat)
            all_rmses.append(rmse)
            all_r2s.append(r2)

            # 8. Graficar y guardar figura
            save_path = os.path.join(
                plot_dir,
                f"{station}_best_trial_{model_type}.png"
            )
            plot_block_predictions(
                y_true_plot,
                y_preds_blocks,
                horizon,
                title=f"{station} — Best Trial ({model_type.capitalize()})",
                save_path=save_path,
                rmse=rmse,
                r2=r2
            )
            print(f"    [INFO] Saved plot at: {save_path}")

        # 9. Una vez procesadas todas las estaciones, calcular media de métricas
        if all_rmses and all_r2s:
            mean_rmse = np.mean(all_rmses)
            mean_r2 = np.mean(all_r2s)
            metrics_file = os.path.join(metrics_dir, "mean_metrics.txt")
            with open(metrics_file, "w") as f:
                f.write(f"Mean RMSE across all stations: {mean_rmse:.4f}\n")
                f.write(f"Mean R²   across all stations: {mean_r2:.4f}\n")
            print(f"[INFO] Mean metrics saved at: {metrics_file}")
        else:
            print(f"[WARNING] No hubo métricas válidas para {exp_dir}. No se creó el archivo de métricas.")

    print("[INFO] All experiments and stations processed.")
