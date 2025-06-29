import os
import copy
import torch
import optuna
import torch.nn as nn
import numpy as np
from models import (
    LSTM,
    Transformer,
    DummyModel,
    load_and_preprocess_data,
    generate_sequences,
    device
)
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import logging

# =========================
# Argumentos y rutas base
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", type=str, default="experiment",
    help="Directorio raíz para logs, db, checkpoints, modelos y métricas"
)
parser.add_argument(
    "--horas", type=float, default=7,
    help="Duración máxima del estudio en horas"
)
parser.add_argument(
    "--model_type", type=str, choices=["lstm", "transformer", "dummy"], default="transformer",
    help="Tipo de modelo a entrenar: 'lstm', 'transformer' o 'dummy'"
)
parser.add_argument(
    "--horizon", type=int, default=24,
    help="Número de pasos a predecir"
)
args = parser.parse_args()

# Directorios
OUTPUT_DIR = args.output_dir + "-" + args.model_type + "-" + str(args.horizon)
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DB_DIR = os.path.join(OUTPUT_DIR, "db")
CKPT_ROOT = os.path.join(OUTPUT_DIR, "checkpoints")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
for d in [LOG_DIR, DB_DIR, CKPT_ROOT, MODEL_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

STUDY_DB = f"sqlite:///{os.path.join(DB_DIR, 'optuna_study.db')}"
STUDY_NAME = "train_resume_study"

# Parámetro horizon
horizon = args.horizon

# =========================
# Logging
# =========================
log_path = os.path.join(LOG_DIR, "optuna_training.log")
logging.basicConfig(
    filename=log_path,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.info("Iniciando estudio Optuna...")
print("Iniciando estudio Optuna...")

# =========================
# Carga de datos y partición
# =========================
print("Cargando y preprocesando datos...")
interval = "device_8988228066605450596_20240917_090000_to_20241023_220000"
data_path = os.path.join("data", "intervals")
depth = "0.5000"

x, y, max_val, min_val = load_and_preprocess_data(data_path, interval, depth)
print(f"Datos cargados: features {x.shape}, target {y.shape}")
split1 = int(len(x) * 0.6)
split2 = int(len(x) * 0.8)
print(f"Particiones: train hasta índice {split1}, val hasta índice {split2}, test el resto")

# =========================
# Función de entrenamiento
# =========================
def train_and_evaluate(
    model, x_train, y_train, x_val, y_val,
    LR, weight_decay, batch_size,
    max_epochs, patience_es, clip_grad_norm,
    trial_number
):
    print(f"\n[Trial {trial_number}] Inicializando entrenamiento")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_model_state = None
    best_val_loss = float('inf')
    no_improve = 0
    start_epoch = 1

    ckpt_dir = os.path.join(CKPT_ROOT, f"trial_{trial_number}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'train_ckpt.pt')

    # Reanudar si existe checkpoint
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        best_model_state = ckpt['best_model_state']
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        no_improve = ckpt['no_improve']
        print(f"[Trial {trial_number}] Reanudando desde epoch {start_epoch}, best_val_loss previo {best_val_loss:.4f}")

    for epoch in range(start_epoch, max_epochs+1):
        model.train()
        perm = torch.randperm(x_train.size(0))
        epoch_loss = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            pred = model(xb, horizon=yb.shape[1])
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= x_train.size(0)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val, horizon=y_val.shape[1])
            val_loss = loss_fn(val_pred, y_val).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model).state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience_es:
                print(f"[Trial {trial_number}] Early stopping en epoch {epoch}")
                break

        # Guardar checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'best_model_state': best_model_state,
            'best_val_loss': best_val_loss,
            'no_improve': no_improve
        }, ckpt_path)

    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

# =========================
# Métricas
# =========================
def compute_metrics(y_true, y_pred, max_val, min_val):
    y_true = ((y_true + 1)/2)*(max_val-min_val) + min_val
    y_pred = ((y_pred + 1)/2)*(max_val-min_val) + min_val
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    if y_true.ndim == 3:
        y_true = y_true.squeeze()
    if y_pred.ndim == 3:
        y_pred = y_pred.squeeze()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return r2 - 0.5 * rmse

# =========================
# Objetivo Optuna
# =========================
def objective(trial):
    # ------------------------
    # Primero: solo LOOKBACK si dummy
    # ------------------------
    if args.model_type == 'dummy':
        # Sugerir solo LOOKBACK entre 0 y 36 (inclusivo)
        LB = trial.suggest_int('LOOKBACK', 1, 36)

        # Generar secuencias para train, val y test
        x_train, y_train = generate_sequences(x[:split1], y[:split1], LB, horizon)
        x_val, y_val     = generate_sequences(x[split1:split2], y[split1:split2], LB, horizon)
        x_test, y_test   = generate_sequences(x[split2:], y[split2:], LB, horizon)

        # Si cualquiera de los tres conjuntos queda vacío, podar la trial
        if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
            raise optuna.exceptions.TrialPruned()

        # Instanciar y evaluar DummyModel (no se entrena)
        model = DummyModel().to(device)
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test, horizon=horizon)

        # Guardar estado del modelo para esta trial (opcional)
        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, f"trial_{trial.number}_dummy.pt")
        )

        # Registrar en Optuna el valor de LOOKBACK
        trial.set_user_attr("lookback", LB)

        # Devolver la métrica compuesta
        return compute_metrics(y_test, y_pred, max_val, min_val)

    # ------------------------
    # Si no es dummy, definimos todos los hiperparámetros usuales
    # ------------------------
    # Hiperparámetros comunes
    LR = trial.suggest_float('LR', 1e-5, 5e-3, log=True)
    WD = trial.suggest_float('WEIGHT_DECAY', 1e-7, 1e-4, log=True)
    DO = trial.suggest_float('DROPOUT_RATE', 0.05, 0.3)
    BS = trial.suggest_categorical('BATCH_SIZE', [16, 32])
    LB = trial.suggest_int('LOOKBACK', 1, 48)
    EPOCHS = trial.suggest_int('MAX_EPOCHS', 200, 800)
    PATIENCE = trial.suggest_int('PATIENCE_ES', 20, 60)
    CLIP = trial.suggest_float('CLIP_GRAD_NORM', 0.5, 5.0)

    x_train, y_train = generate_sequences(x[:split1], y[:split1], LB, horizon)
    x_val, y_val     = generate_sequences(x[split1:split2], y[split1:split2], LB, horizon)
    x_test, y_test   = generate_sequences(x[split2:], y[split2:], LB, horizon)

    # Si train o val quedan vacíos, podar la trial
    if len(x_train) == 0 or len(x_val) == 0:
        raise optuna.exceptions.TrialPruned()

    if args.model_type == 'transformer':
        # Hiperparámetros específicos Transformer
        d_model    = trial.suggest_categorical('d_model', [64, 128, 256])
        nhead      = trial.suggest_categorical('nhead', [4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        d_ff       = trial.suggest_categorical('d_ff', [256, 512, 1024])
        model = Transformer(
            input_size=7,
            d_model=d_model,
            nhead=nhead,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=DO
        ).to(device)
    else:
        # Aquí model_type es 'lstm'
        model = LSTM(input_size=7, dropout=DO).to(device)

    # Entrenar
    model = train_and_evaluate(
        model, x_train, y_train, x_val, y_val,
        LR, WD, BS, EPOCHS, PATIENCE, CLIP,
        trial.number
    )

    # Evaluar
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test, horizon=horizon)

    # Guardar modelo de la trial
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, f"trial_{trial.number}_model.pt")
    )
    trial.set_user_attr("lookback", LB)
    return compute_metrics(y_test, y_pred, max_val, min_val)

# =========================
# Ejecutar estudio Optuna
# =========================
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STUDY_DB,
    direction='maximize',
    load_if_exists=True
)
try:
    study.optimize(objective, timeout=int(args.horas * 60 * 60))
except KeyboardInterrupt:
    logging.info("Interrupción manual detectada. Guardando mejor modelo...")
    if study.best_trial:
        bt = study.best_trial
        params = bt.params
        LB = params.get('LOOKBACK', None)

        # Si el mejor modelo es DummyModel
        if args.model_type == 'dummy':
            # Reinstanciamos DummyModel con el lookback de la mejor trial
            model = DummyModel().to(device)
            # Guardar sin entrenar (DummyModel no requiere entrenamiento)
            torch.save({
                'model_state_dict': model.state_dict(),
                'params': {'LOOKBACK': LB},
                'max_val': max_val,
                'min_val': min_val,
                'lookback': LB,
                'horizon': horizon
            }, os.path.join(MODEL_DIR, 'best_model.pt'))
            # Escribir métricas basadas en el valor de la mejor trial
            with open(os.path.join(METRICS_DIR, 'best_model_metrics.txt'), 'w') as f:
                f.write(f"Mejor trial: {bt.number}\nParams: {{'LOOKBACK': {LB}}}\nScore: {bt.value:.4f}\n")
            logging.info("[INTERRUPT] Métricas y DummyModel guardado")
        else:
            if args.model_type == 'transformer':
                model = Transformer(
                    input_size=7,
                    d_model=params['d_model'],
                    nhead=params['nhead'],
                    d_ff=params['d_ff'],
                    num_layers=params['num_layers'],
                    dropout=params['DROPOUT_RATE']
                ).to(device)
            else:
                model = LSTM(input_size=7, dropout=params['DROPOUT_RATE']).to(device)
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"trial_{bt.number}_model.pt")))
            torch.save({
                'model_state_dict': model.state_dict(),
                'params': params,
                'max_val': max_val,
                'min_val': min_val,
                'lookback': LB,
                'horizon': horizon
            }, os.path.join(MODEL_DIR, 'best_model.pt'))
            with open(os.path.join(METRICS_DIR, 'best_model_metrics.txt'), 'w') as f:
                f.write(f"Mejor trial: {bt.number}\nParams: {params}\nScore: {bt.value:.4f}\n")
            logging.info("[INTERRUPT] Métricas y mejor modelo guardado")

# ----------------------------
# Guardado final cuando termina normalmente
# ----------------------------
bt = study.best_trial
params = bt.params
LB = params.get('LOOKBACK', None)

if args.model_type == 'dummy':
    # Instanciar DummyModel una vez más para exportar best_model.pt
    model = DummyModel().to(device)
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': {'LOOKBACK': LB},
        'max_val': max_val,
        'min_val': min_val,
        'lookback': LB,
        'horizon': horizon
    }, os.path.join(MODEL_DIR, 'best_model.pt'))
    with open(os.path.join(METRICS_DIR, 'best_model_metrics.txt'), 'w') as f:
        f.write(f"Mejor trial: {bt.number}\nParams: {{'LOOKBACK': {LB}}}\nScore final: {bt.value:.4f}\n")
    logging.info(f"[FINAL] DummyModel guardado con score={bt.value:.4f} y LOOKBACK={LB}")
else:
    # Para LSTM/Transformer, reentrenamos sobre train+val como antes
    if args.model_type == 'transformer':
        model = Transformer(
            input_size=7,
            d_model=params['d_model'],
            nhead=params['nhead'],
            d_ff=params['d_ff'],
            num_layers=params['num_layers'],
            dropout=params['DROPOUT_RATE']
        ).to(device)
    else:
        model = LSTM(input_size=7, dropout=params['DROPOUT_RATE']).to(device)

    # Re-entrenar el mejor modelo sobre train + val
    x_tr_full, y_tr_full = generate_sequences(x[:split1], y[:split1], params['LOOKBACK'], horizon)
    x_val_full, y_val_full = generate_sequences(x[split1:split2], y[split1:split2], params['LOOKBACK'], horizon)

    model = train_and_evaluate(
        model,
        x_tr_full, y_tr_full,
        x_val_full, y_val_full,
        params['LR'], params['WEIGHT_DECAY'], params['BATCH_SIZE'],
        params['MAX_EPOCHS'], params['PATIENCE_ES'], params['CLIP_GRAD_NORM'],
        bt.number
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params,
        'max_val': max_val,
        'min_val': min_val,
        'lookback': LB,
        'horizon': horizon
    }, os.path.join(MODEL_DIR, 'best_model.pt'))
    with open(os.path.join(METRICS_DIR, 'best_model_metrics.txt'), 'w') as f:
        f.write(f"Mejor trial: {bt.number}\nParams: {params}\nScore final: {bt.value:.4f}\n")
    logging.info(f"[FINAL] Mejor modelo guardado con score={study.best_value:.4f} y params={study.best_params}")

print(f"Estudio completado: mejor params {study.best_params}, score={study.best_value:.4f}")
with open(os.path.join(METRICS_DIR, "best_params.txt"), "w") as f:
    f.write(f"Mejor params: {study.best_params}, Score={study.best_value:.4f}\n"
            f"Trials totales: {len(study.trials)}\n"
            f"Mejor trial: {bt.number}\n"
            f"Valor: {bt.value:.4f}\n")
