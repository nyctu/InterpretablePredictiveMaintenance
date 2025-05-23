# %% [markdown]
# ## 1. Imports and Parameters
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
import kerastuner as kt

# ───────── PARAMETERS ───────────────
W, H = 20, 1           # sliding window length
N_CLASSES = 5
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 50        # for tuning / final training

# cost matrix (unchanged)
type_cost = np.zeros((N_CLASSES, N_CLASSES), int)
type_cost[0,1:] = [7,8,9,10]
type_cost[1,0]  = 200; type_cost[1,2:] = [7,8,9]
type_cost[2,0]  = 300; type_cost[2,1]  = 200; type_cost[2,3:] = [7,8]
type_cost[3,0]  = 400; type_cost[3,1]  = 300; type_cost[3,2]  = 200; type_cost[3,4] = 7
type_cost[4,0]  = 500; type_cost[4,1]  = 400; type_cost[4,2]  = 300; type_cost[4,3] = 200

# ───────── 2. UTILITY FUNCTIONS ───────────────
def build_sliding_windows(df):
    X, y = [], []
    drops = ["vehicle_id","time_step","cluster","class_label"]
    for _, grp in df.sort_values(["vehicle_id","time_step"]).groupby("vehicle_id"):
        feats = grp.drop(columns=drops, errors="ignore").select_dtypes(float).values
        labs  = grp["class_label"].values
        for end in range(W-1, len(feats)):
            X.append(feats[end-W+1:end+1])
            y.append(labs[end])
    return np.stack(X), np.array(y)

def build_last_window_all(df):
    X, y = [], []
    drops = ["vehicle_id","time_step","cluster","class_label"]
    for _, grp in df.sort_values(["vehicle_id","time_step"]).groupby("vehicle_id"):
        feats = grp.drop(columns=drops, errors="ignore").select_dtypes(float).values
        if feats.shape[0] >= W:
            seq = feats[-W:]
        else:
            pad = np.zeros((W-feats.shape[0], feats.shape[1]))
            seq = np.vstack([pad, feats])
        X.append(seq)
        y.append(grp["class_label"].iloc[-1])
    return np.stack(X), np.array(y)

def evaluate_and_record(name, y_true, y_pred, y_proba, results):
    labels = list(range(N_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    f2 = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(
            label_binarize(y_true, classes=labels), 
            y_proba, 
            multi_class="ovr"
        )
    except:
        auc = np.nan
    tot_cost = sum(type_cost[i,j] * cm[i,j] for i in labels for j in labels)
    results.append({
        "Model": name, 
        "Cost": tot_cost, 
        "F2": f2, 
        "AUC": auc
    })
    print(f"{name} | Cost: {tot_cost}, F2: {f2:.3f}, AUC: {auc:.3f}")
    print(pd.DataFrame(
        cm, 
        index=[f"True_{i}" for i in labels], 
        columns=[f"Pred_{j}" for j in labels]
    ))
    print("-"*60)

# ───────── 3. LOAD & PREPARE DATA ───────────────
train_df = pd.read_csv("train_df.csv")
val_df   = pd.read_csv("val_df.csv")
test_df  = pd.read_csv("test_df.csv")

# Build windows
X, y = build_sliding_windows(train_df)
X_tr, X_hold, y_tr, y_hold = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# Standardize
ns, w, nf = X_tr.shape
scaler = StandardScaler().fit(X_tr.reshape(ns, w*nf))
def scale(arr):
    n = len(arr)
    return scaler.transform(arr.reshape(n, w*nf)).reshape(n, w, nf)

X_tr_s   = scale(X_tr)
X_hold_s = scale(X_hold)
Xv_s, yv = build_last_window_all(val_df); Xv_s = scale(Xv_s)
Xt_s, yt = build_last_window_all(test_df); Xt_s = scale(Xt_s)

# ───────── 4. DEFINE HYPERMODEL ───────────────
def build_lstm_model(hp):
    """Builds a compiled LSTM model with hyperparameters from hp."""
    inp = Input(shape=(W, nf))
    
    # number of LSTM layers: 1 or 2
    x = inp
    for i in range(hp.Int("n_lstm_layers", 1, 2)):
        units = hp.Int(f"lstm_units_{i}", 32, 128, step=32)
        return_seq = (i < hp.get("n_lstm_layers") - 1)
        x = layers.LSTM(
            units, 
            return_sequences=return_seq,
            dropout=hp.Float(f"lstm_dropout_{i}", 0.0, 0.3, step=0.1)
        )(x)
    
    # optional dense layer before output
    if hp.Boolean("use_dense"):
        x = layers.Dense(
            hp.Int("dense_units", 32, 128, step=32), 
            activation="relu"
        )(x)
        x = layers.Dropout(hp.Float("dense_dropout", 0.0, 0.5, step=0.1))(x)
    
    out = layers.Dense(N_CLASSES, activation="softmax")(x)
    model = Model(inp, out)
    
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[]
    )
    return model

# ───────── 5. RUN KERAS‑TUNER ───────────────
tuner = kt.Hyperband(
    build_lstm_model,
    objective="val_loss",
    max_epochs=MAX_EPOCHS,
    factor=3,
    directory="lstm_tuning",
    project_name="lstm_cost_sensitive"
)

# Callbacks for tuner search
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
]

tuner.search(
    X_tr_s, y_tr,
    validation_data=(X_hold_s, y_hold),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Retrieve best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
for key, val in best_hp.values.items():
    print(f" - {key}: {val}")

# ───────── 6. TRAIN FINAL MODEL ───────────────
# (Optionally combine train+holdout for final training)
X_final_train = np.vstack([X_tr_s, X_hold_s])
y_final_train = np.concatenate([y_tr, y_hold])

best_model = build_lstm_model(best_hp)

final_callbacks = [
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
]

history = best_model.fit(
    X_final_train, y_final_train,
    validation_split=0.1,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=final_callbacks,
    verbose=1
)

# ───────── 7. EVALUATE ON ALL SETS ───────────────
results = []

for data, lbl, tag in [
    (X_hold_s, y_hold, 'Holdout'),
    (Xv_s,    yv,    'Validation-LastWindow'),
    (Xt_s,    yt,    'Test-LastWindow')
]:
    proba = best_model.predict(data, verbose=0)
    preds = np.argmax(proba, axis=1)
    evaluate_and_record(f"LSTM-Tuned {tag}", lbl, preds, proba, results)

# ───────── 8. SUMMARY ───────────────
df_res = pd.DataFrame(results)
print("\nFinal Comparison:")
print(df_res)

# Optionally save to CSV
df_res.to_csv("lstm_tuned_comparison.csv", index=False)



best_model.save("best_lstm_tuned_model.h5")

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = tuner.hypermodel.build(best_hp)

best_model.fit(
    X_tr_s, y_tr,
    validation_data=(X_hold_s, y_hold),
    epochs=50,  # or whatever you'd like
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
)

best_model.save("best_lstm_tuned_model.h5")
