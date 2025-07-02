import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

def clean_data(df, device):
    """
    Filtra por dispositivo, resampleo horario, elimina duplicados, 
    outliers, saltos >0.1 y forward-fill.
    Devuelve un DataFrame con columnas 'ts', 'soil_moisture', etc.
    """
    df = df.copy()
    df['device'] = df['device'].astype(str)
    df = df[df['device'] == str(device)]
    
    df['ts'] = pd.to_datetime(df['ts'])
    df = (df.drop_duplicates(subset='ts')
            .set_index('ts')
            .resample('h').asfreq()
            .ffill()
            .reset_index())
    
    df['Index'] = df.index
    df = df.rename(columns={'sensor1': 'soil_moisture'})
    df['soil_moisture'] = (4095 - df['soil_moisture']) / 4095
    df = df.drop(columns=['sensor2','var_s1','var_s2','voltaje','device'])
    '''
        df = df.rename(columns={
        'precipitacion': 'p',
        'temperatura': 'ta',
        'humedad_ambiente': 'RH',
        'viento': 'WS',
        'radiacion_solar': 'SW'
    })
    '''

    # Outliers IQR
    Q1 = df['soil_moisture'].quantile(0.25)
    Q3 = df['soil_moisture'].quantile(0.75)
    IQR = Q3 - Q1
    lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df = df[(df['soil_moisture'] >= lb) & (df['soil_moisture'] <= ub)]

    # Saltos bruscos > 0.1
    df = df.sort_values('ts').set_index('ts')
    jumps = df['soil_moisture'].diff(-1).abs() > 0.1
    df = df[~jumps]

    # Reconstruir índice horario y ffill
    df = df.resample('h').asfreq().ffill().reset_index()
    return df

def segment_data_by_time_gaps(device_data, timestamp_col='ts',
                              max_gap=timedelta(days=7),
                              min_points=200,
                              min_freq_per_day=None):
    """
    Divide device_data en intervalos continuos separados por gaps > max_gap
    y descarta intervalos con < min_points (o freq < min_freq_per_day).
    """
    df = device_data.sort_values(timestamp_col).reset_index(drop=True)
    intervals, cut_points = [], []
    if df.empty:
        return intervals, cut_points
    
    start = 0
    cut_points.append(df.loc[0, timestamp_col])
    for i in range(1, len(df)):
        if (df.loc[i, timestamp_col] - df.loc[i-1, timestamp_col]) > max_gap:
            intervals.append(df.iloc[start:i])
            cut_points.append(df.loc[i, timestamp_col])
            start = i
    intervals.append(df.iloc[start:])
    
    valid = []
    for seg in intervals:
        if len(seg) < min_points:
            continue
        if min_freq_per_day is not None:
            days = max(1, (seg[timestamp_col].iloc[-1] - seg[timestamp_col].iloc[0]).days)
            if len(seg)/days < min_freq_per_day:
                continue
        valid.append(seg)
    return valid, cut_points

def compute_custom_score(interval_df,
                         weight_duration=1,
                         weight_count=1,
                         weight_corr=1,
                         weight_outlier=1,
                         weight_low=1,
                         low_threshold=0.02):
    """
    Score = base_score * penalty_factor
    base_score = duration^w1 * count^w2 * corr^w3 * outlier_factor^w4
    penalty_factor = max(0, 1 - weight_low * low_frac)
    """
    start, end = interval_df["ts"].iloc[0], interval_df["ts"].iloc[-1]
    duration = (end - start).total_seconds()
    count = len(interval_df)

    corr = interval_df["sensor1"].corr(interval_df["sensor2"])
    corr = abs(corr) if not pd.isnull(corr) else 0

    def frac_out(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
        return len(s[(s<lb)|(s>ub)]) / len(s)

    outlier_factor = max(0, 1 - (frac_out(interval_df["sensor1"]) + frac_out(interval_df["sensor2"]))/2)
    base_score = (duration**weight_duration) * \
                 (count**weight_count) * \
                 (corr**weight_corr) * \
                 (outlier_factor**weight_outlier)

    sm_norm = (4095 - interval_df["sensor1"]) / 4095
    low_frac = (sm_norm < low_threshold).mean()
    penalty = max(0, 1 - weight_low * low_frac)
    return base_score * penalty

def main():
    # Parámetros
    max_gap = timedelta(days=3)
    min_points = 200
    min_freq_per_day = None
    top_N = 5
    score_weights = dict(
        weight_duration=1,
        weight_count=1,
        weight_corr=2,
        weight_outlier=1,
        weight_low=5,
        low_threshold=0.02
    )

    df = pd.read_csv("data/merged.csv")
    df["ts"] = pd.to_datetime(df["ts"])
    out_dir = "data/intervals"
    os.makedirs(out_dir, exist_ok=True)

    all_info = []
    by_dev = {}
    for dev in df["device"].unique():
        dev_data = df[df["device"] == dev]
        segs, _ = segment_data_by_time_gaps(dev_data, "ts",
                                            max_gap, min_points, min_freq_per_day)
        by_dev[dev] = []
        for seg in segs:
            sc = compute_custom_score(seg, **score_weights)
            info = {
                "device": dev,
                "start": seg["ts"].iloc[0],
                "end":   seg["ts"].iloc[-1],
                "count": len(seg),
                "score": sc,
                "interval_df": seg,
                "outstanding": False
            }
            by_dev[dev].append(info)
            all_info.append(info)

    top = sorted(all_info, key=lambda x: x["score"], reverse=True)[:top_N]
    for t in top:
        t["outstanding"] = True

    print("Selected highlighted intervals:")
    for info in top:
        dev = info["device"]
        s, e = info["start"], info["end"]
        days = (e - s).days + 1
        pts = info["count"]
        print(f"Device: {dev} | Range: {s.date()} → {e.date()} | Days: {days} | Points: {pts}")


    for info in top:
        dev = info["device"]
        seg = info["interval_df"]
        cleaned = clean_data(seg.copy(), dev)

        s, e = info["start"], info["end"]
        fname = f"device_{dev}_{s:%Y%m%d_%H%M%S}_to_{e:%Y%m%d_%H%M%S}"
        
        # Export CSV
        cleaned.to_csv(os.path.join(out_dir, fname + ".csv"), index=False)

        # 1) Plot of the cleaned interval
        fig, ax = plt.subplots(figsize=(15,5))
        ax.scatter(cleaned["ts"], cleaned["soil_moisture"],
                   s=3, color="tab:green", label="Processed SM")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.set_title(f"{dev} cleaned: {s.date()} → {e.date()}")
        ax.set_xlabel("Time (ts)")
        ax.set_ylabel("Normalized SM")
        ax.legend()
        ax.tick_params(axis="x", labelrotation=45)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, fname + "_cleaned.png"), bbox_inches='tight')
        plt.close(fig)

        # 2) Plot of the full device with the interval highlighted
        raw = df[df["device"] == dev].copy()
        fig, ax = plt.subplots(figsize=(15,5))
        ax.scatter(raw["ts"], raw["sensor1"], s=3, label="Sensor 1", color="tab:blue")
        ax.scatter(raw["ts"], raw["sensor2"], s=3, label="Sensor 2", color="tab:orange")
        # Highlight the selected interval with a green background
        ax.axvspan(s, e, facecolor="lightgreen", alpha=0.3, zorder=1)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.set_title(f"{dev} full: interval {s.date()} → {e.date()}")
        ax.set_xlabel("Time (ts)")
        ax.set_ylabel("Sensor Value")
        ax.legend()
        ax.tick_params(axis="x", labelrotation=45)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, fname + "_full.png"), bbox_inches='tight')
        plt.close(fig)

    print("Exported CSVs and both plots for each highlighted interval.")

if __name__ == "__main__":
    main()
