"""Convert remaining raw .txt signal files to .csv (6-channel)."""
import os
import pandas as pd
from io import StringIO

COLS = ["Microphone","Fingergrip","Axial_Pressure","X_Acceleration","Y_Acceleration","Z_Acceleration"]

def txt2csv(txt_path, csv_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_lines = [l for l in lines if not l.strip().startswith('#')]
    df = pd.read_csv(StringIO(''.join(data_lines)), delim_whitespace=True, header=None)
    df.columns = COLS
    df.to_csv(csv_path, index=False)

for src, dst in [('patient/Signal', 'patient/PatientSignal'),
                 ('healthy/Signal', 'healthy/HealthySignal')]:
    n = 0
    for f in os.listdir(src):
        if not f.endswith('.txt'):
            continue
        csv_name = f[:-4] + '.csv'
        dst_path = os.path.join(dst, csv_name)
        if not os.path.exists(dst_path):
            txt2csv(os.path.join(src, f), dst_path)
            n += 1
    print(f'{src} -> {dst}: converted {n} new csv')
