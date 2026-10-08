"""Rebuild index using the SAME matching logic as build.ipynb,
but only include rows where the signal file actually exists."""
import os
import pandas as pd

def exists(p):
    return os.path.exists(p)

# ===== Circle =====
h_imgs = sorted(os.listdir('healthy/Circle'))
p_imgs = sorted(os.listdir('patient/Circle'))

rows = []
# healthy: circA-P{n}.jpg -> circA-H{n}.csv
for f in h_imgs:
    stem = f[:-4]  # circA-P1
    sig = f'healthy/HealthySignal/{stem[:6]}H{stem[7:]}.csv'
    img = f'healthy/Circle/{f}'
    if exists(img) and exists(sig):
        rows.append([img, sig, 0])
# patient: circA-P{n}.jpg -> circA-P{n}.csv
for f in p_imgs:
    stem = f[:-4]
    sig = f'patient/PatientSignal/{stem}.csv'
    img = f'patient/Circle/{f}'
    if exists(img) and exists(sig):
        rows.append([img, sig, 1])
pd.DataFrame(rows).to_csv('circle.csv', index=False, header=False)
print(f'circle: total={len(rows)}, h={sum(1 for r in rows if r[2]==0)}, p={sum(1 for r in rows if r[2]==1)}')

# ===== Meander =====
h_imgs = sorted(os.listdir('healthy/HealthyMeander'))
p_imgs = sorted(os.listdir('patient/PatientMeander'))
rows = []
# healthy: mea1-H{n}.jpg -> sigMea1-H{n}.csv  (i.e. 'sigM' + f[1:])
for f in h_imgs:
    stem = f[:-4]  # mea1-H1
    sig = f'healthy/HealthySignal/sigM{stem[1:]}.csv'
    img = f'healthy/HealthyMeander/{f}'
    if exists(img) and exists(sig):
        rows.append([img, sig, 0])
# patient: mea1-P{n}.jpg -> sigMea1-P{n}.csv
for f in p_imgs:
    stem = f[:-4]
    sig = f'patient/PatientSignal/sigM{stem[1:]}.csv'
    img = f'patient/PatientMeander/{f}'
    if exists(img) and exists(sig):
        rows.append([img, sig, 1])
pd.DataFrame(rows).to_csv('meander.csv', index=False, header=False)
print(f'meander: total={len(rows)}, h={sum(1 for r in rows if r[2]==0)}, p={sum(1 for r in rows if r[2]==1)}')

# ===== Spiral =====
h_imgs = sorted(os.listdir('healthy/HealthySpiral'))
p_imgs = sorted(os.listdir('patient/PatientSpiral'))
rows = []
for f in h_imgs:
    stem = f[:-4]  # sp1-H1
    sig = f'healthy/HealthySignal/sigS{stem[1:]}.csv'
    img = f'healthy/HealthySpiral/{f}'
    if exists(img) and exists(sig):
        rows.append([img, sig, 0])
for f in p_imgs:
    stem = f[:-4]
    sig = f'patient/PatientSignal/sigS{stem[1:]}.csv'
    img = f'patient/PatientSpiral/{f}'
    if exists(img) and exists(sig):
        rows.append([img, sig, 1])
pd.DataFrame(rows).to_csv('spiral.csv', index=False, header=False)
print(f'spiral: total={len(rows)}, h={sum(1 for r in rows if r[2]==0)}, p={sum(1 for r in rows if r[2]==1)}')
