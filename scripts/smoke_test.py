"""Quick manual smoke test: load one sample, compensate, transform, gate and
cluster end-to-end against a REAL FCS dataset.

Not a pytest test (it needs real data). Point it at your dataset and edit the
SAMPLE / FMO basenames below to match your files:
    OPENFLO_TEST_FCS_DIR=/path/to/fcs  python scripts/smoke_test.py
    # or:  python scripts/smoke_test.py /path/to/fcs
Requires the package installed (``pip install -e .``).
"""
import os
import sys

# Required on Windows for PhenoGraph's multiprocessing Pool
if __name__ != '__main__':
    raise SystemExit(0)

from openflo import FlowSample, FMOGater

FCS_DIR = (os.environ.get('OPENFLO_TEST_FCS_DIR')
           or (sys.argv[1] if len(sys.argv) > 1 else ''))
if not FCS_DIR or not os.path.isdir(FCS_DIR):
    raise SystemExit("set OPENFLO_TEST_FCS_DIR (or pass an FCS dir argument) "
                     "to a real dataset")

# Edit these to match a sample + its FMO-control basenames in your dataset.
SAMPLE = 'sample_1'
FMO_BASENAMES = {
    'BV421-A': 'fmo_bv421',
    'APC-A':           'fmo_apc',
    'PE-Cy7-A':        'fmo_cy7',
}

def p(name):
    for f in sorted(os.listdir(FCS_DIR)):
        if not f.lower().endswith('.fcs'):
            continue
        base = f.lower()
        name_l = name.lower()
        if f'_{name_l}_' in base or base.endswith(f'_{name_l}.fcs'):
            return os.path.join(FCS_DIR, f)
    raise FileNotFoundError(name)

print("=" * 60)
print(f"1. Load {SAMPLE}")
s = FlowSample(p(SAMPLE))
print("   columns:", list(s.data.columns))
print("   fluor channels:", s.fluor_channels)
print("   scatter channels:", s.scatter_channels)
print("   data shape:", s.data.shape)

print("\n2. QC")
s.run_qc()
print("   events after QC:", len(s.data))

print("\n3. Compensate")
s.auto_compensate()
print("   fluor range after comp:")
for ch in s.fluor_channels:
    print(f"     {ch}: [{s.data[ch].min():.1f}, {s.data[ch].max():.1f}]")

print("\n4. Transform")
s.apply_transform()
print("   fluor range after logicle:")
for ch in s.fluor_channels:
    print(f"     {ch}: [{s.data[ch].min():.3f}, {s.data[ch].max():.3f}]")

print("\n5. FMO thresholds")
gater = FMOGater()
for channel, basename in FMO_BASENAMES.items():
    gater.add_fmo(channel, p(basename))
gater.prepare()
thresh = gater.compute(percentile=99.5)
print("   thresholds:", thresh)

print("\n6. Apply gates")
s.apply_threshold_gates(thresh)
pos_cols = [c for c in s.data.columns if c.endswith('_pos')]
print("   gate columns added:", pos_cols)
for col in pos_cols:
    print(f"     {col}: {s.data[col].mean()*100:.1f}% positive")

print("\n7. Cluster (k=10 for speed)")
s.cluster(k=10)
print("   clusters:", sorted(s.data['cluster'].unique()))

print("\nSmoke test PASSED")
