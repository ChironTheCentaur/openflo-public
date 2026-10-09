import sys

import flowio

if len(sys.argv) < 2:
    sys.exit("usage: python -m openflo.inspect_fcs <path-to.fcs>")
fcs_path = sys.argv[1]

f = flowio.FlowData(fcs_path)
print("Channels:")
print("  channel_count:", f.channel_count)
for key, ch in f.channels.items():
    print(f"  key={key!r} (type={type(key).__name__}): {ch}")

print()
# Dump all text keys that look channel-related
print("Text keys (PnN/PnS/P*N):")
for k, v in sorted(f.text.items()):
    if str(k).upper().startswith('$P') or 'name' in str(k).lower():
        print(f"  {k!r}: {v!r}")

print()
spill_keys = [k for k in f.text if 'spill' in str(k).lower()]
print("Spillover keys:", spill_keys)
if spill_keys:
    print("Value:", f.text[spill_keys[0]][:300])
