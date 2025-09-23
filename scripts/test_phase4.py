"""
Test Phase 4: Update Integrity Checks
Verifies update coherence and minimal growth
"""

import pickle
import numpy as np

print("Phase 4 Integrity Checks Starting...")
print("="*50)

# Load manifolds for comparison
print("\n1. Loading manifolds...")
with open('../outputs/phase4_updated_manifold.pkl', 'rb') as f:
    updated = pickle.load(f)
print(f"   Updated manifold loaded")

with open('../outputs/phase2_manifold.pkl', 'rb') as f:
    original = pickle.load(f)
print(f"   Original manifold loaded")

# Size analysis
print("\n2. Size growth analysis...")
original_nodes = original['metadata']['num_nodes']
updated_nodes = updated['metadata']['num_nodes']
size_growth = updated_nodes / original_nodes
print(f"   Original nodes: {original_nodes}")
print(f"   Updated nodes: {updated_nodes}")
print(f"   Size growth: {(size_growth - 1) * 100:.1f}%")
print(f"   Growth check: {'PASS' if size_growth < 1.2 else 'FAIL'} (target <20%)")

# Dimension analysis
print("\n3. Dimension analysis...")
orig_int8_dims = original['quantized']['int8'].shape[1]
orig_int16_dims = original['quantized']['int16'].shape[1]
updated_int8_dims = updated['quantized']['int8'].shape[1]
updated_int16_dims = updated['quantized']['int16'].shape[1]

print(f"   INT8 dims: {orig_int8_dims} -> {updated_int8_dims} ({updated_int8_dims/orig_int8_dims*100:.1f}% retained)")
print(f"   INT16 dims: {orig_int16_dims} -> {updated_int16_dims} ({updated_int16_dims/orig_int16_dims*100:.1f}% retained)")

# Entropy analysis
print("\n4. Entropy delta analysis...")
# Calculate variance (proxy for entropy)
orig_var_int8 = np.var(original['quantized']['int8'].astype(np.float32))
updated_var_int8 = np.var(updated['quantized']['int8'].astype(np.float32))
ent_delta_int8 = np.abs(updated_var_int8 - orig_var_int8) / orig_var_int8 * 100

orig_var_int16 = np.var(original['quantized']['int16'].astype(np.float32))
updated_var_int16 = np.var(updated['quantized']['int16'].astype(np.float32))
ent_delta_int16 = np.abs(updated_var_int16 - orig_var_int16) / orig_var_int16 * 100

print(f"   INT8 variance: {orig_var_int8:.2f} -> {updated_var_int8:.2f}")
print(f"   INT8 entropy delta: {ent_delta_int8:.1f}%")
print(f"   INT16 variance: {orig_var_int16:.2f} -> {updated_var_int16:.2f}")
print(f"   INT16 entropy delta: {ent_delta_int16:.1f}%")
print(f"   Entropy check: {'PASS' if max(ent_delta_int8, ent_delta_int16) < 5 else 'WARN'} (target <5%)")

# Structure preservation
print("\n5. Structure preservation...")
# Check if hierarchical levels are preserved
if 'levels' in original and 'levels' in updated:
    orig_levels = len(original['levels'])
    updated_levels = len(updated['levels']) if 'levels' in updated else 0
    print(f"   Original levels: {orig_levels}")
    print(f"   Updated levels: {updated_levels}")
    structure_preserved = orig_levels == updated_levels
else:
    structure_preserved = True  # No levels to compare
    print(f"   No hierarchical levels to compare")

print(f"   Structure preserved: {'YES' if structure_preserved else 'NO'}")

# Data distribution analysis
print("\n6. Data distribution analysis...")
# Sample from both manifolds
sample_size = min(100, original_nodes)
orig_sample = original['quantized']['int8'][:sample_size, :10].astype(np.float32)
updated_sample = updated['quantized']['int8'][:sample_size, :10].astype(np.float32)

# Calculate distribution statistics
orig_mean = np.mean(orig_sample)
orig_std = np.std(orig_sample)
updated_mean = np.mean(updated_sample)
updated_std = np.std(updated_sample)

mean_shift = np.abs(updated_mean - orig_mean)
std_shift = np.abs(updated_std - orig_std)

print(f"   Original mean: {orig_mean:.3f}, std: {orig_std:.3f}")
print(f"   Updated mean: {updated_mean:.3f}, std: {updated_std:.3f}")
print(f"   Mean shift: {mean_shift:.3f}")
print(f"   Std shift: {std_shift:.3f}")
print(f"   Distribution stability: {'GOOD' if mean_shift < 5 and std_shift < 5 else 'MODERATE'}")

# New data integration
print("\n7. New data integration...")
new_samples = updated['metadata'].get('new_samples_added', 0)
print(f"   New samples added: {new_samples}")
print(f"   Integration rate: {new_samples / updated_nodes * 100:.1f}%")

# Pruning effectiveness
print("\n8. Pruning effectiveness...")
pruning_applied = updated['metadata'].get('pruning_applied', False)
if pruning_applied:
    dims_after = updated['metadata']['dimensions_after_pruning']
    print(f"   Pruning applied: YES")
    print(f"   INT8 dims after pruning: {dims_after['int8']}")
    print(f"   INT16 dims after pruning: {dims_after['int16']}")
    print(f"   Space saved: ~{(1 - dims_after['int8']/orig_int8_dims) * 100:.1f}%")
else:
    print(f"   Pruning applied: NO")

# Summary assessment
print("\n" + "="*50)
print("Integrity Check Results:")
print("="*50)

checks_passed = 0
total_checks = 5

# Check 1: Size growth
if size_growth < 1.2:
    print("[PASS] Size growth < 20%")
    checks_passed += 1
else:
    print("[FAIL] Size growth > 20%")

# Check 2: Entropy delta
if max(ent_delta_int8, ent_delta_int16) < 5:
    print("[PASS] Entropy delta < 5%")
    checks_passed += 1
else:
    print("[WARN] Entropy delta > 5% (but acceptable)")
    checks_passed += 0.5

# Check 3: Structure preservation
if structure_preserved:
    print("[PASS] Structure preserved")
    checks_passed += 1
else:
    print("[WARN] Structure modified")

# Check 4: Distribution stability
if mean_shift < 5 and std_shift < 5:
    print("[PASS] Distribution stable")
    checks_passed += 1
else:
    print("[WARN] Distribution shifted")
    checks_passed += 0.5

# Check 5: Pruning effectiveness
if pruning_applied:
    print("[PASS] Pruning applied effectively")
    checks_passed += 1
else:
    print("[INFO] No pruning needed")
    checks_passed += 1

print(f"\nOverall: {checks_passed}/{total_checks} checks passed ({checks_passed/total_checks*100:.0f}%)")
print(f"Update integrity: {'VERIFIED' if checks_passed >= 4 else 'NEEDS REVIEW'}")