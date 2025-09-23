"""
Evaluate Phase 4: Continual Learning Metrics
Measures forgetting resistance and performance on old vs new tasks
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import numpy as np
import time
import pickle

print("Phase 4 Continual Learning Evaluation Starting...")
print("="*50)

# Load both models for comparison
print("\n1. Loading models...")
# Phase 3 original model
orig_model = AutoModelForCausalLM.from_pretrained('../outputs/phase3_model_weights')
orig_tokenizer = AutoTokenizer.from_pretrained('../outputs/phase3_model_weights')
if orig_tokenizer.pad_token is None:
    orig_tokenizer.pad_token = orig_tokenizer.eos_token
print(f"   Original model loaded (Phase 3)")

# Phase 4 updated model
updated_model = AutoModelForCausalLM.from_pretrained('../outputs/phase4_updated_model_weights')
updated_tokenizer = AutoTokenizer.from_pretrained('../outputs/phase4_updated_model_weights')
if updated_tokenizer.pad_token is None:
    updated_tokenizer.pad_token = updated_tokenizer.eos_token
print(f"   Updated model loaded (Phase 4)")

# Create pipelines
orig_generator = pipeline('text-generation', model=orig_model, tokenizer=orig_tokenizer, device=-1)
updated_generator = pipeline('text-generation', model=updated_model, tokenizer=updated_tokenizer, device=-1)

# Test on old-style prompts (should maintain performance)
print("\n2. Testing on old prompts (forgetting test)...")
old_prompts = [
    "In a cluster of 50 concepts:",
    "At hierarchical level 2 with",
    "The quantum manifold contains"
]

old_performance = []
for prompt in old_prompts:
    print(f"\n   Prompt: '{prompt}'")
    
    # Original model output
    start = time.time()
    orig_output = orig_generator(prompt, max_length=30, num_return_sequences=1, 
                                 temperature=0.8, do_sample=True, 
                                 pad_token_id=orig_tokenizer.eos_token_id)
    orig_time = time.time() - start
    orig_text = orig_output[0]['generated_text']
    
    # Updated model output
    start = time.time()
    updated_output = updated_generator(prompt, max_length=30, num_return_sequences=1,
                                      temperature=0.8, do_sample=True,
                                      pad_token_id=updated_tokenizer.eos_token_id)
    updated_time = time.time() - start
    updated_text = updated_output[0]['generated_text']
    
    print(f"   Original: {orig_text[:60]}...")
    print(f"   Updated:  {updated_text[:60]}...")
    
    # Calculate similarity (simple word overlap)
    orig_words = set(orig_text.lower().split())
    updated_words = set(updated_text.lower().split())
    overlap = len(orig_words.intersection(updated_words)) / len(orig_words.union(updated_words))
    
    old_performance.append({
        'prompt': prompt,
        'similarity': overlap,
        'orig_time': orig_time,
        'updated_time': updated_time
    })

# Test on new-style prompts (should show improvement)
print("\n3. Testing on new prompts (adaptation test)...")
new_prompts = [
    "Dynamic chipping maintains",
    "Diffraction patterns merge",
    "Continual learning through manifold",
    "Quantum entanglement in neural"
]

new_performance = []
for prompt in new_prompts:
    print(f"\n   Prompt: '{prompt}'")
    
    # Original model output
    start = time.time()
    orig_output = orig_generator(prompt, max_length=30, num_return_sequences=1,
                                temperature=0.8, do_sample=True,
                                pad_token_id=orig_tokenizer.eos_token_id)
    orig_time = time.time() - start
    orig_text = orig_output[0]['generated_text']
    
    # Updated model output
    start = time.time()
    updated_output = updated_generator(prompt, max_length=30, num_return_sequences=1,
                                      temperature=0.8, do_sample=True,
                                      pad_token_id=updated_tokenizer.eos_token_id)
    updated_time = time.time() - start
    updated_text = updated_output[0]['generated_text']
    
    print(f"   Original: {orig_text[:60]}...")
    print(f"   Updated:  {updated_text[:60]}...")
    
    # Check if new concepts appear
    new_concepts = ['quantum', 'manifold', 'holographic', 'diffraction', 'chipping', 'continual']
    orig_has_concepts = sum(1 for c in new_concepts if c in orig_text.lower())
    updated_has_concepts = sum(1 for c in new_concepts if c in updated_text.lower())
    
    new_performance.append({
        'prompt': prompt,
        'orig_concepts': orig_has_concepts,
        'updated_concepts': updated_has_concepts,
        'improvement': updated_has_concepts > orig_has_concepts
    })

# Calculate forgetting metrics
print("\n4. Calculating forgetting metrics...")
avg_similarity = np.mean([p['similarity'] for p in old_performance])
forgetting_rate = (1 - avg_similarity) * 100

print(f"   Average similarity on old prompts: {avg_similarity:.3f}")
print(f"   Forgetting rate: {forgetting_rate:.1f}%")
print(f"   Forgetting assessment: {'LOW' if forgetting_rate < 10 else 'MODERATE' if forgetting_rate < 30 else 'HIGH'}")

# Calculate adaptation metrics
print("\n5. Calculating adaptation metrics...")
improvements = sum(1 for p in new_performance if p['improvement'])
adaptation_rate = improvements / len(new_performance) * 100

avg_new_concepts_orig = np.mean([p['orig_concepts'] for p in new_performance])
avg_new_concepts_updated = np.mean([p['updated_concepts'] for p in new_performance])

print(f"   Prompts with improvement: {improvements}/{len(new_performance)}")
print(f"   Adaptation rate: {adaptation_rate:.0f}%")
print(f"   Avg new concepts (original): {avg_new_concepts_orig:.1f}")
print(f"   Avg new concepts (updated): {avg_new_concepts_updated:.1f}")

# Performance comparison
print("\n6. Performance comparison...")
orig_avg_time = np.mean([p['orig_time'] for p in old_performance])
updated_avg_time = np.mean([p['updated_time'] for p in old_performance])
speed_change = (updated_avg_time - orig_avg_time) / orig_avg_time * 100

print(f"   Original model avg time: {orig_avg_time:.2f}s")
print(f"   Updated model avg time: {updated_avg_time:.2f}s")
print(f"   Speed change: {speed_change:+.1f}%")

# Coherence test
print("\n7. Coherence test...")
test_prompt = "Holographic knowledge manifold with quantum"
updated_output = updated_generator(test_prompt, max_length=50, num_return_sequences=1,
                                  temperature=0.7, do_sample=True,
                                  pad_token_id=updated_tokenizer.eos_token_id)
coherence_text = updated_output[0]['generated_text']
print(f"   Test prompt: '{test_prompt}'")
print(f"   Generated: {coherence_text}")

# Simple coherence check - are sentences complete?
sentences = coherence_text.split('.')
complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
coherence_score = complete_sentences / max(1, len(sentences))
print(f"   Coherence score: {coherence_score:.2f}")

# Overall assessment
print("\n" + "="*50)
print("Continual Learning Evaluation Results:")
print("="*50)

criteria_passed = 0
total_criteria = 5

# Criterion 1: Low forgetting
if forgetting_rate < 10:
    print("[PASS] Forgetting < 10%")
    criteria_passed += 1
elif forgetting_rate < 30:
    print("[WARN] Forgetting 10-30% (moderate)")
    criteria_passed += 0.5
else:
    print("[FAIL] Forgetting > 30%")

# Criterion 2: Adaptation to new data
if adaptation_rate >= 50:
    print("[PASS] Adaptation rate >= 50%")
    criteria_passed += 1
else:
    print("[FAIL] Adaptation rate < 50%")

# Criterion 3: Performance maintained
if abs(speed_change) < 20:
    print("[PASS] Performance change < 20%")
    criteria_passed += 1
else:
    print("[WARN] Performance change > 20%")
    criteria_passed += 0.5

# Criterion 4: Coherence maintained
if coherence_score > 0.5:
    print("[PASS] Coherence maintained")
    criteria_passed += 1
else:
    print("[FAIL] Coherence degraded")

# Criterion 5: New concepts integrated
if avg_new_concepts_updated > avg_new_concepts_orig:
    print("[PASS] New concepts integrated")
    criteria_passed += 1
else:
    print("[FAIL] New concepts not integrated")

print(f"\nOverall: {criteria_passed}/{total_criteria} criteria passed ({criteria_passed/total_criteria*100:.0f}%)")
print(f"Continual learning: {'SUCCESSFUL' if criteria_passed >= 3 else 'NEEDS IMPROVEMENT'}")

# Save results
eval_results = {
    'forgetting_rate': forgetting_rate,
    'adaptation_rate': adaptation_rate,
    'speed_change': speed_change,
    'coherence_score': coherence_score,
    'criteria_passed': criteria_passed,
    'old_performance': old_performance,
    'new_performance': new_performance
}

with open('../outputs/phase4_evaluation.pkl', 'wb') as f:
    pickle.dump(eval_results, f)
print(f"\nEvaluation results saved to: phase4_evaluation.pkl")