import { describe, it, expect } from 'vitest';
import { createAdapter, applyFeedbackGradient, scoreWithAdapter, dot } from '../../src/lib/tasteliftnet/adapter';

describe('TasteLiftNet Personal Preference Adapter (Gradient Descent)', () => {
  it('initializes adapter with zero weights', () => {
    const adapter = createAdapter(64, 0.1);
    expect(adapter.weights).toHaveLength(64);
    expect(adapter.weights.every(w => w === 0)).toBe(true);
    expect(adapter.updateCount).toBe(0);
  });

  it('verifies weights before -> gradient -> weights after -> score increase for "like"', () => {
    const adapter = createAdapter(64, 0.1);
    // Create candidate embedding with non-zero vector
    const candidateEmbedding = new Array(64).fill(0);
    candidateEmbedding[0] = 0.8;
    candidateEmbedding[1] = 0.6; // Norm = 1.0

    const baseScore = 0.5;
    const result = applyFeedbackGradient(adapter, candidateEmbedding, 'like', baseScore);

    // 1. Verify weights before was all zeros
    expect(result.weightsBefore.every(w => w === 0)).toBe(true);

    // 2. Verify gradient: y=1, p = sigmoid(0.5) ~ 0.622459, p - y ~ -0.37754
    // grad = (p - y) * c -> negative along candidate direction
    expect(result.gradient[0]).toBeLessThan(0);
    expect(result.gradient[1]).toBeLessThan(0);
    expect(Math.abs(result.gradient[2])).toBe(0);

    // 3. Verify weights after: theta = 0 - eta * grad = 0 - 0.1 * negative = positive!
    expect(result.weightsAfter[0]).toBeGreaterThan(0);
    expect(result.weightsAfter[1]).toBeGreaterThan(0);
    expect(adapter.weights[0]).toBe(result.weightsAfter[0]);

    // 4. Verify score strictly increased
    expect(result.scoreAfter).toBeGreaterThan(result.scoreBefore);
    expect(result.scoreAfter - result.scoreBefore).toBeCloseTo(0.1 * (1.0 - 1 / (1 + Math.exp(-0.5))), 4);
  });

  it('verifies weights before -> gradient -> weights after -> score decrease for "dislike"', () => {
    const adapter = createAdapter(64, 0.1);
    const candidateEmbedding = new Array(64).fill(0);
    candidateEmbedding[0] = 0.707;
    candidateEmbedding[1] = 0.707;

    const baseScore = 0.0;
    const result = applyFeedbackGradient(adapter, candidateEmbedding, 'dislike', baseScore);

    // 1. Verify weights before
    expect(result.weightsBefore.every(w => w === 0)).toBe(true);

    // 2. For dislike (y=0), p = sigmoid(0) = 0.5, p - y = 0.5 > 0
    // grad = 0.5 * c > 0
    expect(result.gradient[0]).toBeGreaterThan(0);
    expect(result.gradient[1]).toBeGreaterThan(0);

    // 3. Weights after: theta = 0 - 0.1 * (0.5 * c) < 0
    expect(result.weightsAfter[0]).toBeLessThan(0);
    expect(result.weightsAfter[1]).toBeLessThan(0);

    // 4. Verify score strictly decreased
    expect(result.scoreAfter).toBeLessThan(result.scoreBefore);
  });

  it('converges across multiple gradient descent steps minimizing loss', () => {
    const adapter = createAdapter(64, 0.2);
    const candidateEmbedding = new Array(64).fill(0);
    candidateEmbedding[0] = 1.0;

    let prevLoss = Infinity;
    for (let step = 0; step < 5; step++) {
      const res = applyFeedbackGradient(adapter, candidateEmbedding, 'like', 0.0);
      expect(res.loss).toBeLessThan(prevLoss);
      prevLoss = res.loss;
    }

    // After 5 likes, candidate score should be significantly higher
    const finalScore = scoreWithAdapter(0.0, candidateEmbedding, adapter);
    expect(finalScore).toBeGreaterThan(0.4);
  });
});
