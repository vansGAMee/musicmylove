/**
 * src/lib/tasteliftnet/adapter.ts
 * Client-Side Personal Preference Adapter with Exact Analytical Gradient Descent.
 * Bounded learning rate, mathematical gradient updates, local browser persistence.
 */

export interface AdapterState {
  weights: number[]; // theta_user in R^64
  learningRate: number; // eta, e.g. 0.1
  updateCount: number;
}

export function createAdapter(dim: number = 64, learningRate: number = 0.1): AdapterState {
  return {
    weights: new Array(dim).fill(0),
    learningRate,
    updateCount: 0,
  };
}

export function sigmoid(x: number): number {
  if (x < -15) return 0;
  if (x > 15) return 1;
  return 1 / (1 + Math.exp(-x));
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * (b[i] ?? 0);
  return s;
}

/**
 * Computes the personalized relevance score shift for a candidate.
 * s_personalized = s_base + c^T * theta_user
 */
export function scoreWithAdapter(
  baseScore: number,
  candidateEmbedding: ArrayLike<number>,
  adapter: AdapterState
): number {
  const userDot = dot(candidateEmbedding, adapter.weights);
  return baseScore + userDot;
}

/**
 * Performs one exact gradient descent update step on theta_user given user feedback:
 * Loss L = - y * log(sigma(s)) - (1 - y) * log(1 - sigma(s))
 * grad_theta L = (sigma(s) - y) * c
 * theta_new = theta_old - eta * grad_theta L
 */
export function applyFeedbackGradient(
  adapter: AdapterState,
  candidateEmbedding: ArrayLike<number>,
  feedback: 'like' | 'dislike',
  baseScore: number = 0
): {
  loss: number;
  gradient: number[];
  weightsBefore: number[];
  weightsAfter: number[];
  scoreBefore: number;
  scoreAfter: number;
} {
  const y = feedback === 'like' ? 1.0 : 0.0;
  const currentScore = scoreWithAdapter(baseScore, candidateEmbedding, adapter);
  const p = sigmoid(currentScore);

  // Exact binary cross entropy loss
  const eps = 1e-12;
  const loss = -(y * Math.log(Math.max(eps, p)) + (1.0 - y) * Math.log(Math.max(eps, 1.0 - p)));

  // Analytical gradient: (p - y) * c
  const gradScalar = p - y;
  const gradient = Array.from(candidateEmbedding).map(c => gradScalar * c);

  const weightsBefore = [...adapter.weights];

  // Gradient descent update step: theta <- theta - eta * grad
  const weightsAfter = adapter.weights.map((w, i) => w - adapter.learningRate * gradient[i]);
  adapter.weights = weightsAfter;
  adapter.updateCount += 1;

  const scoreAfter = scoreWithAdapter(baseScore, candidateEmbedding, adapter);

  return {
    loss,
    gradient,
    weightsBefore,
    weightsAfter,
    scoreBefore: currentScore,
    scoreAfter,
  };
}
