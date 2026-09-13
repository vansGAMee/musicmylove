import { expect, test } from "vitest";
import { forward, validateModel } from "../../src/lib/mlp";

const artifact = {
  version: 1,
  feature_names: ["a", "b"],
  activation: "relu" as const,
  production_ranker: "neural" as const,
  layers: [
    { weight: [[1, -1], [-1, 1]], bias: [0, 0] },
    { weight: [[2, 3]], bias: [0.5] },
  ],
};

test("runs dense and ReLU layers identically", () => {
  expect(forward([3, 1], artifact)).toBe(4.5);
});

test("rejects a feature width mismatch", () => {
  expect(() => validateModel({ ...artifact, feature_names: ["a"] })).toThrow(/width/i);
});

test("rejects an invalid ensemble coefficient", () => {
  expect(() => validateModel({ ...artifact, production_ranker: "ensemble", ensemble_alpha: 1.2 })).toThrow(/ensemble/i);
});
