export interface DenseLayer { weight: number[][]; bias: number[] }
export interface ModelArtifact {
  version: number;
  feature_names: string[];
  activation: "relu";
  production_ranker: "max" | "rrf" | "neural" | "ensemble";
  ensemble_alpha?: number | null;
  layers: DenseLayer[];
}

export function validateModel(model: ModelArtifact): void {
  if (!model.layers.length || !model.feature_names.length) throw new Error("Model has no layers or features");
  let width = model.feature_names.length;
  for (const layer of model.layers) {
    if (layer.weight.length !== layer.bias.length || layer.weight.some((row) => row.length !== width)) {
      throw new Error("Model layer width mismatch");
    }
    width = layer.bias.length;
  }
  if (width !== 1) throw new Error("Model output width must be one");
}

export function forward(features: readonly number[], model: ModelArtifact): number {
  validateModel(model);
  if (features.length !== model.feature_names.length) throw new Error("Feature width mismatch");
  let values = [...features];
  model.layers.forEach((layer, layerIndex) => {
    values = layer.weight.map((row, rowIndex) =>
      row.reduce((sum, weight, column) => sum + weight * values[column], layer.bias[rowIndex]),
    );
    if (layerIndex < model.layers.length - 1) values = values.map((value) => Math.max(0, value));
  });
  return values[0];
}
