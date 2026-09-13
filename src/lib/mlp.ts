export interface DenseLayer { weight: number[][]; bias: number[] }
export interface ModelArtifact {
  version: number;
  feature_names: string[];
  activation: "relu";
  production_ranker: "max" | "rrf" | "neural" | "ensemble";
  ensemble_alpha?: number | null;
  residual_feature?: number | null;
  residual_scale?: number | null;
  layers: DenseLayer[];
}

export function validateModel(model: ModelArtifact): void {
  if (model.production_ranker === "ensemble" && (typeof model.ensemble_alpha !== "number" || model.ensemble_alpha < 0 || model.ensemble_alpha > 1)) {
    throw new Error("Ensemble coefficient must be between zero and one");
  }
  if (!model.layers.length || !model.feature_names.length) throw new Error("Model has no layers or features");
  if (model.residual_feature != null && (!Number.isInteger(model.residual_feature) || model.residual_feature < 0 || model.residual_feature >= model.feature_names.length)) throw new Error("Residual feature index is invalid");
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
  return model.residual_feature == null ? values[0] : features[model.residual_feature] + (model.residual_scale ?? 1) * values[0];
}
