import { readFileSync } from "node:fs";
import { forward, type ModelArtifact } from "../src/lib/mlp";

const payload = JSON.parse(readFileSync(process.argv[2], "utf8")) as { model: ModelArtifact; vectors: number[][] };
process.stdout.write(JSON.stringify(payload.vectors.map((vector) => forward(vector, payload.model))));
