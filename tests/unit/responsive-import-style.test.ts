import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "../..");
const css = fs.readFileSync(path.join(root, "app/globals.css"), "utf8");
const component = fs.readFileSync(path.join(root, "src/components/MusicRecommender.tsx"), "utf8");

describe("import screen responsive contract", () => {
  it("does not show internal model implementation copy", () => {
    expect(component).not.toContain("Отдельный режим «Открытия»");
    expect(component).not.toContain("modelCoverage");
  });

  it("keeps the upload disc square and non-shrinking on mobile", () => {
    expect(css).toMatch(/\.card-import \.tactile-disc-dropzone\s*\{[^}]*aspect-ratio:\s*1\s*\/\s*1[^}]*flex:\s*0\s+0\s+auto/s);
    expect(css).toMatch(/\.card-import \.tactile-disc-inner\s*\{[^}]*aspect-ratio:\s*1\s*\/\s*1[^}]*flex:\s*0\s+0\s+auto/s);
  });

  it("prevents long imported names from creating horizontal page scrolling", () => {
    expect(css).toMatch(/html,\s*body\s*\{[^}]*overflow-x:\s*hidden/s);
    expect(css).toMatch(/\.seed-chip\s*\{[^}]*max-width:\s*100%/s);
    expect(css).toMatch(/\.seed-chip span\s*\{[^}]*overflow:\s*hidden[^}]*text-overflow:\s*ellipsis/s);
  });

  it("keeps the intended card content as the only vertical scrolling region", () => {
    expect(css).toMatch(/\.card-scrollable\s*\{[^}]*min-height:\s*0/s);
  });
});
