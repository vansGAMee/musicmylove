import { expect, test } from "@playwright/test";

test("five real-shaped selections automatically run TasteLift and show forty recommendations", async ({ page }) => {
  await page.route("**/api/search?q=*", async (route) => {
    const title = new URL(route.request().url()).searchParams.get("q")!;
    await route.fulfill({ json: [{ mbid: `seed-${title}`, title, artist: `Artist ${title}` }] });
  });
  await page.route("**/api/tastelift", async (route) => route.fulfill({ json: {
    candidateCount: 500,
    seeds: Array.from({ length: 5 }, (_, i) => ({ input: { artist: `Artist ${i}`, title: `Seed ${i}` }, status: "resolved", source: "text" })),
    recommendations: Array.from({ length: 40 }, (_, i) => ({ mbid: `candidate-${i}`, title: `Candidate ${i}`, artist: `Artist ${i}`, score: 100 - i, strongestTasteHead: i % 4, seedSupport: 2, popularityPercentile: 0.2, noveltyLiftScore: 1, spotifyLink: `https://open.spotify.com/search/candidate-${i}` })),
  } }));
  await page.goto("/");
  for (const title of ["one", "two", "three", "four", "five"]) {
    await page.getByRole("combobox").fill(title);
    await page.getByRole("button", { name: new RegExp(title, "i") }).click();
  }
  await expect(page.getByTestId("recommendation")).toHaveCount(40);
  await expect(page.getByText("5 / 5")).toBeVisible();
  await expect(page.getByRole("button", { name: /generate/i })).toHaveCount(0);
});
