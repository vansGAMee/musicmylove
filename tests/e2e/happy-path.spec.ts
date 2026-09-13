import { expect, test } from "@playwright/test";

test("five real-shaped selections automatically show twenty recommendations", async ({ page }) => {
  await page.route("**/api/search?q=*", async (route) => {
    const title = new URL(route.request().url()).searchParams.get("q")!;
    await route.fulfill({ json: [{ mbid: `seed-${title}`, title, artist: `Artist ${title}` }] });
  });
  await page.route("**/api/similar/*", async (route) => route.fulfill({ json: Array.from({ length: 25 }, (_, i) => ({ mbid: `candidate-${i}`, title: `Candidate ${i}`, artist: `Artist ${i}`, score: 100 - i })) }));
  await page.goto("/");
  for (const title of ["one", "two", "three", "four", "five"]) {
    await page.getByRole("combobox").fill(title);
    await page.getByRole("button", { name: new RegExp(title, "i") }).click();
  }
  await expect(page.getByTestId("recommendation")).toHaveCount(20);
  await expect(page.getByText("5 / 5")).toBeVisible();
  await expect(page.getByRole("button", { name: /generate/i })).toHaveCount(0);
});
