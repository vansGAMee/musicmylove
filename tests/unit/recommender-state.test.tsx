// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import MusicRecommender from "../../src/components/MusicRecommender";

afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); localStorage.clear(); });

test("prefetches each selection and automatically returns results after the fifth", async () => {
  vi.useFakeTimers();
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/api/search")) {
      const title = new URL(url, "http://x").searchParams.get("q")!;
      return new Response(JSON.stringify([{ mbid: `seed-${title}`, title, artist: `Artist ${title}` }]));
    }
    if (url.startsWith("/api/similar/")) {
      const seed = url.split("/").pop()!;
      return new Response(JSON.stringify(Array.from({ length: 25 }, (_, index) => ({ mbid: `candidate-${index}`, title: `Candidate ${index}`, artist: `Artist ${index}`, score: 100 - index + seed.length }))));
    }
    return new Response(JSON.stringify({ id: null }));
  });
  vi.stubGlobal("fetch", fetcher);
  render(<MusicRecommender />);
  expect(screen.queryByRole("button", { name: /generate/i })).not.toBeInTheDocument();
  for (const title of ["one", "two", "three", "four", "five"]) {
    fireEvent.change(screen.getByRole("combobox"), { target: { value: title } });
    await act(async () => { await vi.advanceTimersByTimeAsync(350); });
    expect(screen.getByRole("button", { name: new RegExp(title, "i") })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: new RegExp(title, "i") }));
  }
  await act(async () => { await Promise.resolve(); });
  expect(screen.getAllByTestId("recommendation")).toHaveLength(20);
  expect(screen.getByText("5 / 5")).toBeInTheDocument();
  expect(fetcher.mock.calls.filter(([url]) => String(url).startsWith("/api/similar/"))).toHaveLength(5);
});
