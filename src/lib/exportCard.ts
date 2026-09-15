export interface TasteCardTrack {
  title: string;
  artist: string;
  headIndex?: number;
  liftScore?: number;
  popularityPercentile?: number;
}

export interface TasteCardOptions {
  seeds: TasteCardTrack[];
  recommendations: TasteCardTrack[];
  isRussian: boolean;
  theme?: "light" | "dark";
}

/**
 * Draws the aesthetic Joy Division / MusicMyLove Taste Card onto a canvas.
 * Dimensions: 1080 x 1350 px (4:5 vertical poster ratio).
 */
export function renderTasteCardToCanvas(
  canvas: HTMLCanvasElement,
  options: TasteCardOptions
): void {
  const topTracks = options.seeds.slice(0, 5);
  const recTracks = options.recommendations.slice(0, 5);
  const totalTrackRows = Math.max(1, topTracks.length) + Math.max(1, recTracks.length);

  const width = 1080;
  const height = 48 * 2 + 260 + totalTrackRows * 72 + 80;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const isDark = options.theme !== "light";
  const isRu = options.isRussian;

  // Colors
  const bgColor = isDark ? "#0e0e0d" : "#e8e7e1";
  const cardBg = isDark ? "#171715" : "#fbfbfa";
  const cardBorder = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.08)";
  const ink = isDark ? "#f3f2ee" : "#141413";
  const muted = isDark ? "#8e8d87" : "#727069";
  const pillBg = isDark ? "#242421" : "#ebeae4";
  const green = isDark ? "#92d492" : "#2d7a3a";

  const fontSans = isRu
    ? "'Onest', 'Inter', -apple-system, sans-serif"
    : "'Inter', -apple-system, sans-serif";

  // 1. Fill Outer Canvas Background
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, width, height);

  // 2. Outer Margin / Card Container
  const pad = 48;
  const cX = pad;
  const cY = pad;
  const cW = width - pad * 2;
  const cH = height - pad * 2;
  const cR = 36;

  // Draw card with rounded corners
  ctx.save();
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(cX, cY, cW, cH, cR);
  } else {
    ctx.rect(cX, cY, cW, cH);
  }
  ctx.fillStyle = cardBg;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = cardBorder;
  ctx.stroke();
  ctx.clip();

  // 3. Header: Brand & Pulsar Waves (Joy Division Unknown Pleasures Style)
  const headerTop = cY + 48;

  // Brand title
  ctx.fillStyle = ink;
  ctx.font = `800 26px ${fontSans}`;
  ctx.textAlign = "left";
  ctx.fillText("MUSICMYLOVE", cX + 44, headerTop + 14);

  // Draw miniature pulsar wave lines
  const waveStartX = cX + 44;
  const waveWidth = cW - 88;
  const waveCenterY = headerTop + 64;

  ctx.strokeStyle = isDark ? "rgba(255, 255, 255, 0.2)" : "rgba(0, 0, 0, 0.2)";
  ctx.lineWidth = 1.5;

  for (let line = 0; line < 5; line++) {
    const lineY = waveCenterY + (line - 2) * 9;
    ctx.beginPath();
    ctx.moveTo(waveStartX, lineY);

    const mid = waveStartX + waveWidth / 2;
    const spread = 180;

    for (let x = waveStartX; x <= waveStartX + waveWidth; x += 4) {
      const distFromMid = Math.abs(x - mid);
      let bump = 0;
      if (distFromMid < spread) {
        const factor = Math.cos((distFromMid / spread) * (Math.PI / 2));
        const freq = 0.05 + line * 0.01;
        bump = -Math.sin((x - mid) * freq) * 16 * factor;
      }
      ctx.lineTo(x, lineY + bump);
    }
    ctx.stroke();
  }

  // 4. Section 1: Top user tracks (Seeds / Favorites)
  let curY = waveCenterY + 60;

  // Section badge
  const sec1Badge = isRu ? "МОЙ ВЫБОР" : "MY SELECTION";
  drawSectionHeader(ctx, cX + 44, curY, sec1Badge, fontSans, muted);
  curY += 38;

  for (let i = 0; i < topTracks.length; i++) {
    const track = topTracks[i];
    drawTrackRow(ctx, {
      x: cX + 44,
      y: curY,
      width: cW - 88,
      index: i + 1,
      title: track.title,
      artist: track.artist,
      fontSans,
      ink,
      muted,
      pillBg,
      isDark,
    });
    curY += 72;
  }

  // 5. Minimal Sleek Divider
  curY += 12;
  ctx.strokeStyle = isDark ? "rgba(255, 255, 255, 0.12)" : "rgba(0, 0, 0, 0.1)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cX + 44, curY);
  ctx.lineTo(cX + cW - 44, curY);
  ctx.stroke();

  curY += 34;

  // 6. Section 2: Recommendations
  const sec2Badge = isRu ? "РЕКОМЕНДАЦИИ" : "RECOMMENDATIONS";
  drawSectionHeader(ctx, cX + 44, curY, sec2Badge, fontSans, muted);
  curY += 38;

  for (let i = 0; i < recTracks.length; i++) {
    const track = recTracks[i];
    drawTrackRow(ctx, {
      x: cX + 44,
      y: curY,
      width: cW - 88,
      index: i + 1,
      title: track.title,
      artist: track.artist,
      fontSans,
      ink,
      muted,
      pillBg,
      isDark,
    });
    curY += 72;
  }

  // 7. Footer
  const footerY = cY + cH - 40;
  ctx.fillStyle = muted;
  ctx.font = `500 14px ${fontSans}`;
  ctx.textAlign = "left";
  ctx.fillText("musicmylove.vercel.app", cX + 44, footerY);

  ctx.restore();
}

function drawSectionHeader(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  fontSans: string,
  muted: string
) {
  ctx.font = `800 12px ${fontSans}`;
  ctx.fillStyle = muted;
  ctx.textAlign = "left";
  ctx.fillText(text, x, y);
}

interface DrawRowOptions {
  x: number;
  y: number;
  width: number;
  index: number;
  title: string;
  artist: string;
  fontSans: string;
  ink: string;
  muted: string;
  pillBg: string;
  isDark: boolean;
  headIndex?: number;
  liftScore?: number;
  popularityPercentile?: number;
  green?: string;
  isRu?: boolean;
}

function drawTrackRow(ctx: CanvasRenderingContext2D, opt: DrawRowOptions) {
  const rowH = 60;
  const rowY = opt.y;

  // Background subtle card
  ctx.fillStyle = opt.isDark ? "rgba(255, 255, 255, 0.03)" : "rgba(0, 0, 0, 0.02)";
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(opt.x, rowY, opt.width, rowH, 12);
    ctx.fill();
  } else {
    ctx.fillRect(opt.x, rowY, opt.width, rowH);
  }

  // Track index number
  ctx.fillStyle = opt.muted;
  ctx.font = `700 15px ${opt.fontSans}`;
  ctx.textAlign = "center";
  const numStr = opt.index < 10 ? `0${opt.index}` : `${opt.index}`;
  ctx.fillText(numStr, opt.x + 24, rowY + 36);

  // Title and Artist
  ctx.textAlign = "left";
  ctx.fillStyle = opt.ink;
  ctx.font = `700 18px ${opt.fontSans}`;

  // Truncate title if long
  const maxTitleWidth = opt.width - 80;
  let displayTitle = opt.title;
  if (ctx.measureText(displayTitle).width > maxTitleWidth) {
    while (displayTitle.length > 3 && ctx.measureText(`${displayTitle}…`).width > maxTitleWidth) {
      displayTitle = displayTitle.slice(0, -1);
    }
    displayTitle = `${displayTitle}…`;
  }
  ctx.fillText(displayTitle, opt.x + 52, rowY + 28);

  ctx.fillStyle = opt.muted;
  ctx.font = `400 14px ${opt.fontSans}`;
  let displayArtist = opt.artist;
  if (ctx.measureText(displayArtist).width > maxTitleWidth) {
    while (displayArtist.length > 3 && ctx.measureText(`${displayArtist}…`).width > maxTitleWidth) {
      displayArtist = displayArtist.slice(0, -1);
    }
    displayArtist = `${displayArtist}…`;
  }
  ctx.fillText(displayArtist, opt.x + 52, rowY + 48);
}

/**
 * Downloads the Taste Card as a PNG file.
 */
export function downloadTasteCardPng(options: TasteCardOptions, filename = "musicmylove-taste.png"): void {
  if (typeof document === "undefined") return;
  const canvas = document.createElement("canvas");
  renderTasteCardToCanvas(canvas, options);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, "image/png");
}

/**
 * Shares the Taste Card using the Web Share API (native share sheet on mobile),
 * or falls back to downloading the PNG.
 */
export async function shareTasteCardImage(options: TasteCardOptions): Promise<boolean> {
  if (typeof document === "undefined") return false;
  const canvas = document.createElement("canvas");
  renderTasteCardToCanvas(canvas, options);

  return new Promise((resolve) => {
    canvas.toBlob(async (blob) => {
      if (!blob) {
        resolve(false);
        return;
      }
      const file = new File([blob], "musicmylove-taste.png", { type: "image/png" });
      if (typeof navigator !== "undefined" && navigator.canShare && navigator.canShare({ files: [file] })) {
        try {
          await navigator.share({
            title: "MusicMyLove Taste Profile",
            text: options.isRussian
              ? "Мой музыкальный вкус и рекомендации TasteLift AI на MusicMyLove!"
              : "My music taste profile and TasteLift AI recommendations on MusicMyLove!",
            url: "https://musicmylove.vercel.app",
            files: [file],
          });
          resolve(true);
          return;
        } catch {
          // user cancelled or share failed
        }
      }
      // Fallback: download PNG directly
      downloadTasteCardPng(options);
      resolve(true);
    }, "image/png");
  });
}
