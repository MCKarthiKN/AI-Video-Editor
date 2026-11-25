// usage: node rec.js <url> <outdir> <t0> <t1> [fps] [step]  -> outdir/00001.png ...
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [url, out, t0, t1, fps = 25, only] = process.argv.slice(2);
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  p.on('console', m => { if (m.type() === 'error') console.log('page:', m.text()); });
  p.on('pageerror', e => console.log('pageerror:', e.message));
  await p.goto(url); await p.waitForFunction(() => window.render && window.ready);
  await p.evaluate(() => window.ready); await p.evaluate(() => document.fonts.ready);
  const times = only ? only.split(',').map(Number) : null;
  const n = times ? times.length : Math.round((t1 - t0) * fps);
  const st = Date.now();
  for (let i = 0; i < n; i++) {
    const t = times ? times[i] : +t0 + i / fps;
    await p.evaluate(t => window.render(t), t);
    await p.screenshot({ path: `${out}/${String(i + 1).padStart(5, '0')}.png` });
    if (i % 25 === 0) console.log(`frame ${i}/${n} ${((Date.now() - st) / 1000).toFixed(0)}s`);
  }
  await b.close();
})();
