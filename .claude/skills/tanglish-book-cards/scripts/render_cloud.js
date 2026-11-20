// Render the cards headlessly with Playwright (cloud sessions, no built-in browser).
// Does Step 4 in one command: starts upload_server.py, opens SITE_DIR/index.html in
// Chromium, runs assets/render_job.js, waits for it to finish, stops the server.
//
// Usage:  node render_cloud.js SITE_DIR OUT_DIR [PORT]
//   SITE_DIR  holds index.html (the QUOTE CARDS tool), cards.txt and render_job.js
//   OUT_DIR   receives sans-card-NN.png, sans-collage-NN.png, serif-...
// Exits 1 if the job reports an error or does not finish within 5 minutes.
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const [site, out, portArg] = process.argv.slice(2);
if (!site || !out) { console.error('usage: node render_cloud.js SITE_DIR OUT_DIR [PORT]'); process.exit(2); }
const port = portArg || '8766';
for (const f of ['index.html', 'cards.txt', 'render_job.js'])
  if (!fs.existsSync(path.join(site, f))) { console.error('missing ' + path.join(site, f)); process.exit(2); }
fs.mkdirSync(out, { recursive: true });

const server = spawn('python3', [path.join(__dirname, 'upload_server.py'), out, site, port], { stdio: 'inherit' });
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  let code = 1;
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
    page.on('pageerror', e => console.error('page error:', e.message));
    for (let t = 0; ; t++) {            // wait for the server to come up
      try { await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'networkidle' }); break; }
      catch (e) { if (t > 40) throw e; await sleep(250); }
    }
    const started = await page.evaluate(async () =>
      (0, eval)(await fetch('render_job.js?' + Date.now()).then(r => r.text())));
    console.log('job', started);
    let p;
    for (let t = 0; t < 600; t++) {
      await sleep(500);
      p = await page.evaluate(() => window.__p);
      if (p && (p.err || p.stage === 'finished')) break;
    }
    console.log(JSON.stringify(p));
    const expected = p && p.n ? 2 * (p.n + Math.ceil(p.n / 4)) : null;
    if (p && !p.err && p.stage === 'finished') {
      code = 0;
      if (expected !== null && p.done !== expected) { console.error(`done ${p.done}, expected ${expected}`); code = 1; }
    }
  } catch (e) {
    console.error(e);
  } finally {
    await browser.close();
    server.kill();
    process.exit(code);
  }
})();
