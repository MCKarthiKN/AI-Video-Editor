// usage: node raster_icons.js <icons_dir> <size>
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
(async () => {
  const dir = process.argv[2], size = +process.argv[3] || 512;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: size, height: size } });
  for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.svg'))) {
    const svg = fs.readFileSync(path.join(dir, f), 'utf8')
      .replace(/width="24"/, `width="${size}"`).replace(/height="24"/, `height="${size}"`)
      .replace(/stroke-width="2"/, 'stroke-width="1.6"');
    await page.setContent(`<html><body style="margin:0;background:transparent;color:white">${svg}</body></html>`);
    await page.locator('svg').screenshot({ path: path.join(dir, f.replace('.svg', '.png')), omitBackground: true });
    console.log('ok', f);
  }
  await browser.close();
})();
