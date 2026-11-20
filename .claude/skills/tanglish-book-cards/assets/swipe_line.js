// Optional: add a small "English version →" line to finished Tanglish cards,
// for a 2-image carousel (Tanglish first, English second).
// Put the source PNGs in SITE_DIR/src/ as day-01.png, day-02.png, ... then open
// http://127.0.0.1:8766/index.html (so Poppins is loaded) and run this.
// Set COUNT below. Output PNGs are POSTed back with the same names.
const COUNT = 14;
window.__c = {done: 0, err: null};
(async () => {
  try {
    await document.fonts.load('400 32px "Poppins"'); await document.fonts.ready;
    for (let d = 1; d <= COUNT; d++) {
      const n = String(d).padStart(2, '0');
      const im = new Image(); im.src = 'src/day-' + n + '.png?' + Date.now(); await im.decode();
      const c = document.createElement('canvas'); c.width = im.width; c.height = im.height;
      const x = c.getContext('2d'); x.drawImage(im, 0, 0);
      x.font = '400 32px Poppins, "Avenir Next", sans-serif'; x.fillStyle = '#b5b5b5';
      x.textAlign = 'right'; x.fillText('English version  →', 998, 1292);
      const b = await (await fetch(c.toDataURL('image/png'))).blob();
      const r = await fetch('/day-' + n + '.png', {method: 'POST', body: b});
      if (!r.ok) throw new Error('upload ' + n);
      window.__c.done++;
    }
  } catch (e) { window.__c.err = String(e); }
})();
'started';
