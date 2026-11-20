// Paste this whole file into the built-in browser's javascript tool while the
// page http://127.0.0.1:8766/index.html (the QUOTE CARDS tool) is open.
// It loads cards.txt from the same folder, renders every card in two styles
// (geo = Poppins "sans", serif = EB Garamond), builds 2x2 collages, and
// POSTs each PNG to the upload server. It runs in the background: check
// progress afterwards with  JSON.stringify(window.__p)
// Expect {"done": cards*2 + collages*2, "err": null, "stage": "finished"}.
window.__p = {done: 0, err: null, stage: 'start'};
(async () => {
  try {
    const txt = await fetch('cards.txt?' + Date.now()).then(r => r.text());
    $('text').value = txt;
    $('sig').value = 'MCkarthiKN';          // exact case, never uppercased
    $('accent').value = '#FFFFFF';          // highlight colour: pure white
    state.ratio = '4:5'; state.theme = 'dark';
    await Promise.all(['EB Garamond', 'Inter', 'Poppins'].flatMap(f => [
      document.fonts.load(`400 40px "${f}"`), document.fonts.load(`600 40px "${f}"`),
      document.fonts.load(`italic 400 40px "${f}"`)]));
    await document.fonts.ready;
    const up = async (c, name) => {
      const b = await (await fetch(c.toDataURL('image/png'))).blob();
      const r = await fetch('/' + name, {method: 'POST', body: b});
      if (!r.ok) throw new Error('upload ' + name);
      window.__p.done++;
    };
    const pad = n => String(n).padStart(2, '0');
    for (const [st, pre] of [['geo', 'sans'], ['serif', 'serif']]) {
      state.style = st; window.__p.stage = pre; draw();
      const cvs = [...document.querySelectorAll('#grid canvas')];
      window.__p.n = cvs.length; window.__p.size = cvs[0].width + 'x' + cvs[0].height;
      for (let i = 0; i < cvs.length; i++) await up(cvs[i], `${pre}-card-${pad(i + 1)}.png`);
      for (let g = 0; g < cvs.length / 4; g++) {
        const c = document.createElement('canvas'); c.width = 1080; c.height = 1350;
        const x = c.getContext('2d'); x.fillStyle = '#000'; x.fillRect(0, 0, 1080, 1350);
        const tw = 508, th = 635, gap = 14, mx = 25, my = 33;
        for (let k = 0; k < 4; k++) {
          const cv = cvs[g * 4 + k]; if (!cv) continue;
          const px = mx + (k % 2) * (tw + gap), py = my + Math.floor(k / 2) * (th + gap);
          x.imageSmoothingQuality = 'high'; x.drawImage(cv, px, py, tw, th);
          x.strokeStyle = '#2a2a2a'; x.lineWidth = 2; x.strokeRect(px, py, tw, th);
        }
        await up(c, `${pre}-collage-${pad(g + 1)}.png`);
      }
    }
    window.__p.first = cards()[0].slice(0, 40);
    window.__p.stage = 'finished';
  } catch (e) { window.__p.err = String(e); }
})();
'started';
