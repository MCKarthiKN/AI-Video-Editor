# three.js hologram graphics (higher-quality 3D beats)

Rendered frame-by-frame in headless Chromium, then encoded with ffmpeg.

Layout expected next to the page (in the episode work dir, served over http):
- `npm/node_modules/three` (`npm i three@0.170.0`)
- `gfx/tex/` textures from the `three-globe@2.41.12` npm tarball (`package/example/img/*`, NASA Blue Marble etc.)
  plus `india.png` (equirect mask baked from world-atlas)
- `gfx/geo50.json` from `node geo_export.js <dir with world-atlas + topojson-client> geo50.json`
- `gfx/fonts/LEMONMILK-*.otf` (user-supplied font; capitals only; donationware for commercial use)

Run:
```
python3 -m http.server 8765 --bind 127.0.0.1      # from the work dir
node rec.js http://127.0.0.1:8765/gfx/earth.html frames 0 9.6 25       # or a comma list of times for stills
ffmpeg -framerate 25 -i frames/%05d.png -c:v libx264 -crf 16 -pix_fmt yuv420p out.mp4
```
`earth.html`: close shot over the pin, Earth turns 2.5 deg (10 min of spin) carrying the pin with a comet trail
kept in space, "250 KM" pops on arrival, then a pull-back to the whole spinning Earth. Bloom must stay subtle
(strength ~.55, threshold ~.32); a wide scan band washes the close-up - keep it a thin line.
