const tj=require(process.argv[2]+'/node_modules/topojson-client');
const land=require(process.argv[2]+'/node_modules/world-atlas/land-50m.json');
const c=require(process.argv[2]+'/node_modules/world-atlas/countries-50m.json');
const rings=g=>g.type==='Polygon'?g.coordinates:g.coordinates.flat();
const L=tj.feature(land,land.objects.land).features.flatMap(f=>rings(f.geometry));
const india=rings(tj.feature(c,c.objects.countries).features.find(f=>f.id==='356').geometry);
const r=a=>a.map(x=>x.map(p=>[+p[0].toFixed(3),+p[1].toFixed(3)]));
require('fs').writeFileSync(process.argv[3],JSON.stringify({land:r(L),india:r(india)}));
