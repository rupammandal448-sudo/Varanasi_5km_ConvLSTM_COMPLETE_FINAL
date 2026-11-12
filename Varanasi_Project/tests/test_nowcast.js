// node tests/test_nowcast.js   -> checks (1) JS model == sklearn, (2) end-to-end forecast scenarios
const fs=require('fs'),path=require('path');
const N=require('../web/model.js');
const W=path.join(__dirname,'..','web');
const models=JSON.parse(fs.readFileSync(path.join(W,'models.json')));
const grid=JSON.parse(fs.readFileSync(path.join(W,'grid_static.json')));
const fx=JSON.parse(fs.readFileSync(path.join(__dirname,'..','training','reports','js_parity_fixture.json')));
let bad=0;
for(const v of ['full','lite']){
  let mx=0;fx[v].X.forEach((x,i)=>{mx=Math.max(mx,Math.abs(N.rainProb(models[v],x)-fx[v].p[i]))});
  console.log('parity',v,'max |JS - sklearn| =',mx.toExponential(2));if(mx>1e-4)bad++;
}
// scenarios --------------------------------------------------------------
const obsMs=Date.UTC(2025,6,15,9,0);          // 15 Jul 2025 09:00Z (monsoon)
const iso=ms=>new Date(ms).toISOString().replace('.000','');
function st(id,lat,lon,o){return Object.assign({id,name:id,lat,lon,obs_time:iso(obsMs),n_obs:4,
  temp_c:[30,30.5,31,31.5],rh:[80,78,75,72],pres_hpa:[998,998.6,999,999.4],wind_ms:[3,3,2.5,2.5],wind_dir:[120,120,110,100],rain_mm:[0,0,0]},o||{})}
function run(name,live,age,lat,lon){
  const r=N.forecast(models,Object.assign({cadence_min:60},live),grid,lat||25.30,lon||82.97,obsMs+age*60000);
  console.log('\n['+name+'] age',age,'min ->',r.headline,'|',r.detail,'| p=',r.prob==null?'-':r.prob.toFixed(2),'| variant',r.variant,'| notes',JSON.stringify(r.notes||[]));
  return r;
}
const dry={stations:[st('A',25.45,82.86),st('B',25.27,82.99)]};
const wetNow={stations:[st('A',25.45,82.86,{rain_mm:[3,1.5,0.2],rh:[97,95,92,88],temp_c:[25,26,28,30],pres_hpa:[996,997,998,999]}),
                        st('B',25.27,82.99,{rain_mm:[2.5,1,0],rh:[96,94,90,86],temp_c:[25.5,26.5,28.5,30.5],pres_hpa:[996,997,998,999]})]};
const clearing={stations:[st('A',25.45,82.86,{rain_mm:[0.4,2.5,3],rh:[88,92,95,96],temp_c:[27,26,25.5,25],pres_hpa:[1000,999,998,997]}),
                          st('B',25.27,82.99,{rain_mm:[0.3,2,3.5],rh:[87,92,95,96],temp_c:[27,26,25.5,25],pres_hpa:[1000,999,998,997]})]};
const noRain={stations:[st('A',25.45,82.86,{rain_mm:null,pres_hpa:null}),st('B',25.27,82.99,{rain_mm:null,pres_hpa:null})]};
const r1=run('dry afternoon',dry,5);
const r2=run('raining now',wetNow,5);
const r3=run('rain easing',clearing,5);
run('no rain/pressure data (lite)',noRain,5);
run('obs 40 min old (next update in 20)',dry,40);
const r4=run('expired',dry,65);
run('outside coverage point still computes by nearest cell',dry,5,25.60,82.97);
if(!(r2.prob>r1.prob))bad++;                       // wet scenario must be rainier than dry
if(!(r3.prob<r2.prob))bad++;                       // easing scenario less than raining
if(r4.headline.indexOf('Waiting')<0)bad++;         // after next-update time the forecast must expire
if(r1.horizon>30)bad++;
console.log(bad?'\nFAILED checks: '+bad:'\nALL CHECKS PASSED');process.exit(bad?1:0);
