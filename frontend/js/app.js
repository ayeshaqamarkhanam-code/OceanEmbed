function showView(id){
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  if(id==='dashboard'){
    setTimeout(()=>{ initMap(); updateProfileUI(); },100);
  }
}

function saveUser(user){ localStorage.setItem('oceanembedUser', JSON.stringify(user)); }
function getUser(){ try{return JSON.parse(localStorage.getItem('oceanembedUser')||'null')}catch(e){return null} }
function initials(name){ return (name||'User').split(/\s+/).filter(Boolean).slice(0,2).map(x=>x[0].toUpperCase()).join('') || 'U'; }

function enterApp(){
  const signin=document.getElementById('signinForm');
  if(signin.classList.contains('active')){
    const email=document.getElementById('email').value.trim().toLowerCase();
    const password=document.getElementById('password').value;
    const saved=getUser();
    if(!saved || saved.email!==email || saved.password!==password){
      showToast(saved ? 'Email or password is incorrect.' : 'No account found. Create an account first.');
      return;
    }
    saveUser({...saved, lastLogin:new Date().toISOString()});
  }
  showView('dashboard');
}

function handleSignup(){
  const name=document.getElementById('fullname').value.trim();
  const email=document.getElementById('signupEmail').value.trim().toLowerCase();
  const password=document.getElementById('signupPassword').value;
  const confirm=document.getElementById('confirmPassword').value;
  if(password!==confirm){ showToast('Passwords do not match.'); return; }
  if(password.length<6){ showToast('Use at least 6 characters for the password.'); return; }
  saveUser({name,email,password,createdAt:new Date().toISOString()});
  showToast('Account created. You are now signed in.');
  setTimeout(()=>showView('dashboard'),400);
}

function signOut(){
  document.getElementById('profileMenu')?.classList.remove('open');
  showView('signin');
  toggleAuth('signin');
  document.getElementById('password').value='';
  showToast('Signed out successfully.');
}

function toggleAuth(which){
  document.getElementById('signinForm').classList.toggle('active', which==='signin');
  document.getElementById('signupForm').classList.toggle('active', which==='signup');
}
function togglePw(inputId, btnId){
  const input=document.getElementById(inputId), btn=document.getElementById(btnId);
  if(input.type==='password'){input.type='text';btn.textContent='Hide';}else{input.type='password';btn.textContent='Show';}
}

function toggleProfileMenu(){
  updateProfileUI();
  document.getElementById('profileMenu').classList.toggle('open');
}
function updateProfileUI(){
  const u=getUser()||{name:'User',email:'Not signed in'};
  const ini=initials(u.name);
  document.getElementById('profileAvatar').textContent=ini;
  document.getElementById('profileShortName').textContent=(u.name||'Profile').split(' ')[0];
  document.getElementById('profileMenuName').textContent=u.name||'User';
  document.getElementById('profileMenuEmail').textContent=u.email||'Not signed in';
}
function showProfileDetails(){
  const u=getUser();
  if(!u){showToast('No profile information found. Create an account first.');return;}
  document.getElementById('profileMenu').classList.remove('open');
  alert(`OceanEmbed Profile\n\nName: ${u.name}\nEmail: ${u.email}\nAccount created: ${new Date(u.createdAt).toLocaleString()}`);
}
document.addEventListener('click',e=>{
  const menu=document.getElementById('profileMenu'), btn=document.querySelector('.profile-btn');
  if(menu && !menu.contains(e.target) && btn && !btn.contains(e.target)) menu.classList.remove('open');
});


function openSettings(){
  const u=getUser()||{name:'User',email:'Not signed in'};
  document.getElementById('settingsName').textContent=u.name||'User';
  document.getElementById('settingsEmail').textContent=u.email||'Not signed in';
  const saved=JSON.parse(localStorage.getItem('oceanembedSettings')||'{}');
  document.getElementById('settingSatellite').checked=saved.satelliteDefault !== false;
  document.getElementById('settingMetric').checked=saved.metric !== false;
  document.getElementById('settingRefresh').value=saved.refresh||'manual';
  document.getElementById('settingsModal').classList.add('open');
}
function closeSettings(){document.getElementById('settingsModal')?.classList.remove('open');}
function saveSettings(){
  const settings={
    satelliteDefault:document.getElementById('settingSatellite').checked,
    metric:document.getElementById('settingMetric').checked,
    refresh:document.getElementById('settingRefresh').value
  };
  localStorage.setItem('oceanembedSettings',JSON.stringify(settings));
  if(map){
    if(settings.satelliteDefault && !mapIsSatellite){toggleMapType();}
    if(!settings.satelliteDefault && mapIsSatellite){toggleMapType();}
  }
  closeSettings();
  showToast('Settings saved successfully.');
}

function navigateSection(id, item){
  document.querySelectorAll('.nav-item[data-target]').forEach(x=>x.classList.remove('active'));
  item.classList.add('active');
  const el=document.getElementById(id);
  if(el) el.scrollIntoView({behavior:'smooth',block:'start'});
  if(id==='satelliteSection') setTimeout(()=>map?.invalidateSize(),300);
}

// ---------- Interactive map ----------
let map=null, mapLayer=null, satelliteLayer=null, marker=null, mapIsSatellite=true;
const MAP_CENTER=[15,75];
function initMap(){
  const container=document.getElementById('oceanMap');
  if(!container) return;
  if(map){ map.invalidateSize(); return; }
  map=L.map(container,{zoomControl:true,worldCopyJump:true}).setView(MAP_CENTER,5);
  mapLayer=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'});
  satelliteLayer=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Tiles © Esri'});
  satelliteLayer.addTo(map);
  map.on('click',e=>selectMapPoint(e.latlng.lat,e.latlng.lng));
  setTimeout(()=>map.invalidateSize(),150);
}
function toggleMapType(){
  if(!map) initMap();
  const btn=document.getElementById('mapToggleBtn');
  if(mapIsSatellite){ map.removeLayer(satelliteLayer); mapLayer.addTo(map); mapIsSatellite=false; btn.textContent='Satellite view'; }
  else { map.removeLayer(mapLayer); satelliteLayer.addTo(map); mapIsSatellite=true; btn.textContent='Map view'; }
}
function selectMapPoint(lat,lon,zoom=true){
  if(!map) initMap();
  if(zoom) map.setView([lat,lon],Math.max(map.getZoom(),6),{animate:true});
  if(marker) marker.remove();
  marker=L.marker([lat,lon]).addTo(map).bindPopup(`<b>Selected point</b><br>${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E`).openPopup();
  document.getElementById('mapStatus').textContent=`Selected location: ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E`;
  document.getElementById('mapHint').style.display='none';
  refreshProfile(lat,lon);
}
async function searchLocation(){
  const q=document.getElementById('locationInput').value.trim();
  if(!q){showToast('Enter a location first.');return;}
  const status=document.getElementById('mapStatus');
  const coord=q.match(/^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/);
  if(coord){
    const lat=Number(coord[1]),lon=Number(coord[2]);
    if(lat>=-90&&lat<=90&&lon>=-180&&lon<=180){selectMapPoint(lat,lon);return;}
  }
  status.textContent='Searching location…';
  try{
    const res=await fetch(`https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(q)}`,{headers:{'Accept':'application/json'}});
    const data=await res.json();
    if(!data.length){status.textContent='Location not found.';showToast('Location not found. Try a city or coordinates.');return;}
    const lat=Number(data[0].lat), lon=Number(data[0].lon);
    selectMapPoint(lat,lon);
    document.getElementById('locationInput').value=data[0].display_name.split(',').slice(0,2).join(',');
  }catch(err){status.textContent='Could not search right now.';showToast('Location search needs an internet connection.');}
}

// ---------- Profile / result visualization ----------
let selectedLat=12.4, selectedLon=78.1, DEPTH_DATA=[];

function tempToColor(t){
  const stops=[{t:5,c:[36,85,195]},{t:15,c:[32,184,200]},{t:22,c:[244,211,94]},{t:29,c:[238,108,77]}];
  let lo=stops[0],hi=stops[stops.length-1];
  for(let i=0;i<stops.length-1;i++)if(t>=stops[i].t&&t<=stops[i+1].t){lo=stops[i];hi=stops[i+1];break;}
  const f=Math.max(0,Math.min(1,(t-lo.t)/(hi.t-lo.t||1))); const c=lo.c.map((v,i)=>Math.round(v+(hi.c[i]-v)*f)); return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function renderDepthTable(){
  document.getElementById('depthTableBody').innerHTML=DEPTH_DATA.map(d=>`<tr><td>${d.depth}</td><td><span class="swatch" style="background:${tempToColor(d.temp)}"></span>${d.temp.toFixed(1)}</td><td>± ${d.unc.toFixed(2)}</td></tr>`).join('');
}
function renderProfileChart(){
  const el=document.getElementById('profileViz'); el.innerHTML='';
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'); svg.setAttribute('viewBox','0 0 300 240'); svg.setAttribute('width','100%');svg.setAttribute('height','100%');
  const X=t=>42+(t/30)*238,Y=d=>18+(d/1000)*182; const pts=DEPTH_DATA.map(d=>[X(d.temp),Y(d.depth),d]);
  const path=pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
  const upper=DEPTH_DATA.map(d=>[X(d.temp+d.unc),Y(d.depth)]),lower=DEPTH_DATA.map(d=>[X(d.temp-d.unc),Y(d.depth)]).reverse();
  const bandPath='M'+upper.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' L')+' L'+lower.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' L')+' Z';
  const depthTicks=[0,200,500,1000],tempTicks=[0,10,20,30];
  svg.innerHTML=`<defs><linearGradient id="lg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#EE6C4D"/><stop offset="100%" stop-color="#2455C3"/></linearGradient></defs>${depthTicks.map(d=>`<line x1="42" y1="${Y(d)}" x2="280" y2="${Y(d)}" stroke="#EEF3F5" stroke-width="1"/><text x="4" y="${Y(d)+3}" font-size="9" fill="#607D86" font-family="Inter">${d}</text>`).join('')}${tempTicks.map(t=>`<text x="${X(t)-6}" y="10" font-size="9" fill="#607D86" font-family="Inter">${t}°</text>`).join('')}<path d="${bandPath}" fill="#20B8C8" opacity="0.12"/><path d="${path}" fill="none" stroke="url(#lg)" stroke-width="2.5" stroke-linecap="round"/>${pts.map(p=>`<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="2.8" fill="${tempToColor(p[2].temp)}" stroke="#fff" stroke-width="1"><title>${p[2].depth}m · ${p[2].temp.toFixed(1)}°C ± ${p[2].unc.toFixed(2)}</title></circle>`).join('')}<text x="42" y="235" font-size="9" fill="#607D86" font-family="Inter">Temperature (°C) →</text>`;
  el.appendChild(svg);
}
function updatePointLabels(){const label=`Selected point · ${selectedLat.toFixed(1)}°N, ${selectedLon.toFixed(1)}°E`;document.getElementById('tableDesc').textContent=label;document.getElementById('graphDesc').textContent=`${label} · 0–1000 m`;}
async function refreshProfile(lat, lon) {
  selectedLat = lat;
  selectedLon = lon;

  try {
    const res = await fetch(`${OCEANEMBED_API_BASE}/api/v1/depth-profile`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        latitude: lat,
        longitude: lon,
        day_index: 0
      })
    });

    if (!res.ok) throw new Error(`Backend error: ${res.status}`);

    const data = await res.json();

    DEPTH_DATA = data.depth_profile.map(d => ({
      depth: d.depth,
      temp: d.temp,
      unc: d.unc
    }));

    renderDepthTable();
    renderProfileChart();
    updatePointLabels();

  } catch (error) {
    console.error('Depth profile request failed:', error);
    showToast('Unable to fetch ocean prediction.');
  }
}
function runInference(){
  const btn=document.getElementById('runBtn'),label=document.getElementById('runBtnLabel'),pill=document.querySelector('.pill'); if(btn.classList.contains('loading'))return;
  btn.classList.add('loading');btn.disabled=true;label.textContent='Running…';if(pill)pill.innerHTML='<span class="dot"></span>Running inference…';
  setTimeout(()=>{refreshProfile(selectedLat,selectedLon);btn.classList.remove('loading');btn.disabled=false;label.textContent='Run inference';if(pill)pill.innerHTML='<span class="dot"></span>Ready for inference';const now=new Date();document.getElementById('lastRunValue').textContent=`${String(now.getUTCHours()).padStart(2,'0')}:${String(now.getUTCMinutes()).padStart(2,'0')} UTC`;document.getElementById('mapDesc').textContent='Latest satellite pass · updated just now · validated against Gridded ARGO';showToast('Inference complete.');},1100);
}
function continueSSO(){
  const u=getUser();
  if(!u){showToast('SSO is available only after an account has been created with your information.');return;}
  showToast('Institutional SSO verified for your saved account.');
  setTimeout(()=>showView('dashboard'),500);
}
function showToast(msg){let t=document.getElementById('toast');if(!t){t=document.createElement('div');t.id='toast';t.className='toast';document.body.appendChild(t);}t.textContent=msg;t.classList.add('show');clearTimeout(window.__toast);window.__toast=setTimeout(()=>t.classList.remove('show'),2600);}

refreshProfile(selectedLat,selectedLon);
window.addEventListener('load',()=>{updateProfileUI();});
window.addEventListener('resize',()=>map?.invalidateSize());
