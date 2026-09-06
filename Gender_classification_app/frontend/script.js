const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');
const facesList = document.getElementById('facesList');
const faceCount = document.getElementById('faceCount');
const noFacesMsg = document.getElementById('noFacesMsg');
const downloadBtn = document.getElementById('downloadBtn');

// Config
const configBtn = document.getElementById('configBtn');
const configModal = document.getElementById('configModal');
const closeModal = document.getElementById('closeModal');
const thresholdSlider = document.getElementById('thresholdSlider');
const minSizeSlider = document.getElementById('minSizeSlider');
const genderSlider = document.getElementById('genderSlider');
const thresholdVal = document.getElementById('thresholdVal');
const minSizeVal = document.getElementById('minSizeVal');
const genderVal = document.getElementById('genderVal');
const saveConfig = document.getElementById('saveConfig');
const resetConfig = document.getElementById('resetConfig');
const backendInfo = document.getElementById('backendInfo');

let selectedFile = null;
let lastAnnotatedDataUrl = null;
let config = { detection_threshold: 0.6, min_face_size: 40, gender_threshold: 0.0 };

const DEFAULTS = { detection_threshold: 0.6, min_face_size: 40, gender_threshold: 0.0 };

// Load config from backend
async function loadConfig() {
  try {
    const r = await fetch('/config');
    if (r.ok) {
      const j = await r.json();
      config.detection_threshold = j.detection_threshold;
      config.min_face_size = j.min_face_size;
      config.gender_threshold = j.gender_threshold;
      thresholdSlider.value = config.detection_threshold;
      minSizeSlider.value = config.min_face_size;
      genderSlider.value = config.gender_threshold;
      thresholdVal.textContent = config.detection_threshold.toFixed(2);
      minSizeVal.textContent = config.min_face_size;
      genderVal.textContent = parseFloat(config.gender_threshold).toFixed(2);
      backendInfo.textContent = j.detector_backend + ' • ' + j.defaults.note;
    }
  } catch(e){ backendInfo.textContent = 'Could not load config'; }
}
loadConfig();

thresholdSlider.addEventListener('input', () => thresholdVal.textContent = parseFloat(thresholdSlider.value).toFixed(2));
minSizeSlider.addEventListener('input', () => minSizeVal.textContent = minSizeSlider.value);
genderSlider.addEventListener('input', () => genderVal.textContent = parseFloat(genderSlider.value).toFixed(2));

configBtn.addEventListener('click', () => configModal.classList.remove('hidden'));
closeModal.addEventListener('click', () => configModal.classList.add('hidden'));
configModal.addEventListener('click', (e) => { if(e.target===configModal) configModal.classList.add('hidden'); });

saveConfig.addEventListener('click', () => {
  config.detection_threshold = parseFloat(thresholdSlider.value);
  config.min_face_size = parseInt(minSizeSlider.value);
  config.gender_threshold = parseFloat(genderSlider.value);
  configModal.classList.add('hidden');
});

resetConfig.addEventListener('click', () => {
  thresholdSlider.value = DEFAULTS.detection_threshold;
  minSizeSlider.value = DEFAULTS.min_face_size;
  genderSlider.value = DEFAULTS.gender_threshold;
  thresholdVal.textContent = DEFAULTS.detection_threshold.toFixed(2);
  minSizeVal.textContent = DEFAULTS.min_face_size;
  genderVal.textContent = DEFAULTS.gender_threshold.toFixed(2);
  config = {...DEFAULTS};
});

// File handling
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e)=>{ e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', ()=> dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e)=>{
  e.preventDefault(); dropZone.classList.remove('dragover');
  if(e.dataTransfer.files.length){ handleFile(e.dataTransfer.files[0]); }
});
fileInput.addEventListener('change', ()=>{ if(fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file){
  if(!file.type.startsWith('image/')){ alert('Please select an image'); return; }
  selectedFile = file;
  predictBtn.disabled = false;
  // Show preview in drop zone
  const reader = new FileReader();
  reader.onload = (e)=>{
    dropZone.innerHTML = `<img src="${e.target.result}" style="max-height:220px; max-width:100%; border-radius:6px; display:block; margin:0 auto;"><p style="margin-top:8px; font-size:13px; color:#6b7280;">${file.name} • ${(file.size/1024).toFixed(0)} KB</p>`;
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', ()=>{
  selectedFile = null;
  fileInput.value = '';
  predictBtn.disabled = true;
  resultSection.classList.add('hidden');
  lastAnnotatedDataUrl = null;
  dropZone.innerHTML = `<div class="drop-icon">+</div><p><strong>Drag & drop image here</strong> or click to browse</p><p class="small">JPG, PNG • Group or single person</p>`;
});

predictBtn.addEventListener('click', async ()=>{
  if(!selectedFile) return;
  predictBtn.textContent = 'Processing...';
  predictBtn.disabled = true;
  
  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('detection_threshold', config.detection_threshold);
  fd.append('min_face_size', config.min_face_size);
  fd.append('gender_threshold', config.gender_threshold);
  
  try{
    const res = await fetch('/predict', { method:'POST', body: fd });
    const data = await res.json();
    if(!res.ok) throw new Error(data.error || 'Prediction failed');
    renderResults(data);
  }catch(err){
    alert('Error: '+err.message);
  }finally{
    predictBtn.textContent = 'Detect & Classify';
    predictBtn.disabled = false;
  }
});

function renderResults(data){
  resultSection.classList.remove('hidden');
  resultSection.scrollIntoView({behavior:'smooth'});
  lastAnnotatedDataUrl = data.annotated_image;
  resultImage.src = data.annotated_image;
  faceCount.textContent = data.num_faces + (data.num_faces===1 ? ' face' : ' faces');
  
  facesList.innerHTML = '';
  if(data.num_faces===0){
    noFacesMsg.classList.remove('hidden');
    facesList.classList.add('hidden');
  } else {
    noFacesMsg.classList.add('hidden');
    facesList.classList.remove('hidden');
    data.faces.forEach((f, idx)=>{
      const isMale = f.gender==='Male';
      const div = document.createElement('div');
      div.className = 'face-item ' + (isMale?'male':'female');
      div.innerHTML = `
        <div>
          <div class="face-name ${isMale?'male':'female'}">#${idx+1} ${f.gender}</div>
          <div class="face-meta">Box [${f.bbox.join(', ')}] • Detection ${(f.detection_confidence*100).toFixed(0)}%</div>
        </div>
        <div class="face-conf">${(f.confidence*100).toFixed(1)}%</div>
      `;
      facesList.appendChild(div);
    });
  }
}

downloadBtn.addEventListener('click', ()=>{
  if(!lastAnnotatedDataUrl) return;
  const a = document.createElement('a');
  a.href = lastAnnotatedDataUrl;
  a.download = 'gender_annotated.jpg';
  a.click();
});
