// Intersection Observer — trigger animations when sections enter view
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('in-view');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll('.result-card, .pipe-step, .cluster-card, .concl-block, .scard').forEach(el => {
  observer.observe(el);
});

// Active nav link on scroll
const sections = document.querySelectorAll('section[id]');
const navLinks  = document.querySelectorAll('.nav-links a');

const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.getAttribute('id');
      navLinks.forEach(a => {
        a.style.color = a.getAttribute('href') === `#${id}`
          ? 'var(--green)'
          : '';
      });
    }
  });
}, { threshold: 0.4 });

sections.forEach(s => sectionObserver.observe(s));

// RPD fill bars — re-trigger on scroll into view
const rpdFills = document.querySelectorAll('.rpd-fill, .dt-fill, .rc-r2-fill');
const fillObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animationPlayState = 'running';
      fillObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });

rpdFills.forEach(el => {
  el.style.animationPlayState = 'paused';
  fillObserver.observe(el);
});

/* INFERENCE API LOGIC */
const API_URL = 'https://amsh4-hypesoilnet.hf.space/predict';
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('npz-file');
const loadingDiv = document.getElementById('loading');
const resultsPanel = document.getElementById('results-panel');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  uploadForm.addEventListener(eventName, preventDefaults, false);
});
function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

['dragenter', 'dragover'].forEach(eventName => {
  uploadForm.addEventListener(eventName, () => uploadForm.classList.add('dragover'), false);
});
['dragleave', 'drop'].forEach(eventName => {
  uploadForm.addEventListener(eventName, () => uploadForm.classList.remove('dragover'), false);
});

uploadForm.addEventListener('drop', handleDrop, false);
fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFiles(e.target.files[0]);
});

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length) handleFiles(files[0]);
}

async function handleFiles(file) {
  if (!file.name.endsWith('.npz')) {
      alert("Please upload a raw .npz airborne spectral file.");
      return;
  }
  
  uploadForm.style.display = 'none';
  resultsPanel.style.display = 'none';
  loadingDiv.style.display = 'flex';
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
      const response = await fetch(API_URL, {
          method: 'POST',
          body: formData
      });
      
      const data = await response.json();
      loadingDiv.style.display = 'none';
      
      if (data.status === 'success') {
          renderPredictions(data.predictions);
      } else {
          showError(data.message || 'Error processing file.');
      }
  } catch (err) {
      loadingDiv.style.display = 'none';
      showError('Cannot connect to the Hugging Face API. Is it deployed?');
  }
}

function renderPredictions(preds) {
  resultsPanel.innerHTML = '';
  // Order of display
  const elements = ['B', 'Fe', 'Zn', 'Cu', 'Mn', 'S'];
  
  elements.forEach(el => {
      if(preds[el]) {
          const card = document.createElement('div');
          card.className = 'pred-card';
          card.innerHTML = `
              <div class="pred-el">${el}</div>
              <div class="pred-val">${preds[el].value} <span class="pred-unit">${preds[el].unit}</span></div>
          `;
          resultsPanel.appendChild(card);
      }
  });
  
  resultsPanel.style.display = 'grid';
  // Show upload again for next file
  uploadForm.style.display = 'block';
  uploadForm.querySelector('p').innerHTML = "Upload another .npz file";
}

function showError(msg) {
  uploadForm.style.display = 'block';
  resultsPanel.style.display = 'block';
  resultsPanel.innerHTML = `<div class="api-error">${msg}</div>`;
}
