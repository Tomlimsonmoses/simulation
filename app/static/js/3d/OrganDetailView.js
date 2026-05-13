/* ================================================================
   OrganDetailView.js — Isolated 3D organ detail modal
   Opens a full-screen modal with its own Babylon.js scene showing
   just the clicked organ at full resolution with free orbit control.
   ================================================================ */

const OrganDetailView = (() => {
  let _engine = null, _scene = null, _organNode = null;
  let _currentId = null, _modal = null;

  const ORGAN_LABELS = {
    brain: 'Brain', heart: 'Heart', lungs: 'Lungs',
    liver: 'Liver', kidneys: 'Kidneys', stomach: 'Stomach',
    intestines: 'Intestines', pancreas: 'Pancreas', spleen: 'Spleen',
    bladder: 'Bladder', muscle: 'Muscle', skin: 'Skin',
  };

  // Organ-specific camera distances so each fills the frame nicely
  const ORGAN_RADIUS = {
    brain: 0.55, heart: 0.55, lungs: 0.72, liver: 0.60, kidneys: 0.62,
    stomach: 0.52, intestines: 0.75, pancreas: 0.50, spleen: 0.50,
    bladder: 0.50, muscle: 0.55, skin: 0.80,
  };

  const HEALTH_BANDS = [
    { min: 0.90, label: 'Healthy',  color: '#00e676' },
    { min: 0.75, label: 'Good',     color: '#c6ef00' },
    { min: 0.50, label: 'Impaired', color: '#ffab40' },
    { min: 0.25, label: 'Severe',   color: '#ef5350' },
    { min: 0.08, label: 'Critical', color: '#ef1a1a' },
    { min: 0.00, label: 'Fatal',    color: '#9b0000' },
  ];

  function _healthInfo(h) {
    const v = (h != null && isFinite(h)) ? h : 0.85;
    return HEALTH_BANDS.find(b => v >= b.min) || HEALTH_BANDS[HEALTH_BANDS.length - 1];
  }

  // ── Public: open ────────────────────────────────────────────────
  function open(organId, health, simData, drugNames) {
    _currentId = organId;
    const modal = document.getElementById('organ-detail-modal');
    if (!modal) return;
    _modal = modal;

    // Header info
    const label   = ORGAN_LABELS[organId] || organId.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const h       = (health != null && isFinite(health)) ? health : 0.85;
    const info    = _healthInfo(h);
    const pct     = Math.round(h * 100);

    document.getElementById('odv-title').textContent        = label;
    const statusEl = document.getElementById('odv-status');
    statusEl.textContent   = info.label + ' · ' + pct + '%';
    statusEl.style.color   = info.color;
    const fillEl = document.getElementById('odv-health-bar-fill');
    fillEl.style.width      = pct + '%';
    fillEl.style.background = info.color;

    // Store for cell-view button
    modal._simData   = simData;
    modal._drugNames = drugNames;
    modal._health    = h;

    // Show
    modal.style.display = 'flex';
    requestAnimationFrame(() => modal.classList.add('visible'));

    _initScene(organId);
  }

  // ── Public: close ───────────────────────────────────────────────
  function close() {
    const modal = document.getElementById('organ-detail-modal');
    if (!modal) return;
    modal.classList.remove('visible');
    setTimeout(() => {
      modal.style.display = 'none';
      _destroyScene();
    }, 280);
  }

  // ── Scene lifecycle ─────────────────────────────────────────────
  function _destroyScene() {
    if (_scene)  { _scene.dispose();  _scene  = null; }
    if (_engine) { _engine.stopRenderLoop(); _engine.dispose(); _engine = null; }
    _organNode = null;
  }

  function _initScene(organId) {
    _destroyScene();
    const canvas = document.getElementById('odv-canvas');
    if (!canvas) return;

    const _build = () => _buildScene(organId);

    // Try WebGPU first, fall back to WebGL2
    if (typeof BABYLON.WebGPUEngine !== 'undefined') {
      const gpuEngine = new BABYLON.WebGPUEngine(canvas, { antialias: true });
      gpuEngine.initAsync()
        .then(() => { _engine = gpuEngine; _build(); })
        .catch(() => { _engine = new BABYLON.Engine(canvas, true); _build(); });
    } else {
      _engine = new BABYLON.Engine(canvas, true);
      _build();
    }
  }

  function _buildScene(organId) {
    _scene = new BABYLON.Scene(_engine);
    _scene.clearColor = new BABYLON.Color4(0.010, 0.014, 0.032, 1.0);

    // ─── Camera ──────────────────────────────────────────────────
    const radius = ORGAN_RADIUS[organId] || 0.6;
    const cam = new BABYLON.ArcRotateCamera(
      'odv-cam', -Math.PI / 2, Math.PI / 2.3, radius,
      BABYLON.Vector3.Zero(), _scene
    );
    cam.lowerRadiusLimit   = radius * 0.25;
    cam.upperRadiusLimit   = radius * 4.0;
    cam.wheelDeltaPercentage = 0.015;
    cam.minZ = 0.01;
    cam.attachControl(document.getElementById('odv-canvas'), true);

    // ─── Lighting ────────────────────────────────────────────────
    const hemi = new BABYLON.HemisphericLight('h', new BABYLON.Vector3(0, 1, 0), _scene);
    hemi.diffuse     = new BABYLON.Color3(0.36, 0.40, 0.52);
    hemi.groundColor = new BABYLON.Color3(0.20, 0.14, 0.10);
    hemi.intensity   = 0.80;

    const key = new BABYLON.DirectionalLight('key', new BABYLON.Vector3(-1.2, -4, -2.5), _scene);
    key.diffuse   = new BABYLON.Color3(1.00, 0.96, 0.88);
    key.intensity = 3.2;

    const fill = new BABYLON.DirectionalLight('fill', new BABYLON.Vector3(3, -0.5, 2.5), _scene);
    fill.diffuse   = new BABYLON.Color3(0.42, 0.60, 0.90);
    fill.intensity = 0.72;

    const rim = new BABYLON.DirectionalLight('rim', new BABYLON.Vector3(0, 1.5, 5), _scene);
    rim.diffuse   = new BABYLON.Color3(0.75, 0.88, 1.00);
    rim.intensity = 0.38;

    const under = new BABYLON.DirectionalLight('under', new BABYLON.Vector3(0, 1.5, -4), _scene);
    under.diffuse   = new BABYLON.Color3(1.0, 0.82, 0.50);
    under.intensity = 0.22;

    // Point light from inside — makes translucent organs glow beautifully
    const inner = new BABYLON.PointLight('inner', new BABYLON.Vector3(0, 0, 0), _scene);
    inner.diffuse   = new BABYLON.Color3(1.0, 0.88, 0.70);
    inner.intensity = 0.55;
    inner.range     = 1.5;

    // ─── Post-processing ─────────────────────────────────────────
    const pipeline = new BABYLON.DefaultRenderingPipeline('odv-pipe', true, _scene, [cam]);
    pipeline.fxaaEnabled  = true;
    pipeline.bloomEnabled = true;
    pipeline.bloomThreshold = 0.78;
    pipeline.bloomWeight    = 0.28;
    pipeline.bloomKernel    = 56;
    pipeline.imageProcessingEnabled = true;
    pipeline.imageProcessing.toneMappingEnabled = true;
    pipeline.imageProcessing.toneMappingType = BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
    pipeline.imageProcessing.exposure = 1.22;
    pipeline.imageProcessing.contrast = 1.18;

    // Glow layer — makes emissive vessels pop
    const glow = new BABYLON.GlowLayer('odv-glow', _scene);
    glow.intensity = 0.38;

    // ─── Build the organ ─────────────────────────────────────────
    _organNode = OrganModels.createOrgan(organId, _scene);
    if (_organNode) {
      _organNode.position = BABYLON.Vector3.Zero();

      // Compute bounding extent and scale to fit ≈ radius * 0.80 world units
      const meshes = _organNode.getChildMeshes ? _organNode.getChildMeshes(false) : [];
      let maxExt = 0;
      meshes.forEach(m => {
        m.computeWorldMatrix(true);
        const bi = m.getBoundingInfo();
        if (bi) {
          const ext = bi.boundingBox.extendSizeWorld;
          maxExt = Math.max(maxExt, ext.x, ext.y, ext.z);
        }
      });
      if (maxExt > 0.005) {
        const s = (radius * 0.70) / maxExt;
        _organNode.scaling = new BABYLON.Vector3(s, s, s);
      }
    }

    // ─── Slow auto-rotation ──────────────────────────────────────
    let angle = 0;
    _scene.registerBeforeRender(() => {
      if (_organNode) {
        angle += 0.006;
        _organNode.rotation.y = angle;
      }
    });

    // ─── Render loop ─────────────────────────────────────────────
    _engine.runRenderLoop(() => { if (_scene) _scene.render(); });

    // Resize handler
    const onResize = () => { if (_engine) _engine.resize(); };
    window.addEventListener('resize', onResize);
    _scene._odvResize = onResize;
  }

  // ── Wire up UI once DOM is ready ────────────────────────────────
  function init() {
    const modal = document.getElementById('organ-detail-modal');
    if (!modal) return;

    // Close on backdrop click
    modal.addEventListener('click', e => { if (e.target === modal) close(); });

    // Close button
    document.getElementById('odv-close')?.addEventListener('click', close);

    // ESC key
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && modal.style.display !== 'none') close();
    });

    // "Cell Detail" button — close 3D modal, open the D3 cell drawer
    document.getElementById('odv-cell-btn')?.addEventListener('click', () => {
      const id     = _currentId;
      const health = modal._health;
      const sim    = modal._simData;
      const drugs  = modal._drugNames;
      close();
      setTimeout(() => CellView.open(id, health, sim, drugs), 300);
    });
  }

  return { init, open, close };
})();
