/* ================================================================
   HumanBody3D.js — Organic Medical Virtual Human
   Babylon.js engine. WebGPU when available, WebGL fallback.
   Post-processing: ACES tone-mapping, bloom, FXAA.
   Same public API as the Three.js version.
   ================================================================ */

const HumanBody3D = (() => {
  let engine, scene, camera, glowLayer, shadowGen;
  let bodyRoot, bodyShellNode, organs = {};
  let organHealthData = {};
  let age = 65, gender = 'male';
  let isAnimating = false;
  let bodyVisible = false, labelsVisible = false;
  const _labels = {};
  let glowDisc, glowDiscMat, scanNode, scanMats = [];

  /* ------------------------------------------------------------------
     Organ world-positions (same coordinate frame as Three.js version)
  ------------------------------------------------------------------ */
  const ORGAN_DEFINITIONS = [
    { id: 'brain',      label: 'Brain',      pos: [ 0.00,  1.32,  0.00], scale: 1.28 },
    { id: 'heart',      label: 'Heart',      pos: [-0.04,  0.65,  0.06], scale: 1.10 },
    { id: 'lungs',      label: 'Lungs',      pos: [ 0.00,  0.58, -0.02], scale: 1.18 },
    { id: 'liver',      label: 'Liver',      pos: [ 0.12,  0.22,  0.00], scale: 1.08 },
    { id: 'kidneys',    label: 'Kidneys',    pos: [ 0.00,  0.16, -0.10], scale: 1.05 },
    { id: 'stomach',    label: 'Stomach',    pos: [-0.04,  0.26,  0.04], scale: 1.00 },
    { id: 'intestines', label: 'Intestines', pos: [ 0.00,  0.02,  0.00], scale: 0.92 },
    { id: 'pancreas',   label: 'Pancreas',   pos: [-0.02,  0.14, -0.06], scale: 0.95 },
    { id: 'spleen',     label: 'Spleen',     pos: [-0.16,  0.20, -0.04], scale: 1.00 },
    { id: 'bladder',    label: 'Bladder',    pos: [ 0.00, -0.36,  0.05], scale: 1.15 },
    { id: 'muscle',     label: 'Muscle',     pos: [ 0.42,  0.61,  0.00], scale: 1.10 },
  ];

  const HEALTH_COLORS = [
    { min: 0.90, hex: 0x00e676 },
    { min: 0.75, hex: 0xc6ef00 },
    { min: 0.50, hex: 0xffab40 },
    { min: 0.25, hex: 0xef5350 },
    { min: 0.15, hex: 0xef1a1a },
    { min: 0.08, hex: 0x6b0000 },
    { min: 0.00, hex: 0x1a0000 },
  ];

  function _healthColor(v) {
    const h = HEALTH_COLORS.find(c => v >= c.min) || HEALTH_COLORS[HEALTH_COLORS.length - 1];
    return BABYLON.Color3.FromHexString('#' + h.hex.toString(16).padStart(6, '0'));
  }

  function _traverseMeshes(node, fn) {
    if (node instanceof BABYLON.AbstractMesh) fn(node);
    if (node.getChildMeshes) node.getChildMeshes(false).forEach(fn);
  }

  // ─── Init ──────────────────────────────────────────────────────
  async function init(containerId, onOrganClick) {
    const container = document.getElementById(containerId);
    if (!container) { console.error('[HumanBody3D] container not found:', containerId); return; }
    window._onOrganClick = onOrganClick || (() => {});

    const canvas = document.getElementById('human-3d-canvas');
    if (!canvas) { console.error('[HumanBody3D] canvas #human-3d-canvas not found'); return; }

    // Try WebGPU, fall back to WebGL2/WebGL1
    let webgpu = false;
    if (typeof BABYLON.WebGPUEngine !== 'undefined') {
      try {
        const supported = await BABYLON.WebGPUEngine.IsSupportedAsync;
        if (supported) {
          engine = new BABYLON.WebGPUEngine(canvas, { antialias: true });
          await engine.initAsync();
          webgpu = true;
          console.log('[HumanBody3D] WebGPU active');
        }
      } catch (_) { /* fall through */ }
    }
    if (!webgpu) {
      engine = new BABYLON.Engine(canvas, true, {
        preserveDrawingBuffer: true, stencil: true, antialias: true,
      });
    }

    scene = new BABYLON.Scene(engine);
    scene.clearColor = new BABYLON.Color4(0.016, 0.036, 0.072, 1.0); // #041224 — deep midnight blue
    scene.fogMode  = BABYLON.Scene.FOGMODE_LINEAR;
    scene.fogColor = new BABYLON.Color3(0.016, 0.036, 0.072);
    scene.fogStart = 8; scene.fogEnd = 20;

    _setupCamera(canvas);
    _setupLighting();
    _createScene();
    _createBodyShell();
    _createOrgans();
    _createOrganLabels(container);
    _setupPostProcessing();

    InteractionController.init(scene, camera, engine, organs, onOrganClick);
    MorphingEngine.applyAge(age, bodyRoot, organs, false);
    ResponsiveSystem.init(scene, organs);

    engine.runRenderLoop(() => {
      _animateScene();
      scene.render();
      _updateOrganLabels();
    });

    window.addEventListener('resize', () => engine.resize());
  }

  // ─── Camera ───────────────────────────────────────────────────
  function _setupCamera(canvas) {
    camera = new BABYLON.ArcRotateCamera('cam',
      -Math.PI / 2 + 0.24, Math.PI / 2.2, 3.5,
      new BABYLON.Vector3(0, 0.10, 0), scene);
    camera.attachControl(canvas, true);
    camera.lowerRadiusLimit    = 0.30;  // allow zooming close to individual organs
    camera.upperRadiusLimit    = 10;
    camera.upperBetaLimit      = Math.PI * 0.88;
    camera.lowerBetaLimit      = Math.PI * 0.08;
    camera.wheelDeltaPercentage = 0.01;
    camera.pinchDeltaPercentage = 0.01;
    camera.inertia = 0.7;
    camera.minZ = 0.1;
    camera.maxZ = 100;
  }

  // ─── Lighting ─────────────────────────────────────────────────
  function _setupLighting() {
    // Ambient — warm anatomical theatre feel
    const hemi = new BABYLON.HemisphericLight('hemi', new BABYLON.Vector3(0, 1, 0), scene);
    hemi.diffuse     = new BABYLON.Color3(0.35, 0.40, 0.55);  // cool-neutral sky
    hemi.groundColor = new BABYLON.Color3(0.18, 0.14, 0.10);  // warm ground bounce
    hemi.intensity   = 1.0;

    // Key — warm surgical overhead
    const key = new BABYLON.DirectionalLight('key', new BABYLON.Vector3(-1.5, -5, -3), scene);
    key.position  = new BABYLON.Vector3(1.5, 5, 3);
    key.diffuse   = new BABYLON.Color3(1.0, 0.96, 0.88);  // warm white
    key.intensity = 2.2;
    shadowGen = new BABYLON.ShadowGenerator(2048, key);
    shadowGen.useBlurExponentialShadowMap = true;
    shadowGen.blurKernel = 24;
    shadowGen.darkness   = 0.35;

    // Fill — soft cool-blue from left
    const fill = new BABYLON.DirectionalLight('fill', new BABYLON.Vector3(4, -1, 3), scene);
    fill.diffuse   = new BABYLON.Color3(0.45, 0.60, 0.85);
    fill.intensity = 0.55;

    // Rim — subtle blue-white backlight for silhouette depth
    const rim = new BABYLON.DirectionalLight('rim', new BABYLON.Vector3(0, 1.5, 6), scene);
    rim.diffuse   = new BABYLON.Color3(0.70, 0.82, 1.0);
    rim.intensity = 0.30;

    // Under — warm amber bounce from floor
    const under = new BABYLON.DirectionalLight('under', new BABYLON.Vector3(0, 1, -5), scene);
    under.diffuse   = new BABYLON.Color3(1.0, 0.78, 0.45);
    under.intensity = 0.18;

    // Inner organ illumination — dim warm so tissue reads correctly
    const inner = new BABYLON.PointLight('inner', new BABYLON.Vector3(0, 0.30, 0.08), scene);
    inner.diffuse   = new BABYLON.Color3(1.0, 0.85, 0.65);
    inner.intensity = 0.60;
    inner.range     = 2.5;

    // Scan light travels with the scan ring
    const scanLight = new BABYLON.PointLight('scanLight', new BABYLON.Vector3(0, -1.58, 0), scene);
    scanLight.diffuse   = new BABYLON.Color3(0, 1, 0.933);
    scanLight.intensity = 0;
    scanLight.range     = 2.2;
    scene._scanLight    = scanLight;
  }

  // ─── Scene decorations ────────────────────────────────────────
  function _createScene() {
    // Ground plane (dark, replaces GridHelper)
    const ground = BABYLON.MeshBuilder.CreateGround('ground', { width: 12, height: 12, subdivisions: 1 }, scene);
    ground.position.y = -1.58;
    const gMat = new BABYLON.StandardMaterial('groundMat', scene);
    gMat.diffuseColor  = new BABYLON.Color3(0, 0.027, 0.067);
    gMat.emissiveColor = new BABYLON.Color3(0, 0.008, 0.020);
    gMat.specularColor = BABYLON.Color3.Black();
    ground.material = gMat;
    ground.receiveShadows = true;

    // Glow disc under feet
    glowDisc = BABYLON.MeshBuilder.CreateDisc('glowDisc', { radius: 0.65, tessellation: 64 }, scene);
    glowDisc.rotation.x = -Math.PI / 2;
    glowDisc.position.y = -1.57;
    glowDiscMat = new BABYLON.StandardMaterial('glowDiscMat', scene);
    glowDiscMat.emissiveColor  = new BABYLON.Color3(0.10, 0.35, 0.90);
    glowDiscMat.alpha          = 0.08;
    glowDiscMat.backFaceCulling = false;
    glowDiscMat.disableLighting = true;
    glowDisc.material = glowDiscMat;

    // Scanning rings (horizontal, travel up/down body)
    scanNode = new BABYLON.TransformNode('scanNode', scene);
    scanNode.position.y = -1.58;

    function addScanDisc(r, alpha, emissive) {
      const d = BABYLON.MeshBuilder.CreateDisc('sd_' + r, { radius: r, tessellation: 64 }, scene);
      d.rotation.x = -Math.PI / 2;
      d.parent = scanNode;
      const m = new BABYLON.StandardMaterial('sdm_' + r, scene);
      m.emissiveColor   = emissive || new BABYLON.Color3(0, 1, 0.933);
      m.alpha           = alpha;
      m.backFaceCulling = false;
      m.disableLighting = true;
      d.material = m;
      scanMats.push({ mat: m, baseAlpha: alpha });
      return d;
    }

    function addScanRing(innerR, outerR, alpha) {
      // Approximate ring via torus with tiny cross-section
      const torus = BABYLON.MeshBuilder.CreateTorus('sr_' + innerR, {
        diameter: (innerR + outerR), thickness: (outerR - innerR) * 0.8, tessellation: 64
      }, scene);
      torus.rotation.x = -Math.PI / 2;
      torus.parent = scanNode;
      const m = new BABYLON.StandardMaterial('srm_' + innerR, scene);
      m.emissiveColor   = new BABYLON.Color3(0, 1, 1);
      m.alpha           = alpha;
      m.backFaceCulling = false;
      m.disableLighting = true;
      torus.material = m;
      scanMats.push({ mat: m, baseAlpha: alpha });
      return torus;
    }

    addScanDisc(0.46,  0.015);
    addScanRing(0.44,  0.48, 0.40);
    addScanRing(0.25,  0.27, 0.18);
    addScanDisc(0.015, 0.35);
  }

  // ─── Body Shell (stub — organs-only mode) ─────────────────────
  function _createBodyShell() {
    bodyRoot      = new BABYLON.TransformNode('bodyRoot', scene);
    bodyShellNode = new BABYLON.TransformNode('bodyShell', scene);
    bodyShellNode.parent = bodyRoot;
    bodyShellNode.setEnabled(false);
  }

  function _createBodyShell_UNUSED() {
    bodyRoot      = new BABYLON.TransformNode('bodyRoot', scene);
    bodyShellNode = new BABYLON.TransformNode('bodyShell', scene);
    bodyShellNode.parent = bodyRoot;

    const skinMat = new BABYLON.PBRMaterial('skin', scene);
    skinMat.albedoColor   = new BABYLON.Color3(0.900, 0.720, 0.640);
    skinMat.emissiveColor = new BABYLON.Color3(0.028, 0.014, 0.008);
    skinMat.metallic  = 0.02;
    skinMat.roughness = 0.68;
    skinMat.alpha     = 0.72;
    skinMat.backFaceCulling = false;
    skinMat.subSurface.isTranslucencyEnabled = true;
    skinMat.subSurface.translucencyIntensity = 0.55;
    skinMat.subSurface.tintColor = new BABYLON.Color3(0.95, 0.45, 0.30);
    skinMat.transparencyMode = BABYLON.PBRMaterial.PBRMATERIAL_ALPHABLEND;

    function blob(name, sx, sy, sz, px, py, pz) {
      const m = BABYLON.MeshBuilder.CreateSphere(name, { diameter: 1, segments: 24 }, scene);
      m.scaling.set(sx, sy, sz);
      m.position.set(px, py, pz);
      m.material = skinMat;
      m.parent   = bodyShellNode;
      shadowGen.addShadowCaster(m, true);
      return m;
    }

    function tube(name, rT, rB, h, tess, px, py, pz, rx, rz) {
      const m = BABYLON.MeshBuilder.CreateCylinder(name, {
        height: h, diameterTop: rT * 2, diameterBottom: rB * 2, tessellation: tess || 18,
      }, scene);
      m.position.set(px, py, pz);
      if (rx) m.rotation.x = rx;
      if (rz) m.rotation.z = rz;
      m.material = skinMat;
      m.parent   = bodyShellNode;
      shadowGen.addShadowCaster(m, true);
      return m;
    }

    // ─── Head & Neck ───────────────────────────────────────────────
    blob('head',  0.175, 0.215, 0.185, 0, 1.365, 0);
    blob('earL',  0.050, 0.080, 0.030, -0.192, 1.348, 0.008);
    blob('earR',  0.050, 0.080, 0.030,  0.192, 1.348, 0.008);
    // Neck widens at base to blend into shoulders
    tube('neck',  0.075, 0.096, 0.22, 14, 0, 1.062, 0);

    // ─── Torso ─────────────────────────────────────────────────────
    // Upper chest: wide at shoulders, taper toward waist
    tube('chest',     0.268, 0.220, 0.68, 28, 0, 0.622, 0);
    // Mid abdomen: waist
    tube('abdomen',   0.220, 0.228, 0.44, 24, 0, 0.112, 0);
    // Pelvis
    tube('pelvis',    0.228, 0.218, 0.30, 24, 0, -0.212, 0);
    // Fill sphere at chest-abdomen join (hides cylinder seam)
    blob('waistFill', 0.232, 0.090, 0.228, 0, 0.282, 0.000);
    // Hip flares — organic pelvis width
    blob('hipL',      0.188, 0.165, 0.202, -0.218, -0.258, 0.016);
    blob('hipR',      0.188, 0.165, 0.202,  0.218, -0.258, 0.016);
    // Lower pubic fill
    blob('pubis',     0.162, 0.110, 0.150, 0, -0.378, 0.024);

    // Shoulder caps — large enough to hide arm attachment
    blob('shoulL',    0.212, 0.192, 0.190, -0.352, 0.880, 0.012);
    blob('shoulR',    0.212, 0.192, 0.190,  0.352, 0.880, 0.012);
    // Pectorals
    blob('pecL',      0.206, 0.140, 0.134, -0.148, 0.800, 0.092);
    blob('pecR',      0.206, 0.140, 0.134,  0.148, 0.800, 0.092);

    // ─── Arms — straight down, segments chained end-to-end ──────────
    // Arm X: ±0.422  (shoulder cap outer edge)
    const AX = 0.422;
    // Upper arm: top at 0.845, center 0.610, bottom 0.375
    tube('uarmL',  0.078, 0.062, 0.47, 18, -AX, 0.610, 0);
    tube('uarmR',  0.078, 0.062, 0.47, 18,  AX, 0.610, 0);
    // Elbow: sphere covering bottom of upper arm and top of forearm
    blob('elbowL', 0.078, 0.072, 0.078, -AX, 0.348, 0.008);
    blob('elbowR', 0.078, 0.072, 0.078,  AX, 0.348, 0.008);
    // Forearm: top at 0.325, center 0.090, bottom −0.145
    tube('farmL',  0.062, 0.048, 0.47, 16, -AX, 0.090, 0);
    tube('farmR',  0.062, 0.048, 0.47, 16,  AX, 0.090, 0);
    // Wrist
    blob('wristL', 0.059, 0.053, 0.055, -AX, -0.162, 0.010);
    blob('wristR', 0.059, 0.053, 0.055,  AX, -0.162, 0.010);
    // Hand
    blob('handL',  0.100, 0.072, 0.060, -AX + 0.008, -0.222, 0.025);
    blob('handR',  0.100, 0.072, 0.060,  AX - 0.008, -0.222, 0.025);

    // ─── Legs — straight down, segments chained end-to-end ──────────
    const LX = 0.148;
    // Groin fill: hip-to-thigh bridge
    blob('groinL', 0.122, 0.116, 0.120, -LX, -0.434, 0.014);
    blob('groinR', 0.122, 0.116, 0.120,  LX, -0.434, 0.014);
    // Thigh: top at −0.380, center −0.640, bottom −0.900
    tube('thighL', 0.122, 0.096, 0.52, 22, -LX, -0.640, 0);
    tube('thighR', 0.122, 0.096, 0.52, 22,  LX, -0.640, 0);
    // Knee: sphere covering thigh bottom and calf top
    blob('kneeL',  0.104, 0.096, 0.098, -LX, -0.912, 0.006);
    blob('kneeR',  0.104, 0.096, 0.098,  LX, -0.912, 0.006);
    // Calf: top at −0.905, center −1.155, bottom −1.405
    tube('calfL',  0.090, 0.064, 0.50, 20, -LX, -1.155, 0);
    tube('calfR',  0.090, 0.064, 0.50, 20,  LX, -1.155, 0);
    // Ankle
    blob('ankleL', 0.075, 0.067, 0.071, -LX + 0.005, -1.418, 0.005);
    blob('ankleR', 0.075, 0.067, 0.071,  LX - 0.005, -1.418, 0.005);
    // Foot
    blob('footL',  0.084, 0.064, 0.178, -0.166, -1.470, 0.070);
    blob('footR',  0.084, 0.064, 0.178,  0.166, -1.470, 0.070);
    // Toes
    blob('toeL',   0.072, 0.044, 0.072, -0.163, -1.488, 0.152);
    blob('toeR',   0.072, 0.044, 0.072,  0.163, -1.488, 0.152);

    // ─── Spine hint ─────────────────────────────────────────────────
    const spine = BABYLON.MeshBuilder.CreateCylinder('spine', { height: 1.12, diameter: 0.028, tessellation: 6 }, scene);
    spine.position.set(0, 0.28, -0.22);
    const spineMat = new BABYLON.StandardMaterial('spineMat', scene);
    spineMat.emissiveColor = new BABYLON.Color3(0.667, 0.533, 0.400);
    spineMat.alpha = 0.25;
    spine.material = spineMat;
    spine.parent = bodyShellNode;

    // ─── Ribcage arcs ────────────────────────────────────────────────
    const RIB_Y = [0.72, 0.63, 0.54, 0.45, 0.36, 0.28];
    const RIB_R = [0.210, 0.224, 0.234, 0.234, 0.222, 0.210];
    const ribMat = new BABYLON.StandardMaterial('ribMat', scene);
    ribMat.emissiveColor = new BABYLON.Color3(0.8, 0.667, 0.533);
    ribMat.alpha = 0.20;
    ribMat.backFaceCulling = false;
    RIB_Y.forEach((y, i) => {
      const rib = BABYLON.MeshBuilder.CreateTorus('rib' + i, {
        diameter: RIB_R[i] * 2, thickness: 0.018, tessellation: 32
      }, scene);
      rib.scaling.y = 0.3;
      rib.rotation.x = -Math.PI / 2;
      rib.position.set(0, y, 0.018);
      rib.material = ribMat;
      rib.parent = bodyShellNode;
    });

    // ─── Clavicles ───────────────────────────────────────────────────
    const clavMat = new BABYLON.StandardMaterial('clavMat', scene);
    clavMat.emissiveColor = new BABYLON.Color3(0.8, 0.667, 0.533);
    clavMat.alpha = 0.30;
    [-1, 1].forEach(s => {
      const clav = BABYLON.MeshBuilder.CreateCylinder('clav' + s, { height: 0.28, diameter: 0.024, tessellation: 5 }, scene);
      clav.position.set(s * 0.18, 0.92, 0.06);
      clav.rotation.z = s * 0.52;
      clav.rotation.x = 0.18;
      clav.material = clavMat;
      clav.parent = bodyShellNode;
    });

    // Skin shell must never intercept organ clicks
    bodyShellNode.getChildMeshes(false).forEach(m => { m.isPickable = false; });
  }

  // ─── Organs ───────────────────────────────────────────────────
  function _createOrgans() {
    ORGAN_DEFINITIONS.forEach(def => {
      const organNode = OrganModels.createOrgan(def.id, scene);
      organNode.position.set(...def.pos);
      organNode.scaling.setAll(def.scale);
      organNode.parent = bodyRoot;

      organNode.userData = {
        id: def.id, label: def.label,
        health: 0.85, baseScale: def.scale, ageScale: 1.0,
      };

      // Propagate userData to child meshes so picking works from any sub-mesh
      _traverseMeshes(organNode, m => {
        m.userData     = organNode.userData;
        m._organRoot   = organNode;   // used by InteractionController to reach the root TransformNode
        m._isOrganMesh = true;
        m.receiveShadows = true;
        if (shadowGen) shadowGen.addShadowCaster(m);
        if (m.material) {
          // Cache anatomical base colors so health blending preserves them
          if (m.material.albedoColor  !== undefined) m._baseAlbedo  = m.material.albedoColor.clone();
          if (m.material.diffuseColor !== undefined) m._baseDiffuse = m.material.diffuseColor.clone();
          if (m.material.emissiveColor !== undefined) m._baseEmissive = m.material.emissiveColor.clone();
        }
      });

      organs[def.id] = organNode;
    });

    // Glow layer — makes emissive organs glow softly
    glowLayer = new BABYLON.GlowLayer('organGlow', scene);
    glowLayer.intensity = 0.22;  // reduced for PBR — emissive is stronger
    glowLayer.customEmissiveColorSelector = (mesh, _sub, material, result) => {
      if (mesh._isOrganMesh && material && material.emissiveColor) {
        result.r = material.emissiveColor.r;
        result.g = material.emissiveColor.g;
        result.b = material.emissiveColor.b;
        result.a = 0.55;
      } else {
        result.set(0, 0, 0, 0);
      }
    };
  }

  // ─── HTML label overlays ──────────────────────────────────────
  function _createOrganLabels(container) {
    if (!container) return;
    const SKIP = new Set(['skin', 'muscle']);
    Object.entries(organs).forEach(([id, organNode]) => {
      if (SKIP.has(id)) return;
      const el = document.createElement('div');
      el.style.cssText = [
        'position:absolute', 'pointer-events:none', 'display:none',
        'color:rgba(0,220,255,0.88)', 'font-size:9px',
        "font-family:'IBM Plex Mono','Courier New',monospace",
        'white-space:nowrap', 'text-shadow:0 0 8px rgba(0,0,0,1)',
        'letter-spacing:0.06em', 'line-height:1',
        'padding:2px 5px', 'background:rgba(0,20,50,0.55)',
        'border-left:1.5px solid rgba(0,188,212,0.6)', 'border-radius:0 3px 3px 0',
      ].join(';');
      el.textContent = organNode.userData.label;
      container.appendChild(el);
      _labels[id] = el;
    });
  }

  function _updateOrganLabels() {
    if (!labelsVisible || !camera) return;
    const w = engine.getRenderWidth();
    const h = engine.getRenderHeight();
    const viewport = camera.viewport.toGlobal(w, h);

    Object.entries(_labels).forEach(([id, el]) => {
      const organNode = organs[id];
      if (!organNode) return;
      const worldPos = organNode.getAbsolutePosition ? organNode.getAbsolutePosition()
        : organNode.position;
      const screenPos = BABYLON.Vector3.Project(
        worldPos,
        BABYLON.Matrix.Identity(),
        scene.getTransformMatrix(),
        viewport
      );
      // Behind camera check
      if (screenPos.z < 0 || screenPos.z > 1) { el.style.display = 'none'; return; }
      el.style.display = 'block';
      el.style.left = (screenPos.x + 10) + 'px';
      el.style.top  = (screenPos.y -  7) + 'px';
    });
  }

  // ─── Post-processing ──────────────────────────────────────────
  function _setupPostProcessing() {
    const pipeline = new BABYLON.DefaultRenderingPipeline('pp', true, scene, [camera]);
    pipeline.bloomEnabled   = true;
    pipeline.bloomThreshold = 0.82;  // only very bright spots bloom (vessels, scan ring)
    pipeline.bloomWeight    = 0.18;
    pipeline.bloomKernel    = 48;
    pipeline.bloomScale     = 0.5;
    pipeline.fxaaEnabled    = true;
    pipeline.imageProcessingEnabled = true;
    pipeline.imageProcessing.toneMappingEnabled = true;
    pipeline.imageProcessing.toneMappingType =
      BABYLON.ImageProcessingConfiguration.TONEMAPPING_ACES;
    pipeline.imageProcessing.exposure = 1.15;  // less over-bright
    pipeline.imageProcessing.contrast = 1.12;  // slightly more punch
    pipeline.imageProcessing.vignetteEnabled = true;
    pipeline.imageProcessing.vignetteWeight  = 2.8;  // subtle vignette for depth
    pipeline.imageProcessing.vignetteColor   = new BABYLON.Color4(0, 0, 0, 0);
  }

  // ─── Animation loop extras ───────────────────────────────────
  function _animateScene() {
    const t = Date.now() * 0.001;

    // Subtle body bob
    if (bodyRoot && !isAnimating) {
      bodyRoot.position.y = Math.sin(t * 0.45) * 0.009;
    }

    // Glow disc pulse
    if (glowDiscMat) {
      glowDiscMat.alpha = 0.10 + Math.abs(Math.sin(t * 0.6)) * 0.08;
    }

    // Scanning ring travel
    if (scanNode) {
      const PERIOD = 5.0;
      const phase  = (t / PERIOD) % 1.0;
      const scanY  = phase < 0.5
        ? BABYLON.Scalar.Lerp(-1.58, 1.55, phase * 2)
        : BABYLON.Scalar.Lerp(1.55, -1.58, (phase - 0.5) * 2);
      scanNode.position.y = scanY;

      const normPhase = Math.min(phase, 1 - phase) * 2;
      const alpha     = Math.max(0.12, Math.min(1.0, normPhase * 5));
      scanMats.forEach(({ mat, baseAlpha }) => { mat.alpha = baseAlpha * alpha; });

      const sl = scene._scanLight;
      if (sl) { sl.position.y = scanY; sl.intensity = 0.55 * alpha; }
    }
  }

  // ─── Public health API ────────────────────────────────────────
  function updateOrganHealth(organId, healthValue) {
    const organNode = organs[organId];
    if (!organNode) return;
    organNode.userData.health    = healthValue;
    organHealthData[organId]     = healthValue;
    const col = _healthColor(healthValue);

    _traverseMeshes(organNode, m => {
      if (!m.material) return;
      const mat = m.material;
      // Blend factor: 0 = fully anatomical, 1 = fully health-tinted
      // Healthy (>0.80): barely tinted. Sick (<0.50): strongly tinted.
      const tintBlend = healthValue >= 0.80 ? 0.0
        : healthValue >= 0.50 ? (0.80 - healthValue) / 0.60 * 0.45
        : 0.45 + (0.50 - healthValue) / 0.50 * 0.55;
      if (mat.albedoColor !== undefined) {
        const base = m._baseAlbedo || mat.albedoColor;
        mat.albedoColor = BABYLON.Color3.Lerp(base, col, tintBlend);
      }
      if (mat.diffuseColor !== undefined) {
        const base = m._baseDiffuse || mat.diffuseColor;
        mat.diffuseColor = BABYLON.Color3.Lerp(base, col, tintBlend);
      }
      // Emissive: only pulse when unhealthy
      const emissScale = healthValue < 0.50 ? tintBlend * 0.20 : 0.0;
      if (mat.emissiveColor !== undefined && m._baseEmissive !== undefined) {
        mat.emissiveColor = BABYLON.Color3.Lerp(m._baseEmissive, col.scale(emissScale), tintBlend);
      }
      if (mat.alpha !== undefined && mat.alpha < 0.50) {
        // Only adjust alpha for already-transparent meshes (vessel wires, etc)
        mat.alpha = Math.max(0.10, mat.alpha);
      }
    });

    // Update label colour — cyan when healthy, shift to health-tint when sick
    const labelColor = healthValue >= 0.80 ? 'rgba(0,220,255,0.88)'
      : '#' + Math.round(col.r*255).toString(16).padStart(2,'0')
            + Math.round(col.g*255).toString(16).padStart(2,'0')
            + Math.round(col.b*255).toString(16).padStart(2,'0');
    const lbl = _labels[organId];
    if (lbl) { lbl.style.color = labelColor; lbl.style.borderLeftColor = labelColor; }

    ResponsiveSystem.onHealthUpdate(organId, healthValue);
  }

  function updateAge(newAge) {
    age = newAge;
    MorphingEngine.applyAge(age, bodyRoot, organs, true);
  }

  function updateGender(newGender) {
    gender = newGender;
    MorphingEngine.applyGender(gender, bodyRoot, organs, age);
  }

  function animateSeries(series, times, speedMultiplier) {
    if (!series || series.length < 2) return;
    isAnimating = true;
    let idx = 0;
    const frameMs = (4000 / series.length) / Math.max(1, speedMultiplier || 1);
    function step() {
      if (idx >= series.length) { isAnimating = false; return; }
      Object.entries(series[idx]).forEach(([id, h]) => updateOrganHealth(id, h));
      const sc = document.getElementById('organ-time-scrubber');
      const lb = document.getElementById('scrubber-label');
      if (sc) sc.value = idx;
      if (lb) lb.textContent = `t = ${(times[idx] || 0).toFixed(1)} h`;
      idx++;
      setTimeout(step, frameMs);
    }
    step();
  }

  function playMedicationEffect(organIds)    { ResponsiveSystem.playMedicationAbsorption(organIds); }
  function playDiseaseEffect(organHealthMap) { ResponsiveSystem.playDiseaseWave(organHealthMap); }
  function setCameraPreset(name)             { InteractionController.goToPreset(name); }

  function zoomCamera(direction) {
    // direction: +1 = zoom in (decrease radius), -1 = zoom out (increase radius)
    if (!camera) return;
    const step   = camera.radius * 0.22;  // 22% of current distance per click
    const target = Math.max(camera.lowerRadiusLimit, Math.min(camera.upperRadiusLimit, camera.radius - direction * step));
    AnimationUtils.tween({ r: camera.radius }, { r: target }, 320,
      v => { camera.radius = v.r; }, null, AnimationUtils.easeInOutQuad);
  }

  function toggleBodyVisibility() {
    bodyVisible = !bodyVisible;
    if (bodyShellNode) bodyShellNode.setEnabled(bodyVisible);
    const btn = document.getElementById('body-vis-btn');
    if (btn) {
      btn.style.color       = bodyVisible ? 'rgba(255,255,255,0.65)' : '#00bcd4';
      btn.style.borderColor = bodyVisible ? 'rgba(255,255,255,0.25)' : 'rgba(0,188,212,0.6)';
      btn.textContent       = bodyVisible ? '◉ Body' : '○ Body';
    }
  }

  function toggleLabels() {
    labelsVisible = !labelsVisible;
    if (!labelsVisible) Object.values(_labels).forEach(el => { el.style.display = 'none'; });
    const btn = document.getElementById('labels-vis-btn');
    if (btn) {
      btn.style.color       = labelsVisible ? '#00bcd4' : 'rgba(255,255,255,0.65)';
      btn.style.borderColor = labelsVisible ? 'rgba(0,188,212,0.6)' : 'rgba(255,255,255,0.25)';
    }
  }

  function selectOrgan(organIdOrObj) {
    const id = typeof organIdOrObj === 'string' ? organIdOrObj : organIdOrObj && organIdOrObj.userData && organIdOrObj.userData.id;
    const organNode = organs[id];
    if (!organNode || !window._onOrganClick) return;
    window._onOrganClick(organNode.userData.id, organNode.userData.health);
  }

  function dispose() {
    if (engine) {
      engine.stopRenderLoop();
      InteractionController.destroy();
      ResponsiveSystem.destroy();
      window.removeEventListener('resize', () => engine.resize());
      Object.values(_labels).forEach(el => { if (el.parentNode) el.parentNode.removeChild(el); });
      scene.dispose();
      engine.dispose();
    }
  }

  // ── Organ focus (X-ray reveal) ────────────────────────────────────────────
  const _shellMatBaseAlpha = new Map();

  function _tweenBodyShellAlpha(targetAlpha, duration) {
    if (!bodyShellNode) return;
    const mats = new Set();
    bodyShellNode.getChildMeshes(false).forEach(m => { if (m.material) mats.add(m.material); });
    mats.forEach(mat => {
      if (!_shellMatBaseAlpha.has(mat)) _shellMatBaseAlpha.set(mat, mat.alpha);
      AnimationUtils.tween({ a: mat.alpha }, { a: targetAlpha }, duration,
        v => { mat.alpha = v.a; });
    });
  }

  function focusOrgan(organId) {
    // Fade body shell to near-transparent — X-ray view
    _tweenBodyShellAlpha(0.06, 550);
    // Brighten selected organ; dim all others
    Object.entries(organs).forEach(([id, node]) => {
      const seen = new Set();
      _traverseMeshes(node, m => {
        if (!m.material || seen.has(m.material)) return;
        seen.add(m.material);
        if (m._baseOrganAlpha === undefined) m._baseOrganAlpha = m.material.alpha;
        const target = id === organId ? m._baseOrganAlpha : Math.min(0.18, m._baseOrganAlpha);
        AnimationUtils.tween({ a: m.material.alpha }, { a: target }, 550,
          v => { m.material.alpha = v.a; });
      });
    });
  }

  function focusOverview() {
    // Restore body shell
    if (bodyShellNode) {
      const mats = new Set();
      bodyShellNode.getChildMeshes(false).forEach(m => { if (m.material) mats.add(m.material); });
      mats.forEach(mat => {
        const base = _shellMatBaseAlpha.get(mat) || 0.72;
        AnimationUtils.tween({ a: mat.alpha }, { a: base }, 550, v => { mat.alpha = v.a; });
      });
    }
    // Restore all organ alphas
    Object.values(organs).forEach(node => {
      const seen = new Set();
      _traverseMeshes(node, m => {
        if (!m.material || seen.has(m.material) || m._baseOrganAlpha === undefined) return;
        seen.add(m.material);
        const base = m._baseOrganAlpha;
        delete m._baseOrganAlpha;
        AnimationUtils.tween({ a: m.material.alpha }, { a: base }, 550, v => { m.material.alpha = v.a; });
      });
    });
  }

  return {
    init, updateOrganHealth, updateAge, updateGender,
    animateSeries, selectOrgan, setCameraPreset, zoomCamera,
    toggleBodyVisibility, toggleLabels,
    playMedicationEffect, playDiseaseEffect, focusOrgan, focusOverview, dispose,
  };
})();
