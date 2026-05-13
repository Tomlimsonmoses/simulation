/* ================================================================
   OrganModels.js — Anatomically realistic 3D organ geometries
   Babylon.js. Uses PBRMaterial for subsurface scatter / specularity,
   multi-mesh constructions per organ for biological accuracy.
   ================================================================ */

const OrganModels = (() => {

  // ── PBR material factory ────────────────────────────────────────
  function _pbr(scene, hexColor, opts) {
    opts = opts || {};
    const mat = new BABYLON.PBRMaterial('pbr_' + Math.random().toString(36).slice(2), scene);
    mat.albedoColor   = BABYLON.Color3.FromHexString(hexColor);
    mat.metallic      = (opts.metallic  !== undefined) ? opts.metallic  : 0.0;
    mat.roughness     = (opts.roughness !== undefined) ? opts.roughness : 0.72;
    mat.alpha         = (opts.alpha     !== undefined) ? opts.alpha     : 0.95;
    const emissScale  = (opts.emissive  !== undefined) ? opts.emissive  : 0.08;
    mat.emissiveColor = BABYLON.Color3.FromHexString(hexColor).scale(emissScale);
    mat.backFaceCulling = false;
    if (opts.sss !== false) {
      mat.subSurface.isTranslucencyEnabled = true;
      mat.subSurface.translucencyIntensity = (opts.sss !== undefined) ? opts.sss : 0.38;
      mat.subSurface.tintColor = mat.albedoColor.scale(0.6);
    }
    return mat;
  }

  // ── Standard wire / emissive material ───────────────────────────
  function _wire(scene, hexColor, alpha) {
    const mat = new BABYLON.StandardMaterial('wire_' + Math.random().toString(36).slice(2), scene);
    mat.emissiveColor   = BABYLON.Color3.FromHexString(hexColor);
    mat.disableLighting = true;
    mat.alpha           = (alpha !== undefined) ? alpha : 0.55;
    mat.backFaceCulling = false;
    return mat;
  }

  // ── Primitive helpers ────────────────────────────────────────────
  function _sph(name, d, scene, sx, sy, sz) {
    const m = BABYLON.MeshBuilder.CreateSphere(name, { diameter: d, segments: 22 }, scene);
    if (sx != null) m.scaling.set(sx, sy, sz);
    return m;
  }

  function _cyl(name, dT, dB, h, tess, scene) {
    return BABYLON.MeshBuilder.CreateCylinder(name, {
      height: h, diameterTop: dT, diameterBottom: dB,
      tessellation: tess || 12
    }, scene);
  }

  function _torus(name, diam, thick, tess, scene) {
    return BABYLON.MeshBuilder.CreateTorus(name, { diameter: diam, thickness: thick, tessellation: tess || 18 }, scene);
  }

  function _v3(x, y, z) { return new BABYLON.Vector3(x, y, z); }

  // ── Vascular vessel helper ───────────────────────────────────────
  function _vessel(name, scene, mat, path, r) {
    const radius = r || 0.008;
    const tube = BABYLON.MeshBuilder.CreateTube(name, {
      path: path, radius: radius, cap: BABYLON.Mesh.CAP_ALL, tessellation: 7, updatable: false
    }, scene);
    tube.material = mat;
    return tube;
  }

  // ══════════════════════════════════════════════════════════════════
  // BRAIN — cerebral hemispheres, cerebellum, brainstem, gyri, vessels
  // ══════════════════════════════════════════════════════════════════
  function createBrain(scene) {
    const root = new BABYLON.TransformNode('brain', scene);

    const cortexMat = _pbr(scene, '#e8b4b8', { roughness: 0.82, sss: 0.35, emissive: 0.06 });
    const wmMat     = _pbr(scene, '#f5e0e2', { roughness: 0.88, sss: 0.28, alpha: 0.90 });
    const cbMat     = _pbr(scene, '#d9a0a8', { roughness: 0.80, sss: 0.40, emissive: 0.05 });
    const bsMat     = _pbr(scene, '#c07880', { roughness: 0.75, sss: 0.30 });
    const vessMat   = _wire(scene, '#cc4455', 0.72);

    // Hemispheres
    const lh = _sph('brain_lh', 0.27, scene, 1.02, 0.95, 0.90);
    lh.position.set(-0.062, 0, 0); lh.material = cortexMat; lh.parent = root;

    const rh = _sph('brain_rh', 0.27, scene, 1.02, 0.95, 0.90);
    rh.position.set(0.062, 0, 0); rh.material = cortexMat; rh.parent = root;

    // Interhemispheric fissure
    const fiss = BABYLON.MeshBuilder.CreateBox('brain_fiss', { width: 0.004, height: 0.21, depth: 0.22 }, scene);
    fiss.material = _wire(scene, '#b07878', 0.35); fiss.parent = root;

    // Gyri bumps (12 – 6 per side)
    var gyrusCoords = [
      [-0.07, 0.09, 0.09], [-0.11, 0.06, 0.00], [-0.09,-0.04, 0.10],
      [-0.05, 0.10,-0.06], [-0.10, 0.02,-0.08], [-0.06,-0.06,-0.04],
      [ 0.07, 0.09, 0.09], [ 0.11, 0.06, 0.00], [ 0.09,-0.04, 0.10],
      [ 0.05, 0.10,-0.06], [ 0.10, 0.02,-0.08], [ 0.06,-0.06,-0.04],
    ];
    gyrusCoords.forEach(function(c, i) {
      var g = _sph('gyrus_' + i, 0.058, scene, 1.5, 0.5, 1.3);
      g.position.set(c[0], c[1], c[2]); g.material = wmMat; g.parent = root;
    });

    // Cerebellum
    const cb = _sph('brain_cb', 0.13, scene, 1.6, 0.60, 1.15);
    cb.position.set(0, -0.088, -0.11); cb.material = cbMat; cb.parent = root;
    for (var ci = -2; ci <= 2; ci++) {
      var fold = BABYLON.MeshBuilder.CreateBox('cb_fold_' + ci, { width: 0.20, height: 0.004, depth: 0.02 }, scene);
      fold.position.set(0, -0.088 + ci * 0.012, -0.09);
      fold.material = _wire(scene, '#b08090', 0.28); fold.parent = root;
    }

    // Brainstem (midbrain + pons + medulla)
    const mid = _cyl('brain_mid', 0.046, 0.038, 0.055, 8, scene);
    mid.position.set(0, -0.105, -0.05); mid.rotation.x = 0.30;
    mid.material = bsMat; mid.parent = root;

    const pons = _sph('brain_pons', 0.060, scene, 1.5, 0.7, 0.9);
    pons.position.set(0, -0.138, -0.065); pons.material = bsMat; pons.parent = root;

    const medulla = _cyl('brain_med', 0.032, 0.026, 0.068, 8, scene);
    medulla.position.set(0, -0.175, -0.055); medulla.rotation.x = 0.22;
    medulla.material = bsMat; medulla.parent = root;

    // Surface vessels
    var arcVess = [
      [_v3(-0.03,0.13,0.05), _v3(-0.06,0.10,0.10), _v3(-0.10,0.03,0.08)],
      [_v3( 0.03,0.13,0.05), _v3( 0.06,0.10,0.10), _v3( 0.10,0.03,0.08)],
      [_v3(-0.02,0.11,-0.05), _v3(0.00,0.13,-0.02), _v3(0.02,0.11,-0.05)],
    ];
    arcVess.forEach(function(pts, i) {
      var v = _vessel('bvess_' + i, scene, vessMat, pts, 0.006); v.parent = root;
    });

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // HEART — 4 chambers, aortic arch, coronary vessels, pericardial fat
  // ══════════════════════════════════════════════════════════════════
  function createHeart(scene) {
    const root = new BABYLON.TransformNode('heart', scene);

    const ventricleMat = _pbr(scene, '#cc1a28', { roughness: 0.58, sss: 0.45, emissive: 0.12 });
    const atriaMat     = _pbr(scene, '#e83040', { roughness: 0.62, sss: 0.40, emissive: 0.10 });
    const aortaMat     = _pbr(scene, '#b81020', { roughness: 0.55, sss: 0.30, emissive: 0.08 });
    const coronMat     = _wire(scene, '#ff4060', 0.75);
    const fatMat       = _pbr(scene, '#f0d080', { roughness: 0.90, sss: 0.20, alpha: 0.88 });

    // Left ventricle
    const lv = _sph('heart_lv', 0.20, scene, 0.90, 1.42, 0.80);
    lv.rotation.z = -0.26; lv.position.set(-0.015, -0.015, 0.005);
    lv.material = ventricleMat; lv.parent = root;

    // Right ventricle (crescent)
    const rv = _sph('heart_rv', 0.165, scene, 1.10, 1.18, 0.62);
    rv.rotation.z = 0.18; rv.position.set(0.045, 0.010, 0.008);
    rv.material = _pbr(scene, '#e02035', { roughness: 0.60, sss: 0.42, emissive: 0.10 }); rv.parent = root;

    // Atria
    const la = _sph('heart_la', 0.096, scene, 1.0, 0.82, 1.0);
    la.position.set(-0.055, 0.106, -0.012); la.material = atriaMat; la.parent = root;

    const ra = _sph('heart_ra', 0.090, scene, 1.0, 0.82, 1.0);
    ra.position.set(0.042, 0.100, -0.010);
    ra.material = _pbr(scene, '#d82838', { roughness: 0.62, sss: 0.40 }); ra.parent = root;

    // Aorta ascending + arch
    const aortaAsc = _cyl('aorta_asc', 0.040, 0.034, 0.12, 10, scene);
    aortaAsc.position.set(-0.010, 0.178, 0.012); aortaAsc.material = aortaMat; aortaAsc.parent = root;

    const aortaArch = _torus('aorta_arch', 0.086, 0.020, 14, scene);
    aortaArch.position.set(-0.010, 0.224, 0.005);
    aortaArch.rotation.z = -Math.PI / 5; aortaArch.rotation.x = Math.PI / 7;
    aortaArch.material = aortaMat; aortaArch.parent = root;

    // Pulmonary artery
    const pa = _cyl('pa', 0.030, 0.026, 0.10, 9, scene);
    pa.position.set(0.040, 0.165, 0.018); pa.rotation.z = 0.30;
    pa.material = _pbr(scene, '#3060cc', { roughness: 0.55, sss: 0.25, emissive: 0.10 }); pa.parent = root;

    // Pulmonary veins (×4 stubs)
    var pvPos = [[-0.066,0.108,-0.014],[-0.075,0.095,0.010],[-0.044,0.112,-0.016],[-0.052,0.099,0.008]];
    pvPos.forEach(function(p, i) {
      var pv = _cyl('pv_' + i, 0.014, 0.012, 0.040, 7, scene);
      pv.position.set(p[0], p[1], p[2]); pv.rotation.z = (i < 2 ? -0.4 : -0.3);
      pv.material = _pbr(scene, '#4070dd', { roughness: 0.55, sss: 0.22, emissive: 0.08, alpha: 0.88 }); pv.parent = root;
    });

    // Pericardial fat
    var fatPos = [[0.01,0.07,0.12],[-0.03,0.04,0.12],[0.04,0.02,0.11]];
    fatPos.forEach(function(p, i) {
      var f = _sph('fat_' + i, 0.046, scene, 1.6, 0.5, 1.1);
      f.position.set(p[0], p[1], p[2]); f.material = fatMat; f.parent = root;
    });

    // Coronary vessels
    var cadPts = [_v3(-0.02,0.05,0.10),_v3(-0.01,-0.02,0.10),_v3(0.00,-0.10,0.08),_v3(0.01,-0.16,0.05)];
    var cad = _vessel('cad', scene, coronMat, cadPts, 0.0060); cad.parent = root;

    var cpxPts = [_v3(0.06,0.06,0.05),_v3(0.07,0.00,0.04),_v3(0.05,-0.08,0.03)];
    var cp = _vessel('cpx', scene, coronMat, cpxPts, 0.0050); cp.parent = root;

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // LUNGS — right (3 lobes) + left (2 lobes), bronchi, fissures
  // ══════════════════════════════════════════════════════════════════
  function createLungs(scene) {
    const root = new BABYLON.TransformNode('lungs', scene);

    const lMat  = _pbr(scene, '#d45535', { roughness: 0.78, sss: 0.50, emissive: 0.07, alpha: 0.90 });
    const lMat2 = _pbr(scene, '#c04530', { roughness: 0.80, sss: 0.48, emissive: 0.06, alpha: 0.90 });
    const bronMat = _pbr(scene, '#e8c060', { roughness: 0.70, sss: 0.20 });
    const artMat  = _wire(scene, '#dd3344', 0.60);
    const veinMat = _wire(scene, '#4466cc', 0.55);

    // Right lung – 3 lobes
    const rUpper = _sph('lung_ru', 0.180, scene, 0.70, 1.08, 0.60);
    rUpper.position.set(0.105, 0.085, 0); rUpper.material = lMat; rUpper.parent = root;

    const rMiddle = _sph('lung_rm', 0.145, scene, 0.72, 0.72, 0.58);
    rMiddle.position.set(0.108, -0.048, 0.012); rMiddle.material = lMat; rMiddle.parent = root;

    const rLower = _sph('lung_rl', 0.195, scene, 0.68, 1.00, 0.62);
    rLower.position.set(0.100, -0.172, -0.010); rLower.material = lMat2; rLower.parent = root;

    // Fissures R
    var fissR = BABYLON.MeshBuilder.CreateBox('fissR', { width: 0.21, height: 0.004, depth: 0.13 }, scene);
    fissR.position.set(0.104, -0.115, 0); fissR.rotation.z = -0.44;
    fissR.material = _wire(scene, '#b03828', 0.22); fissR.parent = root;

    var hfissR = BABYLON.MeshBuilder.CreateBox('hfissR', { width: 0.20, height: 0.004, depth: 0.11 }, scene);
    hfissR.position.set(0.106, 0.018, 0.008);
    hfissR.material = _wire(scene, '#b03828', 0.18); hfissR.parent = root;

    // Left lung – 2 lobes
    const lUpper = _sph('lung_lu', 0.188, scene, 0.68, 1.14, 0.58);
    lUpper.position.set(-0.100, 0.066, 0); lUpper.material = lMat; lUpper.parent = root;

    const lLower = _sph('lung_ll', 0.198, scene, 0.66, 1.05, 0.60);
    lLower.position.set(-0.098, -0.160, -0.008); lLower.material = lMat2; lLower.parent = root;

    var fissL = BABYLON.MeshBuilder.CreateBox('fissL', { width: 0.20, height: 0.004, depth: 0.13 }, scene);
    fissL.position.set(-0.098, -0.060, 0); fissL.rotation.z = 0.42;
    fissL.material = _wire(scene, '#b03828', 0.22); fissL.parent = root;

    // Cardiac notch
    var notch = _sph('lung_notch', 0.060, scene, 1.0, 0.7, 0.8);
    notch.position.set(-0.060, 0.020, 0.048);
    notch.material = _pbr(scene, '#a03020', { roughness: 0.90, alpha: 0.30, sss: false }); notch.parent = root;

    // Trachea + carina
    var trachea = _cyl('trachea', 0.032, 0.030, 0.175, 8, scene);
    trachea.position.set(0, 0.235, 0); trachea.material = bronMat; trachea.parent = root;

    // Bronchial tree
    var bronDefs = [
      { p: _v3(0,0.148,0), q: _v3(0.050,0.108,0), r: 0.018 },
      { p: _v3(0,0.148,0), q: _v3(-0.045,0.108,0), r: 0.018 },
      { p: _v3(0.050,0.108,0), q: _v3(0.088,0.082,0.005), r: 0.012 },
      { p: _v3(0.050,0.108,0), q: _v3(0.092,-0.005,0.010), r: 0.012 },
      { p: _v3(0.050,0.108,0), q: _v3(0.088,-0.110,-0.005), r: 0.012 },
      { p: _v3(-0.045,0.108,0), q: _v3(-0.085,0.075,0.005), r: 0.012 },
      { p: _v3(-0.045,0.108,0), q: _v3(-0.082,-0.100,-0.005), r: 0.012 },
    ];
    bronDefs.forEach(function(d, i) {
      var v = _vessel('bron_' + i, scene, bronMat, [d.p, d.q], d.r); v.parent = root;
    });

    // Pulmonary arteries & veins
    var paDefs = [
      [_v3(0.010,0.110,0.008), _v3(0.055,0.080,0.010), _v3(0.092,0.050,0.010)],
      [_v3(-0.010,0.110,0.008), _v3(-0.050,0.080,0.010), _v3(-0.088,0.050,0.010)],
    ];
    paDefs.forEach(function(pts, i) { var v = _vessel('lpa_' + i, scene, artMat, pts, 0.007); v.parent = root; });

    var pvDefs = [
      [_v3(0.096,0.050,-0.008), _v3(0.055,0.080,-0.008), _v3(0.010,0.100,-0.008)],
      [_v3(-0.088,0.050,-0.008), _v3(-0.050,0.080,-0.008), _v3(-0.010,0.100,-0.008)],
    ];
    pvDefs.forEach(function(pts, i) { var v = _vessel('lpv_' + i, scene, veinMat, pts, 0.006); v.parent = root; });

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // LIVER — right/left lobe, caudate, quadrate, gallbladder, portal vein
  // ══════════════════════════════════════════════════════════════════
  function createLiver(scene) {
    const root = new BABYLON.TransformNode('liver', scene);

    const parMat  = _pbr(scene, '#8b3510', { roughness: 0.72, sss: 0.30, emissive: 0.06 });
    const parMat2 = _pbr(scene, '#7a2e0c', { roughness: 0.75, sss: 0.28, emissive: 0.05 });
    const gbMat   = _pbr(scene, '#3a8530', { roughness: 0.55, sss: 0.22, alpha: 0.92 });
    const capsule = _pbr(scene, '#c06030', { roughness: 0.40, sss: 0.15, alpha: 0.18 });
    const veinMat = _wire(scene, '#1a4488', 0.65);
    const artMat  = _wire(scene, '#cc3322', 0.60);

    // Right lobe (dominant)
    const rLobe = _sph('liver_r', 0.26, scene, 1.42, 0.78, 1.18);
    rLobe.position.set(0.048, 0.008, 0); rLobe.material = parMat; rLobe.parent = root;

    // Left lobe
    const lLobe = _sph('liver_l', 0.175, scene, 1.05, 0.68, 0.95);
    lLobe.position.set(-0.125, 0.004, 0.008); lLobe.material = parMat; lLobe.parent = root;

    // Caudate lobe
    const caudate = _sph('liver_caud', 0.075, scene, 0.80, 0.65, 0.70);
    caudate.position.set(-0.025, 0.028, -0.070); caudate.material = parMat2; caudate.parent = root;

    // Quadrate lobe
    const quad = _sph('liver_quad', 0.070, scene, 1.0, 0.55, 0.80);
    quad.position.set(0.018, -0.055, 0.060); quad.material = parMat2; quad.parent = root;

    // Glisson capsule sheen
    const caps = _sph('liver_caps', 0.265, scene, 1.43, 0.80, 1.20);
    caps.position.set(0.048, 0.008, 0); caps.material = capsule; caps.parent = root;

    // Gallbladder
    const gbBody = _sph('gb_body', 0.068, scene, 0.76, 1.30, 0.68);
    gbBody.position.set(0.082, -0.065, 0.050); gbBody.material = gbMat; gbBody.parent = root;

    const gbNeck = _cyl('gb_neck', 0.018, 0.025, 0.045, 8, scene);
    gbNeck.position.set(0.054, -0.038, 0.042); gbNeck.rotation.z = -0.80; gbNeck.rotation.x = 0.20;
    gbNeck.material = gbMat; gbNeck.parent = root;

    // Cystic duct
    var cystPts = [_v3(0.054,-0.028,0.040),_v3(0.030,-0.020,0.035),_v3(0.008,-0.010,0.030)];
    var cd = _vessel('cyst_duct', scene, _wire(scene,'#3a8530',0.55), cystPts, 0.0055); cd.parent = root;

    // Portal vein
    var pvPts = [_v3(0.020,-0.035,0.025),_v3(0.015,-0.020,0.018),_v3(0.008,-0.002,0.010),_v3(-0.010,0.010,0.008),_v3(-0.040,0.015,0.005)];
    var pv = _vessel('portal', scene, veinMat, pvPts, 0.0130); pv.parent = root;

    // Hepatic artery
    var haPts = [_v3(0.018,-0.022,0.022),_v3(0.010,-0.010,0.018),_v3(-0.005,0.004,0.015)];
    var ha = _vessel('hep_art', scene, artMat, haPts, 0.0075); ha.parent = root;

    // Hepatic veins
    var hvPos = [[-0.02,0.040,-0.02],[0.04,0.042,-0.02],[0.09,0.038,-0.02]];
    hvPos.forEach(function(p, i) {
      var hv = _cyl('hep_v_' + i, 0.012, 0.008, 0.055, 6, scene);
      hv.position.set(p[0], p[1], p[2]); hv.rotation.x = 0.25;
      hv.material = veinMat; hv.parent = root;
    });

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // KIDNEYS — bean cortex/medulla, renal pelvis, adrenals, vessels
  // ══════════════════════════════════════════════════════════════════
  function createKidneys(scene) {
    const root = new BABYLON.TransformNode('kidneys', scene);

    const cortexMat  = _pbr(scene, '#6b3018', { roughness: 0.68, sss: 0.32, emissive: 0.06 });
    const medullaMat = _pbr(scene, '#8b4028', { roughness: 0.72, sss: 0.28, emissive: 0.04 });
    const pelvisMat  = _pbr(scene, '#e8c860', { roughness: 0.65, sss: 0.20, alpha: 0.92 });
    const artMat     = _wire(scene, '#cc2233', 0.70);
    const veinMat    = _wire(scene, '#2244cc', 0.65);
    const ureterMat  = _pbr(scene, '#f0d070', { roughness: 0.60, sss: 0.18, alpha: 0.88 });

    [-1, 1].forEach(function(side) {
      var sx = side < 0 ? 'L' : 'R';
      var xOff = side * 0.082;

      // Bean: cortex + medulla + pelvis
      var cortex = _sph('kid_' + sx, 0.118, scene, 0.78, 1.48, 0.52);
      cortex.position.set(xOff, 0, -0.008); cortex.material = cortexMat; cortex.parent = root;

      var medulla = _sph('kid_med_' + sx, 0.085, scene, 0.72, 1.28, 0.48);
      medulla.position.set(xOff, 0, -0.008); medulla.material = medullaMat; medulla.parent = root;

      var pelvis = _sph('kid_pelvis_' + sx, 0.030, scene, 1.2, 0.7, 0.6);
      pelvis.position.set(side * 0.038, 0, 0.012); pelvis.material = pelvisMat; pelvis.parent = root;

      // Adrenal gland
      var adrenal = _sph('adrenal_' + sx, 0.040, scene, 1.2, 0.70, 0.60);
      adrenal.position.set(xOff * 0.9, 0.076, -0.010);
      adrenal.material = _pbr(scene, '#f0a830', { roughness: 0.72, sss: 0.22 }); adrenal.parent = root;

      // Vessels
      var raPts = [_v3(0,0,0.002), _v3(side*0.040,0,0.008), _v3(xOff*0.5,0,0.010)];
      var ra = _vessel('r_art_' + sx, scene, artMat, raPts, 0.0065); ra.parent = root;

      var rvPts = [_v3(0,0,-0.002), _v3(side*0.040,0,-0.006), _v3(xOff*0.5,0,-0.008)];
      var rv = _vessel('r_vein_' + sx, scene, veinMat, rvPts, 0.0080); rv.parent = root;

      // Ureter
      var ureter = _cyl('ureter_' + sx, 0.010, 0.010, 0.12, 6, scene);
      ureter.position.set(side*0.036, -0.082, 0.008); ureter.material = ureterMat; ureter.parent = root;
    });

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // STOMACH — J-shape: fundus, body, pyloric antrum + rugal folds
  // ══════════════════════════════════════════════════════════════════
  function createStomach(scene) {
    const root = new BABYLON.TransformNode('stomach', scene);

    const wallMat  = _pbr(scene, '#e06098', { roughness: 0.78, sss: 0.35, emissive: 0.07 });
    const rugaMat  = _wire(scene, '#c04080', 0.40);
    const sphinMat = _pbr(scene, '#a04060', { roughness: 0.68, sss: 0.22 });

    // Fundus
    const fundus = _sph('stom_fundus', 0.108, scene, 1.05, 1.0, 0.82);
    fundus.position.set(-0.028, 0.072, 0); fundus.material = wallMat; fundus.parent = root;

    // Body
    const body = _sph('stom_body', 0.152, scene, 1.30, 0.96, 0.72);
    body.rotation.z = Math.PI / 9; body.position.set(0.008, -0.010, 0);
    body.material = wallMat; body.parent = root;

    // Pyloric antrum
    const antrum = _sph('stom_antrum', 0.072, scene, 1.60, 0.58, 0.62);
    antrum.position.set(0.082, -0.080, 0.008); antrum.material = wallMat; antrum.parent = root;

    // Pyloric sphincter ring
    var pylorus = _torus('pylorus', 0.038, 0.016, 10, scene);
    pylorus.position.set(0.115, -0.090, 0.010); pylorus.rotation.z = Math.PI / 2.5;
    pylorus.material = sphinMat; pylorus.parent = root;

    // Rugal folds
    for (var i = 0; i < 5; i++) {
      var angle = -0.4 + i * 0.22;
      var rugFold = BABYLON.MeshBuilder.CreateBox('ruga_' + i, { width: 0.004, height: 0.14, depth: 0.04 }, scene);
      rugFold.position.set(Math.cos(angle)*0.075, -0.010 + i*0.012, Math.sin(angle)*0.052);
      rugFold.rotation.z = angle; rugFold.material = rugaMat; rugFold.parent = root;
    }

    // Oesophageal junction
    var ej = _cyl('stom_ej', 0.022, 0.028, 0.060, 8, scene);
    ej.position.set(-0.050, 0.118, -0.005); ej.rotation.z = 0.4;
    ej.material = sphinMat; ej.parent = root;

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // INTESTINES — small intestine spirals + large intestine frame
  // ══════════════════════════════════════════════════════════════════
  function createIntestines(scene) {
    const root = new BABYLON.TransformNode('intestines', scene);

    const siMat = _pbr(scene, '#cc8818', { roughness: 0.80, sss: 0.28, alpha: 0.90 });
    const liMat = _pbr(scene, '#a05e14', { roughness: 0.82, sss: 0.25, alpha: 0.92 });
    const mesMat = _wire(scene, '#d4a060', 0.22);

    // Small intestine – 3 concentric spiral layers
    for (var layer = 0; layer < 3; layer++) {
      var radius = 0.055 + layer * 0.032;
      var loops  = 6 + layer * 2;
      for (var si = 0; si < loops; si++) {
        var ang = (si / loops) * Math.PI * 2;
        var seg = _cyl('si_' + layer + '_' + si, 0.028, 0.028, 0.058 + layer*0.006, 7, scene);
        seg.material = siMat;
        seg.position.set(
          Math.cos(ang) * radius,
          (si / loops - 0.5) * 0.095 + (layer - 1) * 0.018,
          Math.sin(ang) * radius
        );
        seg.rotation.z = ang + Math.PI / 2;
        seg.parent = root;
      }
    }

    // Large intestine – ascending / transverse / descending / sigmoid
    var colonDefs = [
      { x:  0.115, y: -0.04, h: 0.20, rz: 0 },           // ascending
      { x:  0.000, y:  0.07, h: 0.22, rz: Math.PI/2 },    // transverse
      { x: -0.115, y: -0.04, h: 0.20, rz: 0 },            // descending
      { x: -0.075, y: -0.14, h: 0.10, rz: Math.PI/4 },    // sigmoid
    ];
    colonDefs.forEach(function(s, i) {
      var seg = _cyl('colon_' + i, 0.042, 0.042, s.h, 10, scene);
      seg.material = liMat; seg.position.set(s.x, s.y, 0.005); seg.rotation.z = s.rz;
      seg.parent = root;
    });

    // Haustra (transverse colon bulges)
    for (var hi = -3; hi <= 3; hi++) {
      var haus = _sph('haus_' + hi, 0.050, scene, 1.0, 0.55, 1.0);
      haus.position.set(hi*0.034, 0.098, 0.022);
      haus.material = _pbr(scene, '#b07820', { roughness: 0.85, sss: 0.20, alpha: 0.78 }); haus.parent = root;
    }

    // Mesentery fan
    var mes = BABYLON.MeshBuilder.CreateBox('mesentery', { width: 0.22, height: 0.15, depth: 0.003 }, scene);
    mes.position.set(0.010, -0.005, -0.025); mes.material = mesMat; mes.parent = root;

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // PANCREAS — head, uncinate, body, tail, Wirsung duct, islets
  // ══════════════════════════════════════════════════════════════════
  function createPancreas(scene) {
    const root = new BABYLON.TransformNode('pancreas', scene);

    const mat     = _pbr(scene, '#f09030', { roughness: 0.75, sss: 0.32, emissive: 0.07 });
    const ductMat = _pbr(scene, '#f8d060', { roughness: 0.62, sss: 0.18 });

    // Head
    const head = _sph('panc_head', 0.082, scene, 0.90, 1.10, 0.72);
    head.position.set(0.072, -0.004, 0); head.material = mat; head.parent = root;

    // Uncinate process
    const unci = _sph('panc_unci', 0.050, scene, 1.10, 0.60, 0.80);
    unci.position.set(0.060, -0.042, -0.012); unci.material = mat; unci.parent = root;

    // Body
    const body = _cyl('panc_body', 0.040, 0.044, 0.160, 8, scene);
    body.rotation.z = Math.PI / 2; body.rotation.x = Math.PI / 14; body.position.set(-0.018, 0, 0);
    body.material = mat; body.parent = root;

    // Tail
    const tail = _cyl('panc_tail', 0.024, 0.036, 0.088, 7, scene);
    tail.rotation.z = Math.PI / 2; tail.rotation.x = -Math.PI / 8; tail.position.set(-0.124, 0.014, -0.012);
    tail.material = mat; tail.parent = root;

    // Wirsung duct
    var wirsPts = [_v3(0.060,-0.004,0), _v3(0,0,0), _v3(-0.124,0.014,-0.012)];
    var wd = _vessel('wirsung', scene, ductMat, wirsPts, 0.0070); wd.parent = root;

    // Islets of Langerhans
    var isletPos = [[0.060,-0.004,0.008],[0.000,0.002,0.006],[-0.050,0.008,-0.002],[-0.100,0.012,-0.008]];
    isletPos.forEach(function(p, i) {
      var islet = _sph('islet_' + i, 0.012, scene, 1, 1, 1);
      islet.position.set(p[0], p[1], p[2]);
      islet.material = _pbr(scene, '#ffe0a0', { roughness: 0.50, sss: 0.15, emissive: 0.18 }); islet.parent = root;
    });

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // SPLEEN — wedge with hilum notch, tortuous splenic artery
  // ══════════════════════════════════════════════════════════════════
  function createSpleen(scene) {
    const root = new BABYLON.TransformNode('spleen', scene);

    const mat     = _pbr(scene, '#880810', { roughness: 0.70, sss: 0.38, emissive: 0.07 });
    const capsule = _pbr(scene, '#aa1018', { roughness: 0.35, sss: 0.12, alpha: 0.20 });
    const artMat  = _wire(scene, '#dd2233', 0.65);

    // Parenchyma
    const body = _sph('spleen_b', 0.128, scene, 1.48, 0.92, 0.72);
    body.rotation.x = Math.PI / 7; body.material = mat; body.parent = root;

    // Hilum notch
    var notch = BABYLON.MeshBuilder.CreateBox('spleen_notch', { width: 0.068, height: 0.018, depth: 0.040 }, scene);
    notch.position.set(0, 0.052, 0.016);
    notch.material = _wire(scene, '#660008', 0.55); notch.parent = root;

    // Capsule sheen
    const caps = _sph('spleen_caps', 0.130, scene, 1.50, 0.93, 0.73);
    caps.rotation.x = Math.PI / 7; caps.material = capsule; caps.parent = root;

    // Splenic artery (tortuous)
    var saPts = [
      _v3(0.010,0.010,0.008), _v3(-0.005,0.020,0.005), _v3(-0.020,0.008,0.004),
      _v3(-0.035,0.018,0.003), _v3(-0.050,0.006,0.002), _v3(-0.062,0.000,0.000),
    ];
    var sa = _vessel('splen_art', scene, artMat, saPts, 0.0070); sa.parent = root;

    // Splenic vein
    var svPts = [_v3(0.010,-0.005,0.005), _v3(-0.030,-0.008,0.003), _v3(-0.062,-0.010,0.000)];
    var sv = _vessel('splen_vein', scene, _wire(scene,'#2244aa',0.60), svPts, 0.0085); sv.parent = root;

    // Trabecular pillars
    for (var ti = 0; ti < 5; ti++) {
      var trab = _cyl('trab_' + ti, 0.006, 0.006, 0.10, 5, scene);
      trab.position.set((ti - 2) * 0.018, 0, 0); trab.rotation.x = Math.PI / (3 + ti * 0.4);
      trab.material = _wire(scene, '#aa1820', 0.25); trab.parent = root;
    }

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // BLADDER — detrusor wall, mucosal lining, ureters, rugal folds
  // ══════════════════════════════════════════════════════════════════
  function createBladder(scene) {
    const root = new BABYLON.TransformNode('bladder', scene);

    const wallMat  = _pbr(scene, '#5ab0e8', { roughness: 0.72, sss: 0.28, alpha: 0.82 });
    const innerMat = _pbr(scene, '#80d0ff', { roughness: 0.65, sss: 0.22, alpha: 0.45 });
    const urMat    = _pbr(scene, '#e8c860', { roughness: 0.60, sss: 0.18, alpha: 0.88 });

    // Detrusor wall
    const wall = _sph('bladder_w', 0.115, scene, 0.98, 1.08, 1.28);
    wall.material = wallMat; wall.parent = root;

    // Mucosal inner
    const inner = _sph('bladder_i', 0.095, scene, 0.94, 1.04, 1.22);
    inner.material = innerMat; inner.parent = root;

    // Rugal folds
    for (var bfi = 0; bfi < 6; bfi++) {
      var ang = (bfi / 6) * Math.PI * 2;
      var bfold = BABYLON.MeshBuilder.CreateBox('blad_fold_' + bfi, { width: 0.004, height: 0.08, depth: 0.02 }, scene);
      bfold.position.set(Math.cos(ang)*0.050, 0, Math.sin(ang)*0.062);
      bfold.rotation.y = ang; bfold.material = _wire(scene, '#3090c0', 0.30); bfold.parent = root;
    }

    // Ureteral openings (trigone)
    [-1, 1].forEach(function(s) {
      var urIn = _cyl('ur_in_' + s, 0.010, 0.010, 0.055, 6, scene);
      urIn.position.set(s*0.038, 0.038, -0.042); urIn.rotation.z = s*0.3; urIn.rotation.x = -0.5;
      urIn.material = urMat; urIn.parent = root;
    });

    // Urethra
    var urethra = _cyl('urethra', 0.014, 0.010, 0.070, 7, scene);
    urethra.position.set(0, -0.078, 0.028); urethra.rotation.x = 0.4;
    urethra.material = urMat; urethra.parent = root;

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // MUSCLE — bicep belly, striations, tendons, fascia
  // ══════════════════════════════════════════════════════════════════
  function createMuscle(scene) {
    const root = new BABYLON.TransformNode('muscle', scene);

    const fiberMat = _pbr(scene, '#cc1828', { roughness: 0.65, sss: 0.35, emissive: 0.06 });
    const tenMat   = _pbr(scene, '#f5e8c0', { roughness: 0.55, sss: 0.15 });
    const fasciaMat = _wire(scene, '#ddaa88', 0.18);

    // Two bicep heads
    const head1 = _sph('musc_h1', 0.105, scene, 0.68, 1.48, 0.62);
    head1.position.set(-0.022, 0.010, 0); head1.material = fiberMat; head1.parent = root;

    const head2 = _sph('musc_h2', 0.085, scene, 0.62, 1.30, 0.58);
    head2.position.set(0.020, 0.004, 0); head2.material = fiberMat; head2.parent = root;

    // Striations
    for (var mfi = 0; mfi < 8; mfi++) {
      var fib = _cyl('musc_fib_' + mfi, 0.014, 0.014, 0.160 + (mfi%3)*0.010, 5, scene);
      fib.position.set((mfi-3.5)*0.014, (mfi%2)*0.008-0.004, (mfi%3)*0.007);
      fib.rotation.z = (mfi%2===0 ? 0.06 : -0.06);
      fib.material = fiberMat; fib.parent = root;
    }

    // Tendons
    var tenTop = _cyl('musc_ten_t', 0.024, 0.028, 0.055, 8, scene);
    tenTop.position.set(0, 0.108, 0); tenTop.material = tenMat; tenTop.parent = root;

    var tenBot = _cyl('musc_ten_b', 0.020, 0.024, 0.050, 8, scene);
    tenBot.position.set(0, -0.108, 0); tenBot.material = tenMat; tenBot.parent = root;

    // Fascia envelope
    var fascia = _sph('musc_fascia', 0.185, scene, 0.68, 1.52, 0.64);
    fascia.material = fasciaMat; fascia.parent = root;

    return root;
  }

  // ══════════════════════════════════════════════════════════════════
  // SKIN — translucent PBR envelope with subsurface scatter
  // ══════════════════════════════════════════════════════════════════
  function createSkin(scene) {
    const root = new BABYLON.TransformNode('skin', scene);

    const skinMat = new BABYLON.PBRMaterial('skin_env_pbr', scene);
    skinMat.albedoColor = new BABYLON.Color3(0.94, 0.76, 0.68);
    skinMat.metallic    = 0.02;
    skinMat.roughness   = 0.55;
    skinMat.alpha       = 0.09;
    skinMat.backFaceCulling = false;
    skinMat.subSurface.isTranslucencyEnabled = true;
    skinMat.subSurface.translucencyIntensity = 0.55;
    skinMat.subSurface.tintColor = new BABYLON.Color3(0.95, 0.45, 0.35);
    skinMat.transparencyMode = BABYLON.PBRMaterial.PBRMATERIAL_ALPHABLEND;

    const env = _sph('skin_env', 0.82, scene, 1.02, 1.38, 0.68);
    env.material = skinMat; env.parent = root;

    // Faint capillary web
    var capMat = _wire(scene, '#dd4455', 0.12);
    for (var ci = 0; ci < 6; ci++) {
      var cAng = (ci / 6) * Math.PI * 2;
      var cr = 0.38;
      var cPts = [
        _v3(Math.cos(cAng)*cr, Math.sin(cAng)*cr*0.9, 0.28),
        _v3(Math.cos(cAng+0.5)*cr*0.8, Math.sin(cAng+0.5)*cr*0.85, 0.22),
      ];
      var cap = _vessel('cap_' + ci, scene, capMat, cPts, 0.0035); cap.parent = root;
    }

    return root;
  }

  // ── Factory ────────────────────────────────────────────────────
  function createOrgan(organId, scene) {
    switch (organId) {
      case 'brain':      return createBrain(scene);
      case 'heart':      return createHeart(scene);
      case 'lungs':      return createLungs(scene);
      case 'liver':      return createLiver(scene);
      case 'kidneys':    return createKidneys(scene);
      case 'stomach':    return createStomach(scene);
      case 'intestines': return createIntestines(scene);
      case 'pancreas':   return createPancreas(scene);
      case 'spleen':     return createSpleen(scene);
      case 'bladder':    return createBladder(scene);
      case 'muscle':     return createMuscle(scene);
      case 'skin':       return createSkin(scene);
      default: {
        const root = new BABYLON.TransformNode('org_fallback', scene);
        const m = BABYLON.MeshBuilder.CreateSphere('org_fallback_m', { diameter: 0.16, segments: 12 }, scene);
        m.material = _pbr(scene, '#888888');
        m.parent = root;
        return root;
      }
    }
  }

  return { createOrgan };
})();
