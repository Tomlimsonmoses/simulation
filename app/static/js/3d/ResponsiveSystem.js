/* ================================================================
   ResponsiveSystem.js — Disease progression and medication visual effects
   Babylon.js version. Works with StandardMaterial emissiveColor.
   ================================================================ */

const ResponsiveSystem = (() => {
  let _organs = {}, _active = false, _scene = null;
  let _pulseObserver = null;

  const ABSORPTION_ORDER = [
    'stomach', 'intestines', 'liver', 'kidneys',
    'heart', 'lungs', 'brain', 'muscle',
    'skin', 'spleen', 'pancreas', 'bladder',
  ];

  function _traverseMeshes(node, fn) {
    if (node instanceof BABYLON.AbstractMesh) fn(node);
    if (node.getChildMeshes) node.getChildMeshes(false).forEach(fn);
  }

  function init(scene, organs) {
    _scene   = scene;
    _organs  = organs;
    _active  = true;
    _pulseObserver = scene.onBeforeRenderObservable.add(_pulseLoop);
  }

  function onHealthUpdate(organId, health) {
    const organ = _organs[organId];
    if (!organ) return;
    organ.userData.pulseEnabled = health < 0.50;
    organ.userData.pulseFreq    = health < 0.20 ? 2.4 : health < 0.35 ? 1.5 : 0.75;
    organ.userData.pulseAmp     = health < 0.20 ? 0.40 : health < 0.35 ? 0.22 : 0.10;
    organ.userData.baseEmissiveScale = health < 0.20 ? 0.28 : health < 0.35 ? 0.14 : health < 0.50 ? 0.07 : 0.04;
  }

  function _pulseLoop() {
    if (!_active) return;
    const t = Date.now() * 0.001;
    Object.values(_organs).forEach(organNode => {
      if (!organNode.userData || !organNode.userData.pulseEnabled) return;
      const freq  = organNode.userData.pulseFreq  || 1;
      const amp   = organNode.userData.pulseAmp   || 0.1;
      const base  = organNode.userData.baseEmissiveScale || 0.05;
      const pulse = base + Math.abs(Math.sin(t * freq * Math.PI)) * amp;

      _traverseMeshes(organNode, mesh => {
        if (!mesh.material || !mesh.material.emissiveColor) return;
        if (!mesh._baseEmissivePulse) {
          mesh._baseEmissivePulse = mesh.material.diffuseColor
            ? mesh.material.diffuseColor.clone()
            : new BABYLON.Color3(0.5, 0.5, 0.5);
        }
        mesh.material.emissiveColor = mesh._baseEmissivePulse.scale(pulse);
      });
    });
  }

  function _fromHex(hex) {
    return BABYLON.Color3.FromHexString('#' + hex.toString(16).padStart(6, '0'));
  }

  function _flashOrgan(organNode, colorStart, colorEnd, duration) {
    const cStart = _fromHex(colorStart);
    const cEnd   = _fromHex(colorEnd);
    const saved  = new Map();

    _traverseMeshes(organNode, mesh => {
      if (!mesh.material || !mesh.material.emissiveColor) return;
      saved.set(mesh, mesh.material.emissiveColor.clone());
      mesh.material.emissiveColor = cStart.clone();
    });

    const start = Date.now();
    const fade  = () => {
      const t = Math.min((Date.now() - start) / duration, 1);
      _traverseMeshes(organNode, mesh => {
        if (!mesh.material || !saved.has(mesh)) return;
        const orig     = saved.get(mesh);
        const flashCol = t > 0.55 ? cEnd : cStart;
        mesh.material.emissiveColor = BABYLON.Color3.Lerp(flashCol, orig, t);
      });
      if (t < 1) requestAnimationFrame(fade);
      else _traverseMeshes(organNode, mesh => {
        if (mesh.material && saved.has(mesh)) {
          mesh.material.emissiveColor = saved.get(mesh).clone();
        }
      });
    };
    fade();
  }

  function playMedicationAbsorption(organIds) {
    const ids      = (organIds && organIds.length) ? organIds : Object.keys(_organs);
    const ordered  = ABSORPTION_ORDER.filter(id => ids.includes(id));
    const rest     = ids.filter(id => !ordered.includes(id));
    const sequence = [...ordered, ...rest];
    sequence.forEach((id, i) => {
      setTimeout(() => {
        const organ = _organs[id];
        if (organ) _flashOrgan(organ, 0x00bcd4, 0x0066ff, 700);
      }, i * 160);
    });
  }

  function playDiseaseWave(organHealthMap) {
    const sorted = Object.entries(organHealthMap)
      .filter(([, h]) => h < 0.75)
      .sort((a, b) => a[1] - b[1]);
    sorted.forEach(([id, health], i) => {
      setTimeout(() => {
        const organ = _organs[id];
        if (!organ) return;
        const c = health < 0.25 ? 0xff1100 : health < 0.50 ? 0xff6600 : 0xffaa00;
        _flashOrgan(organ, c, c, 900);
      }, i * 180);
    });
  }

  function destroy() {
    _active = false;
    if (_scene && _pulseObserver) {
      _scene.onBeforeRenderObservable.remove(_pulseObserver);
    }
  }

  return { init, onHealthUpdate, playMedicationAbsorption, playDiseaseWave, destroy };
})();
