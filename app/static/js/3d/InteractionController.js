/* ================================================================
   InteractionController.js — Babylon.js picking, tooltips, camera presets
   ================================================================ */

const InteractionController = (() => {
  let _scene, _camera, _engine, _organs, _onSelect;
  let _tooltip = null;
  let _hoveredOrgan = null, _selectedOrgan = null;
  let _pointerObserver = null;

  // Camera presets expressed as { alpha, beta, radius, targetY }
  const PRESETS = {
    front:    { alpha: -Math.PI / 2,        beta: Math.PI / 2.2, radius: 3.5, targetY: 0.10 },
    back:     { alpha:  Math.PI / 2,        beta: Math.PI / 2.2, radius: 3.5, targetY: 0.10 },
    left:     { alpha:  0,                  beta: Math.PI / 2.2, radius: 3.5, targetY: 0.10 },
    right:    { alpha:  Math.PI,            beta: Math.PI / 2.2, radius: 3.5, targetY: 0.10 },
    top:      { alpha: -Math.PI / 2,        beta: 0.15,          radius: 4.2, targetY: 0.10 },
    overview: { alpha: -Math.PI / 2 + 0.24, beta: Math.PI / 2.2, radius: 3.5, targetY: 0.30 },
  };

  const HEALTH_LABELS = [
    { min: 0.90, label: 'Healthy',  color: '#00e676' },
    { min: 0.75, label: 'Good',     color: '#c6ef00' },
    { min: 0.50, label: 'Impaired', color: '#ffab40' },
    { min: 0.25, label: 'Severe',   color: '#ef5350' },
    { min: 0.15, label: 'Critical', color: '#ef1a1a' },
    { min: 0.08, label: 'Lethal',   color: '#9b0000' },
    { min: 0.00, label: 'Fatal',    color: '#3a0000' },
  ];

  function _healthInfo(h) {
    return HEALTH_LABELS.find(l => h >= l.min) || HEALTH_LABELS[HEALTH_LABELS.length - 1];
  }

  function init(scene, camera, engine, organs, onSelect) {
    _scene = scene; _camera = camera; _engine = engine;
    _organs = organs; _onSelect = onSelect;
    _buildTooltip();

    // ESC key — zoom back to overview
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') _deselect();
    });

    _pointerObserver = scene.onPointerObservable.add(info => {
      if (info.type === BABYLON.PointerEventTypes.POINTERPICK) {
        _handleClick(info.pickInfo);
      } else if (info.type === BABYLON.PointerEventTypes.POINTERMOVE) {
        _handleMove(info.event);
      } else if (info.type === BABYLON.PointerEventTypes.POINTERLEAVE) {
        _hideTooltip();
        _clearHover();
        scene.getEngine().getRenderingCanvas().style.cursor = 'default';
      }
    });
  }

  function _buildTooltip() {
    const container = document.getElementById('human-3d-container');
    if (!container) return;
    _tooltip = document.createElement('div');
    Object.assign(_tooltip.style, {
      position: 'absolute', pointerEvents: 'none', display: 'none',
      background: 'rgba(8,8,20,0.93)',
      border: '1px solid rgba(0,188,212,0.45)',
      borderRadius: '7px', padding: '9px 13px', color: '#ddd',
      fontSize: '12px', fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      zIndex: '200', boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
      minWidth: '145px', lineHeight: '1.55', transition: 'opacity 0.12s',
    });
    container.appendChild(_tooltip);
  }

  function _showTooltip(organNode, event) {
    if (!_tooltip) return;
    const health = (organNode.userData && organNode.userData.health != null) ? organNode.userData.health : 0.85;
    const pct    = Math.round(health * 100);
    const info   = _healthInfo(health);
    const label  = (organNode.userData && organNode.userData.label) || 'Organ';

    _tooltip.innerHTML =
      `<div style="font-weight:700;color:#00bcd4;font-size:13px;margin-bottom:5px">${label}</div>` +
      `<div style="display:flex;align-items:center;gap:7px;margin-bottom:3px">` +
        `<span style="width:9px;height:9px;border-radius:50%;background:${info.color};flex-shrink:0"></span>` +
        `<span style="color:${info.color};font-size:11px;font-weight:600">${info.label}</span>` +
        `<span style="margin-left:auto;color:rgba(255,255,255,0.55);font-size:11px">${pct}%</span>` +
      `</div>` +
      `<div style="background:rgba(255,255,255,0.04);border-radius:3px;height:4px;margin-bottom:6px;overflow:hidden">` +
        `<div style="height:100%;width:${pct}%;background:${info.color};border-radius:3px"></div>` +
      `</div>` +
      `<div style="font-size:10px;color:rgba(255,255,255,0.32)">Click for cellular detail →</div>`;

    const canvas = _engine.getRenderingCanvas();
    const rect   = canvas.getBoundingClientRect();
    let x = event.clientX - rect.left + 14;
    let y = event.clientY - rect.top  - 8;
    if (x + 165 > rect.width)  x = event.clientX - rect.left - 159;
    if (y + 100 > rect.height) y = event.clientY - rect.top  - 95;
    _tooltip.style.left    = x + 'px';
    _tooltip.style.top     = y + 'px';
    _tooltip.style.display = 'block';
  }

  function _hideTooltip() {
    if (_tooltip) _tooltip.style.display = 'none';
  }

  function _findOrganRoot(node) {
    // Fast path: sub-meshes carry a direct reference to their root TransformNode
    if (node && node._organRoot) return node._organRoot;
    // Fallback: walk up parent chain
    while (node) {
      if (node.userData && node.userData.id) return node;
      node = node.parent;
    }
    return null;
  }

  function _setHighlight(organNode, on) {
    const meshes = organNode.getChildMeshes ? organNode.getChildMeshes(false) : [];
    if (organNode.material) meshes.push(organNode);
    meshes.forEach(m => {
      if (!m.material) return;
      if (!m._baseEmissive) m._baseEmissive = m.material.emissiveColor ? m.material.emissiveColor.clone() : new BABYLON.Color3(0,0,0);
      const scaled = m._baseEmissive.scale(3.5);
      m.material.emissiveColor = on
        ? new BABYLON.Color3(Math.min(1, scaled.r), Math.min(1, scaled.g), Math.min(1, scaled.b))
        : m._baseEmissive.clone();
    });
  }

  function _setScale(organNode, factor) {
    const base = (organNode.userData && organNode.userData.baseScale) || 1;
    const ageS = (organNode.userData && organNode.userData.ageScale)  || 1;
    const s    = base * ageS * factor;
    organNode.scaling = new BABYLON.Vector3(s, s, s);
  }

  function _clearHover() {
    if (_hoveredOrgan && _hoveredOrgan !== _selectedOrgan) {
      _setScale(_hoveredOrgan, 1.0);
      _setHighlight(_hoveredOrgan, false);
    }
    _hoveredOrgan = null;
  }

  function _handleClick(pickInfo) {
    if (!pickInfo || !pickInfo.hit || !pickInfo.pickedMesh) {
      _deselect(); return;
    }
    const organNode = _findOrganRoot(pickInfo.pickedMesh);
    if (!organNode || !organNode.userData || !organNode.userData.id) {
      _deselect(); return;
    }
    if (_selectedOrgan === organNode) { _deselect(); return; }  // click selected organ again → exit zoom
    _selectOrgan(organNode);
    if (_onSelect) _onSelect(organNode.userData.id, organNode.userData.health);
  }

  function _handleMove(event) {
    const pick    = _scene.pick(_scene.pointerX, _scene.pointerY);
    const hovered = (pick && pick.hit && pick.pickedMesh)
      ? _findOrganRoot(pick.pickedMesh) : null;

    if (hovered !== _hoveredOrgan) {
      _clearHover();
      _hoveredOrgan = hovered;
      if (_hoveredOrgan && _hoveredOrgan !== _selectedOrgan) {
        _setScale(_hoveredOrgan, 1.12);
        _setHighlight(_hoveredOrgan, true);
      }
    }

    const canvas = _engine.getRenderingCanvas();
    if (hovered && hovered.userData && hovered.userData.id) {
      _showTooltip(hovered, event);
      canvas.style.cursor = 'pointer';
    } else {
      _hideTooltip();
      canvas.style.cursor = 'default';
    }
  }

  function _selectOrgan(organNode) {
    if (_selectedOrgan && _selectedOrgan !== organNode) {
      _setScale(_selectedOrgan, 1.0);
      _setHighlight(_selectedOrgan, false);
    }
    _selectedOrgan = organNode;
    _setScale(organNode, 1.12);
    _setHighlight(organNode, true);

    // ── Zoom camera into the organ inside the same scene ──────────────────
    const worldPos = organNode.getAbsolutePosition
      ? organNode.getAbsolutePosition()
      : organNode.position.clone();

    // Stop any running preset animations on the camera
    _scene.stopAnimation(_camera);

    AnimationUtils.tween(
      { x: _camera.target.x, y: _camera.target.y, z: _camera.target.z, r: _camera.radius },
      { x: worldPos.x, y: worldPos.y, z: worldPos.z, r: 0.78 },
      700,
      v => { _camera.target.x = v.x; _camera.target.y = v.y;
             _camera.target.z = v.z; _camera.radius   = v.r; },
      null, AnimationUtils.easeInOutQuad
    );

    // ── Fade body shell + dim other organs (X-ray reveal) ─────────────────
    if (typeof HumanBody3D !== 'undefined' && HumanBody3D.focusOrgan) {
      HumanBody3D.focusOrgan(organNode.userData.id);
    }

    // Show overview button
    const backBtn = document.getElementById('organ-back-btn');
    if (backBtn) backBtn.style.display = 'block';
  }

  function _deselect() {
    const wasSelected = !!_selectedOrgan;
    if (_selectedOrgan) {
      _setScale(_selectedOrgan, 1.0);
      _setHighlight(_selectedOrgan, false);
      _selectedOrgan = null;
    }
    _clearHover();
    if (!wasSelected && _camera && _camera.radius > 1.2) return; // already at overview

    // Animate back to overview
    _scene.stopAnimation(_camera);
    const p = PRESETS.overview;
    AnimationUtils.tween(
      { r: _camera.radius, tx: _camera.target.x, ty: _camera.target.y, tz: _camera.target.z,
        a: _camera.alpha, b: _camera.beta },
      { r: p.radius,  tx: 0, ty: p.targetY, tz: 0, a: p.alpha, b: p.beta },
      700,
      v => { _camera.radius = v.r; _camera.target.set(v.tx, v.ty, v.tz);
             _camera.alpha = v.a; _camera.beta = v.b; },
      null, AnimationUtils.easeInOutQuad
    );

    // Restore body shell + organ alphas
    if (typeof HumanBody3D !== 'undefined' && HumanBody3D.focusOverview) {
      HumanBody3D.focusOverview();
    }
    // Close cell detail drawer if open
    if (typeof CellView !== 'undefined' && CellView.close) CellView.close();

    // Hide overview button
    const backBtn = document.getElementById('organ-back-btn');
    if (backBtn) backBtn.style.display = 'none';
  }

  function goToPreset(name, animated) {
    const p = PRESETS[name];
    if (!p) return;
    if (animated === false) {
      _camera.alpha  = p.alpha;
      _camera.beta   = p.beta;
      _camera.radius = p.radius;
      _camera.target.y = p.targetY;
      return;
    }
    // Animate via BABYLON.Animation
    const fps = 30, frames = 30;
    const makeAnim = (prop, start, end) => {
      const a = new BABYLON.Animation('cam_' + prop, prop, fps,
        BABYLON.Animation.ANIMATIONTYPE_FLOAT,
        BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT);
      a.setKeys([{ frame: 0, value: start }, { frame: frames, value: end }]);
      const ease = new BABYLON.CubicEase();
      ease.setEasingMode(BABYLON.EasingFunction.EASINGMODE_EASEINOUT);
      a.setEasingFunction(ease);
      return a;
    };
    _camera.animations = [
      makeAnim('alpha',  _camera.alpha,  p.alpha),
      makeAnim('beta',   _camera.beta,   p.beta),
      makeAnim('radius', _camera.radius, p.radius),
    ];
    _scene.beginAnimation(_camera, 0, frames, false);
    AnimationUtils && AnimationUtils.tween(
      { ty: _camera.target.y }, { ty: p.targetY }, 450,
      v => { _camera.target.y = v.ty; }
    );
  }

  function resetSelection() { _deselect(); }

  function destroy() {
    if (_scene && _pointerObserver) _scene.onPointerObservable.remove(_pointerObserver);
    if (_tooltip && _tooltip.parentNode) _tooltip.parentNode.removeChild(_tooltip);
  }

  return { init, goToPreset, resetSelection, deselect: _deselect, destroy, PRESETS };
})();
