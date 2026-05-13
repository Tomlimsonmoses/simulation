/* ================================================================
   AnimationUtils.js — Shared easing functions and tween utilities
   Babylon.js compatible (no Three.js dependency).
   ================================================================ */

const AnimationUtils = (() => {

  function easeOutCubic(t)  { return 1 - Math.pow(1 - t, 3); }
  function easeInOutQuad(t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; }
  function smoothstep(t)    { return t * t * (3 - 2 * t); }
  function lerp(a, b, t)    { return a + (b - a) * t; }

  function tween(from, to, duration, onUpdate, onComplete, easing) {
    easing = easing || easeOutCubic;
    const startTime = Date.now();
    let raf;

    function tick() {
      const elapsed = Date.now() - startTime;
      const t = Math.min(elapsed / duration, 1);
      const eased = easing(t);

      if (typeof from === 'object' && from !== null && !Array.isArray(from)) {
        const current = {};
        Object.keys(from).forEach(k => { current[k] = lerp(from[k], to[k], eased); });
        onUpdate(current, t);
      } else {
        onUpdate(lerp(+from, +to, eased), t);
      }

      if (t < 1) {
        raf = requestAnimationFrame(tick);
      } else if (onComplete) {
        onComplete();
      }
    }

    tick();
    return { cancel() { cancelAnimationFrame(raf); } };
  }

  // Tweens an ArcRotateCamera target position.
  // camera: BABYLON.ArcRotateCamera
  // targetPos / targetLookAt: [x,y,z] array or {x,y,z} object
  function tweenCamera(camera, _unused, targetPos, targetLookAt, duration) {
    duration = duration || 700;

    const toVec = v => Array.isArray(v) ? { x: v[0], y: v[1], z: v[2] } : { x: v.x, y: v.y, z: v.z };
    const endLook = toVec(targetLookAt);
    const startLook = { x: camera.target.x, y: camera.target.y, z: camera.target.z };

    // For ArcRotateCamera we animate the target position; radius adjusts naturally.
    // If an endPos is provided, compute desired radius from it.
    let endRadius = camera.radius;
    if (targetPos) {
      const ep = toVec(targetPos);
      const dx = ep.x - endLook.x, dy = ep.y - endLook.y, dz = ep.z - endLook.z;
      endRadius = Math.sqrt(dx*dx + dy*dy + dz*dz) || camera.radius;
    }
    const startRadius = camera.radius;

    return tween(0, 1, duration, t => {
      camera.target.x = lerp(startLook.x, endLook.x, t);
      camera.target.y = lerp(startLook.y, endLook.y, t);
      camera.target.z = lerp(startLook.z, endLook.z, t);
      camera.radius   = lerp(startRadius, endRadius, t);
    }, null, easeInOutQuad);
  }

  return { easeOutCubic, easeInOutQuad, smoothstep, lerp, tween, tweenCamera };
})();
