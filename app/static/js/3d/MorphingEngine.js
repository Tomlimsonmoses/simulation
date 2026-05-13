/* ================================================================
   MorphingEngine.js — Age/gender body proportions and organ scaling
   Babylon.js version. Uses BABYLON.Vector3 scaling.
   ================================================================ */

const MorphingEngine = (() => {

  const AGE_PARAMS = [
    { age: 0,   height: 0.55, shoulder: 0.70, headSize: 1.40 },
    { age: 5,   height: 0.65, shoulder: 0.78, headSize: 1.25 },
    { age: 12,  height: 0.82, shoulder: 0.88, headSize: 1.10 },
    { age: 18,  height: 0.95, shoulder: 0.96, headSize: 1.02 },
    { age: 25,  height: 1.00, shoulder: 1.00, headSize: 1.00 },
    { age: 40,  height: 0.99, shoulder: 1.01, headSize: 1.00 },
    { age: 55,  height: 0.97, shoulder: 0.99, headSize: 1.00 },
    { age: 70,  height: 0.94, shoulder: 0.96, headSize: 1.00 },
    { age: 85,  height: 0.91, shoulder: 0.92, headSize: 1.00 },
    { age: 100, height: 0.87, shoulder: 0.88, headSize: 1.00 },
  ];

  const ORGAN_AGE = {
    brain:      { peak: 25, ratePerYear: -0.00050 },
    heart:      { peak: 30, ratePerYear:  0.00030, max: 1.15 },
    liver:      { peak: 25, ratePerYear: -0.00070 },
    kidneys:    { peak: 25, ratePerYear: -0.00080 },
    lungs:      { peak: 25, ratePerYear: -0.00040 },
    muscle:     { peak: 30, ratePerYear: -0.00100 },
    spleen:     { peak: 35, ratePerYear: -0.00030 },
    pancreas:   { peak: 30, ratePerYear: -0.00030 },
    stomach:    { peak: 30, ratePerYear: -0.00020 },
    intestines: { peak: 30, ratePerYear: -0.00020 },
    bladder:    { peak: 25, ratePerYear:  0.00030, max: 1.20 },
    skin:       { peak: 25, ratePerYear:  0.00000 },
  };

  function _interpolate(age) {
    const p = AGE_PARAMS;
    if (age <= p[0].age) return { ...p[0] };
    if (age >= p[p.length - 1].age) return { ...p[p.length - 1] };
    for (let i = 0; i < p.length - 1; i++) {
      if (age >= p[i].age && age <= p[i + 1].age) {
        const t = (age - p[i].age) / (p[i + 1].age - p[i].age);
        const s = t * t * (3 - 2 * t);
        return {
          height:   p[i].height   + (p[i + 1].height   - p[i].height)   * s,
          shoulder: p[i].shoulder + (p[i + 1].shoulder - p[i].shoulder) * s,
          headSize: p[i].headSize + (p[i + 1].headSize - p[i].headSize) * s,
        };
      }
    }
    return { ...p[p.length - 1] };
  }

  function _organAgeScale(id, age) {
    const cfg = ORGAN_AGE[id];
    if (!cfg) return 1;
    if (age < cfg.peak) return 0.70 + (age / cfg.peak) * 0.30;
    const scale = 1 + (age - cfg.peak) * cfg.ratePerYear;
    if (cfg.max) return Math.min(cfg.max, Math.max(0.60, scale));
    return Math.max(0.60, scale);
  }

  function applyAge(age, bodyRoot, organs, animated) {
    const params = _interpolate(age);

    if (bodyRoot) {
      if (animated && typeof AnimationUtils !== 'undefined') {
        const startY = bodyRoot.scaling.y, startX = bodyRoot.scaling.x;
        AnimationUtils.tween(
          { x: startX, y: startY },
          { x: params.shoulder, y: params.height },
          600,
          v => { bodyRoot.scaling.y = v.y; bodyRoot.scaling.x = v.x; }
        );
      } else {
        bodyRoot.scaling.y = params.height;
        bodyRoot.scaling.x = params.shoulder;
      }
    }

    if (organs) {
      Object.entries(organs).forEach(([id, organNode]) => {
        const scale = _organAgeScale(id, age);
        organNode.userData.ageScale = scale;
        const s = (organNode.userData.baseScale || 1) * scale;
        organNode.scaling = new BABYLON.Vector3(s, s, s);
      });
    }
  }

  function applyGender(gender, bodyRoot, organs, age) {
    if (!bodyRoot) return;
    age = age || 25;
    const params = _interpolate(age);
    const isFemale = gender === 'female';

    bodyRoot.scaling.y = params.height   * (isFemale ? 0.94 : 1.00);
    bodyRoot.scaling.x = params.shoulder * (isFemale ? 0.88 : 1.00);

    if (organs) {
      const femaleScale = {
        heart: 0.86, lungs: 0.84, liver: 0.88,
        kidneys: 0.87, muscle: 0.78, bladder: 1.08,
      };
      Object.entries(organs).forEach(([id, organNode]) => {
        const gs   = isFemale ? (femaleScale[id] || 1.0) : 1.0;
        const base = organNode.userData.baseScale || 1;
        const ageS = organNode.userData.ageScale  || 1;
        const s    = base * ageS * gs;
        organNode.scaling = new BABYLON.Vector3(s, s, s);
      });
    }
  }

  return { applyAge, applyGender };
})();
