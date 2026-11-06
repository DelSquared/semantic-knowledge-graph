/**
 * Semantic Knowledge Graph - Entry point (ES module)
 * Imports Three.js, OrbitControls, and the main application logic.
 */

import * as THREE from '/static/js/three.module.js';
import { OrbitControls } from '/static/js/OrbitControls.js';
import { init, loadGraph, rebuildGraph, toggleLabels, resetView, retryLoad } from '/static/js/main.js';

// Make THREE available globally for inline use in main.js (e.g. new THREE.Vector3)
globalThis.THREE = THREE;

// Re-export for main.js to use
window.OrbitControls = OrbitControls;

// Start the app
document.addEventListener('DOMContentLoaded', init);
