/**
 * Semantic Knowledge Graph - 3D Visualization (ES module)
 * Uses Three.js with custom force-directed layout.
 *
 * Dependencies:
 *   three.module.js       - Three.js ES module build
 *   OrbitControls.js      - OrbitControls from three.js examples/jsm/controls/
 */

import * as THREE from '/static/js/three.module.js';
import { OrbitControls } from '/static/js/OrbitControls.js';

// ── State ──────────────────────────────────────────────────────────────────
const state = {
    nodes: [],
    edges: [],
    nodeMeshes: {},
    labelMeshes: {},
    lines: [],
    labelsVisible: true,
    raycaster: new THREE.Raycaster(),
    mouse: new THREE.Vector2(),
    tooltip: document.getElementById('tooltip'),
    container: document.getElementById('canvas-container'),
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    animating: true,
    stats: document.getElementById('stats'),
};

// ── Helpers ─────────────────────────────────────────────────────────────────
function showError(message) {
    console.error('SemanticGraph:', message);
    const overlay = document.getElementById('error-overlay');
    const msgEl = document.getElementById('error-message');
    msgEl.textContent = message;
    overlay.classList.remove('hidden');
}

function hideError() {
    document.getElementById('error-overlay').classList.add('hidden');
}

function log(msg) {
    console.log('SemanticGraph:', msg);
}

// ── Initialization ─────────────────────────────────────────────────────────
export async function init() {
    const overlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');

    // Show loading (overlay starts visible in HTML)
    overlay.classList.remove('hidden');

    try {
        // Create scene
        state.scene = new THREE.Scene();
        state.scene.background = new THREE.Color(0x0a0a0f);

        // Create camera
        state.camera = new THREE.PerspectiveCamera(
            60, window.innerWidth / window.innerHeight, 0.1, 1000
        );
        state.camera.position.set(0, 40, 80);

        // Create renderer
        state.renderer = new THREE.WebGLRenderer({ antialias: true });
        state.renderer.setSize(window.innerWidth, window.innerHeight);
        state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        state.renderer.shadowMap.enabled = false;
        state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        state.renderer.toneMappingExposure = 1.0;
        state.container.appendChild(state.renderer.domElement);
        log('Canvas created: ' + state.renderer.domElement.width + 'x' + state.renderer.domElement.height);

        // Create controls
        state.controls = new OrbitControls(state.camera, state.renderer.domElement);
        state.controls.enableDamping = true;
        state.controls.dampingFactor = 0.05;
        state.controls.minDistance = 10;
        state.controls.maxDistance = 500;
        state.controls.target.set(0, 0, 0);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404060, 0.6);
        state.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(50, 100, 50);
        state.scene.add(dirLight);

        const pointLight = new THREE.PointLight(0x4CAF50, 0.5, 200);
        pointLight.position.set(0, 50, 0);
        state.scene.add(pointLight);



        // Event listeners
        window.addEventListener('resize', onWindowResize);
        state.renderer.domElement.addEventListener('mousemove', onMouseMove);
        state.renderer.domElement.addEventListener('click', onClick);

        // Middle-click pan
        let middleMouse = false;
        state.renderer.domElement.addEventListener('mousedown', (e) => {
            if (e.button === 1) { middleMouse = true; e.preventDefault(); }
        });
        window.addEventListener('mouseup', () => { middleMouse = false; });
        state.renderer.domElement.addEventListener('mousemove', (e) => {
            if (!middleMouse) return;
            const dx = e.movementX;
            const dy = e.movementY;
            const pan = new THREE.Vector3(-dx, dy, 0);
            pan.applyQuaternion(state.camera.quaternion);
            pan.multiplyScalar(0.5);
            state.controls.target.add(pan);
            state.camera.position.add(pan);
            state.controls.update();
        });

        // UI buttons
        document.getElementById('rebuild-btn').addEventListener('click', rebuildGraph);
        document.getElementById('toggle-labels').addEventListener('click', toggleLabels);
        document.getElementById('reset-view').addEventListener('click', resetView);
        document.getElementById('error-retry').addEventListener('click', retryLoad);

        // Load graph
        await loadGraph();

        // Start animation loop
        animate();
    } catch (err) {
        showError(err.message || String(err));
        loadingText.textContent = 'Error: ' + (err.message || String(err));
    }
}

// ── Graph Loading ──────────────────────────────────────────────────────────
export async function loadGraph() {
    const overlay = document.getElementById('loading-overlay');
    try {
        log('Fetching graph data from /api/graph...');
        const resp = await fetch('/api/graph');
        if (!resp.ok) {
            throw new Error('Server returned ' + resp.status + ': ' + resp.statusText);
        }
        const data = await resp.json();

        log('Received ' + data.nodes.length + ' nodes, ' + data.edges.length + ' edges');

        state.nodes = data.nodes;
        state.edges = data.edges;

        // Clear previous
        clearGraph();

        // Build 3D objects
        buildGraph();

        // Update stats
        updateStats();

        log('Graph loaded successfully');
    } catch (err) {
        log('Graph load failed: ' + err.message);
        throw err;
    } finally {
        // Always hide loading overlay, even on error
        if (overlay) overlay.classList.add('hidden');
    }
}

function clearGraph() {
    for (const mesh of Object.values(state.nodeMeshes)) {
        state.scene.remove(mesh);
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
            if (Array.isArray(mesh.material)) {
                mesh.material.forEach(m => m.dispose());
            } else {
                mesh.material.dispose();
            }
        }
    }
    for (const line of state.lines) {
        state.scene.remove(line);
        if (line.geometry) line.geometry.dispose();
        if (line.material) {
            if (Array.isArray(line.material)) {
                line.material.forEach(m => m.dispose());
            } else {
                line.material.dispose();
            }
        }
    }
    for (const mesh of Object.values(state.labelMeshes)) {
        state.scene.remove(mesh);
        if (mesh.material && mesh.material.map) mesh.material.map.dispose();
        if (mesh.material) mesh.material.dispose();
    }
    state.nodeMeshes = {};
    state.labelMeshes = {};
}

function buildGraph() {
    const nodes = state.nodes;
    const edges = state.edges;
    const n = nodes.length;
    log('Building ' + n + ' nodes in 3D scene...');

    // Qualitative color palette (Set1 from matplotlib)
    const CLUSTER_COLORS = [
        new THREE.Color(0xE41A1C), // red
        new THREE.Color(0x377EB8), // blue
        new THREE.Color(0x4DAF4A), // green
        new THREE.Color(0x984EA3), // purple
        new THREE.Color(0xFF7F00), // orange
        new THREE.Color(0xFFFF33), // yellow
    ];

    // Add nodes
    for (let i = 0; i < n; i++) {
        const node = nodes[i];
        const pos = node.position;

        // Node size -- small and consistent
        const radius = 2.5;

        // Node color based on cluster ID
        const clusterId = node.cluster ?? 0;
        const nodeColor = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];

        // Glow effect behind node (slightly brighter version of cluster color)
        const glowColor = nodeColor.clone();
        glowColor.lerp(new THREE.Color(0xffffff), 0.3);

        // Glow effect behind node
        const glowGeometry = new THREE.SphereGeometry(radius * 1.6, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: glowColor,
            transparent: true,
            opacity: 0.15,
            side: THREE.BackSide,
        });
        const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
        glowMesh.position.set(pos[0], pos[1], pos[2]);
        state.scene.add(glowMesh);

        // Main node sphere
        const geometry = new THREE.SphereGeometry(radius, 16, 16);
        const material = new THREE.MeshPhongMaterial({
            color: nodeColor,
            emissive: nodeColor,
            emissiveIntensity: 0.25,
            flatShading: false,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(pos[0], pos[1], pos[2]);
        mesh.castShadow = false;
        mesh.userData = { nodeId: node.id, node: node };
        state.scene.add(mesh);
        state.nodeMeshes[node.id] = mesh;

        // Label sprite
        if (state.labelsVisible) {
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 80;
            const ctx = canvas.getContext('2d');
            ctx.font = 'bold 28px Segoe UI, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#ffffff';

            const displayId = node.id.length > 20 ? node.id.substring(0, 17) + '...' : node.id;
            ctx.fillText(displayId, 256, 40);

            const tex = new THREE.CanvasTexture(canvas);
            tex.needsUpdate = true;
            const labelMaterial = new THREE.SpriteMaterial({
                map: tex,
                transparent: true,
                depthTest: false,
            });
            const label = new THREE.Sprite(labelMaterial);
            label.position.set(pos[0], pos[1] + radius + 5, pos[2]);
            label.scale.set(14, 3.5, 1);
            label.userData = { nodeId: node.id };
            state.scene.add(label);
            state.labelMeshes[node.id] = label;
        }
    }

    // Static positions - freeze after initial render
    state.animating = false;
}

// ── Interaction ────────────────────────────────────────────────────────────
function onMouseMove(event) {
    // Calculate mouse position in normalized device coordinates (-1 to 1)
    state.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    state.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    // Update raycaster
    state.raycaster.setFromCamera(state.mouse, state.camera);

    // Check node intersections
    const meshes = Object.values(state.nodeMeshes);
    const intersects = state.raycaster.intersectObjects(meshes);

    if (intersects.length > 0) {
        const hit = intersects[0].object;
        const node = state.nodes.find(n => n.id === hit.userData.nodeId);
        if (node) {
            state.tooltip.style.display = 'block';
            const clusterId = node.cluster ?? 0;
            const clusterColors = ['red','blue','green','purple','orange','yellow'];
            const clusterName = clusterId < clusterColors.length
                ? clusterColors[clusterId]
                : 'cluster ' + clusterId;
            state.tooltip.innerHTML = '<div class="node-id">' + node.id + '</div>' +
                '<div class="node-content">' + node.content_preview + '</div>' +
                '<div class="node-similarity">Cluster: ' + clusterName + '</div>';
            state.tooltip.style.left = (event.clientX + 14) + 'px';
            state.tooltip.style.top = (event.clientY - 10) + 'px';
            document.body.style.cursor = 'pointer';
            return;
        }
    }

    // Nothing hit
    state.tooltip.style.display = 'none';
    document.body.style.cursor = 'default';
}

function onClick(event) {
    state.raycaster.setFromCamera(state.mouse, state.camera);

    const meshes = Object.values(state.nodeMeshes);
    const intersects = state.raycaster.intersectObjects(meshes);

    if (intersects.length > 0) {
        const hit = intersects[0].object;
        const node = state.nodes.find(n => n.id === hit.userData.nodeId);
        if (node) {
            state.controls.target.set(node.position[0], node.position[1], node.position[2]);
            const offset = new THREE.Vector3(0, 10, 25);
            offset.applyQuaternion(state.camera.quaternion);
            state.camera.position.set(
                node.position[0] + offset.x,
                node.position[1] + offset.y,
                node.position[2] + offset.z
            );
            state.animating = false;
            return;
        }
    }
}

// ── UI Controls ────────────────────────────────────────────────────────────
export function toggleLabels() {
    state.labelsVisible = !state.labelsVisible;
    for (const mesh of Object.values(state.labelMeshes)) {
        mesh.visible = state.labelsVisible;
    }
    log('Labels: ' + (state.labelsVisible ? 'visible' : 'hidden'));
}

export function resetView() {
    state.camera.position.set(0, 40, 80);
    state.controls.target.set(0, 0, 0);
    state.controls.update();
}

export async function rebuildGraph() {
    const overlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');

    overlay.classList.remove('hidden');
    hideError();
    loadingText.textContent = 'Rebuilding graph...';

    const btn = document.getElementById('rebuild-btn');
    btn.disabled = true;
    btn.textContent = 'Rebuilding...';

    try {
        await loadGraph();
        loadingText.textContent = 'Building semantic graph from notes...';
        log('Rebuild complete');
    } catch (err) {
        console.error('Rebuild failed:', err);
        loadingText.textContent = 'Rebuild failed: ' + err.message;
    } finally {
        // Always hide loading overlay, even on error
        if (overlay) overlay.classList.add('hidden');
    }

    btn.disabled = false;
    btn.textContent = 'Rebuild Graph';
}

export async function retryLoad() {
    hideError();
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.remove('hidden');
    document.getElementById('loading-text').textContent = 'Loading graph...';

    try {
        await loadGraph();
    } catch (err) {
        showError('Failed to load graph: ' + err.message);
    } finally {
        // Always hide loading overlay, even on error
        if (overlay) overlay.classList.add('hidden');
    }
}

function updateStats() {
    const nNodes = state.nodes.length;
    const nEdges = state.edges.length;
    state.stats.textContent = nNodes + ' nodes, ' + nEdges + ' edges';
}

// ── Animation ──────────────────────────────────────────────────────────────
function animate() {
    requestAnimationFrame(animate);

    if (state.animating && state.scene && state.camera) {
        const time = Date.now() * 0.001;
        for (const mesh of Object.values(state.nodeMeshes)) {
            mesh.rotation.y = time * 0.1;
            mesh.rotation.x = Math.sin(time * 0.05) * 0.1;
        }
    }

    if (state.controls && state.camera) {
        state.controls.update();
    }
    if (state.renderer && state.scene) {
        state.renderer.render(state.scene, state.camera);
    }
}

function onWindowResize() {
    if (!state.camera || !state.renderer) return;
    state.camera.aspect = window.innerWidth / window.innerHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(window.innerWidth, window.innerHeight);
}
