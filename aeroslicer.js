// Aeroslicer - 2D CFD Simulation
// Incompressible Navier-Stokes solver with immersed boundary method
// WebGL2 renderer with Canvas 2D fallback

// ── WebGL2 Shader Sources ──────────────────────────────────────────────────

const VERTEX_SHADER_SRC = `#version 300 es
// Full-screen triangle from gl_VertexID — no vertex buffer needed
void main() {
    // Generates a triangle covering [-1,1]x[-1,1]
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

// Fragment shader template — HAS_FLOAT_LINEAR is injected at compile time
const FRAGMENT_SHADER_TEMPLATE = `#version 300 es
precision highp float;

uniform vec2 u_resolution;
uniform vec2 u_domainMin;
uniform vec2 u_domainMax;
uniform vec2 u_viewMin;
uniform vec2 u_viewMax;
uniform vec2 u_texelSize;   // 1/Nx, 1/Ny
uniform float u_fieldMin;
uniform float u_fieldMax;
uniform int u_vizMode;      // 0 = velocity, 1 = pressure

uniform sampler2D u_texU;
uniform sampler2D u_texV;
uniform sampler2D u_texP;
uniform sampler2D u_texMask;
uniform sampler2D u_texViridis;
uniform sampler2D u_texInvMapX;
uniform sampler2D u_texInvMapY;

out vec4 fragColor;

// Bilinear sampling — compile-time branch eliminates per-pixel cost
float sampleBilinear(sampler2D tex, vec2 uv) {
#ifdef HAS_FLOAT_LINEAR
    return texture(tex, uv).r;
#else
    vec2 texSize = 1.0 / u_texelSize;
    vec2 texelCoord = uv * texSize - 0.5;
    vec2 f = fract(texelCoord);
    vec2 base = (floor(texelCoord) + 0.5) * u_texelSize;
    float tl = texture(tex, base).r;
    float tr = texture(tex, base + vec2(u_texelSize.x, 0.0)).r;
    float bl = texture(tex, base + vec2(0.0, u_texelSize.y)).r;
    float br = texture(tex, base + u_texelSize).r;
    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);
#endif
}

void main() {
    // Screen UV: gl_FragCoord / resolution
    vec2 screenUV = gl_FragCoord.xy / u_resolution;

    // Map to world coordinates
    // WebGL Y=0 is bottom, which matches simulation Y-min at bottom
    vec2 world = mix(u_viewMin, u_viewMax, screenUV);

    // World to texture UV via inverse coordinate maps (non-uniform grid)
    vec2 worldNorm = (world - u_domainMin) / (u_domainMax - u_domainMin);
    float gridU = texture(u_texInvMapX, vec2(worldNorm.x, 0.5)).r;
    float gridV = texture(u_texInvMapY, vec2(worldNorm.y, 0.5)).r;
    vec2 texUV = vec2(gridU, gridV);

    // Outside domain check
    if (texUV.x < 0.0 || texUV.x > 1.0 || texUV.y < 0.0 || texUV.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Sample mask
    float maskVal = sampleBilinear(u_texMask, texUV);
    if (maskVal < 0.5) {
        // Solid body — dark gray
        fragColor = vec4(0.118, 0.118, 0.118, 1.0); // ~30/255
        return;
    }

    // Sample field value
    float val;
    if (u_vizMode == 0) {
        // Velocity magnitude
        float uVal = sampleBilinear(u_texU, texUV);
        float vVal = sampleBilinear(u_texV, texUV);
        val = sqrt(uVal * uVal + vVal * vVal);
    } else {
        // Pressure
        val = sampleBilinear(u_texP, texUV);
    }

    // Normalize to [0,1]
    float range = u_fieldMax - u_fieldMin;
    float t = (range > 0.0) ? clamp((val - u_fieldMin) / range, 0.0, 1.0) : 0.5;

    // Look up viridis colormap (1D texture)
    vec3 color = texture(u_texViridis, vec2(t, 0.5)).rgb;
    fragColor = vec4(color, 1.0);
}
`;

// ── AeroSlicer Class ───────────────────────────────────────────────────────

class AeroSlicer {
    constructor() {
        // Physical constants
        this.L = 1.0;  // Reference length
        this.U = 1.0;  // Free stream velocity

        // Domain setup
        this.xMin = -10.0;
        this.xMax = 30.0;
        this.yMin = -10.0;
        this.yMax = 10.0;

        // View window (base values, adjusted for aspect ratio in resizeCanvas)
        this._viewCenterX = 2.5;   // Focus center x (body at origin, show some wake)
        this._viewCenterY = 0.0;   // Focus center y
        this._viewHalfH = 3.5;     // Base half-height of view
        this.viewXMin = -2.0;
        this.viewXMax = 10.0;
        this.viewYMin = -3.0;
        this.viewYMax = 3.0;

        // Grid parameters (fine grid default)
        this.Ny = 200;
        this.Nx = 400;

        // Simulation parameters
        this.Re = 10000;
        this.nu = 1.0 / this.Re;
        this.dt = 0.01;
        this.time = 0.0;

        // Body parameters
        this.bodyType = 'circle';
        this.angle = 0.0; // degrees
        this.customPolygon = null; // Array of {x, y} in body-frame coords
        this._uploadState = 'idle'; // idle | selecting | previewing
        this.addGround = false;

        // Visualization
        this.vizMode = 'velocity';

        // Force tracking
        this.liftHistory = [];
        this.dragHistory = [];
        this.flowThroughTime = 40.0; // Lx/U
        this.maxHistoryLength = 0;

        // Precompute viridis LUT (256 entries)
        this.viridisLUT = this._buildViridisLUT(256);

        // Initialize
        this.initializeGrid();
        this.initializeFields();
        this.generateBody();
        this.setupCanvas();
        this.setupControls();
        this.setupUploadUI();

        const canvasContainer = document.getElementById('canvas-container');

        // Mesh overlay canvas (transparent, on top of render canvas)
        this.meshCanvas = document.createElement('canvas');
        this.meshCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:5';
        canvasContainer.appendChild(this.meshCanvas);
        this._meshDirty = true;

        // Mobile panel toggle
        const panelToggle = document.getElementById('panel-toggle');
        const controlPanel = document.getElementById('control-panel');
        if (panelToggle) {
            panelToggle.addEventListener('click', () => {
                controlPanel.classList.toggle('collapsed');
            });
        }

        // Dismiss loading overlay
        const loader = document.getElementById('loading-overlay');
        if (loader) {
            loader.style.opacity = '0';
            setTimeout(() => loader.remove(), 300);
        }

        // Deferred resize to catch mobile layout settling after loading overlay removal
        setTimeout(() => this.resizeCanvas(), 50);

        // Start simulation
        this.running = true;
        this.lastTime = performance.now();
        this.frameCount = 0;
        this._boundAnimate = () => this.animate(); // bind once, reuse
        this.animate();
    }

    // Build a precomputed viridis lookup table with N entries
    _buildViridisLUT(N) {
        const c0 = [68, 1, 84];
        const c1 = [59, 82, 139];
        const c2 = [33, 145, 140];
        const c3 = [94, 201, 98];
        const c4 = [253, 231, 37];

        // Store as flat Uint8Array: [r0, g0, b0, r1, g1, b1, ...]
        const lut = new Uint8Array(N * 3);

        for (let i = 0; i < N; i++) {
            const t = i / (N - 1);
            let r, g, b;
            if (t < 0.25) {
                const s = t / 0.25;
                r = c0[0] + s * (c1[0] - c0[0]);
                g = c0[1] + s * (c1[1] - c0[1]);
                b = c0[2] + s * (c1[2] - c0[2]);
            } else if (t < 0.5) {
                const s = (t - 0.25) / 0.25;
                r = c1[0] + s * (c2[0] - c1[0]);
                g = c1[1] + s * (c2[1] - c1[1]);
                b = c1[2] + s * (c2[2] - c1[2]);
            } else if (t < 0.75) {
                const s = (t - 0.5) / 0.25;
                r = c2[0] + s * (c3[0] - c2[0]);
                g = c2[1] + s * (c3[1] - c2[1]);
                b = c2[2] + s * (c3[2] - c2[2]);
            } else {
                const s = (t - 0.75) / 0.25;
                r = c3[0] + s * (c4[0] - c3[0]);
                g = c3[1] + s * (c4[1] - c3[1]);
                b = c3[2] + s * (c4[2] - c3[2]);
            }
            lut[i * 3] = Math.round(r);
            lut[i * 3 + 1] = Math.round(g);
            lut[i * 3 + 2] = Math.round(b);
        }
        return lut;
    }

    initializeGrid() {
        this.Ny = 200;
        this.Nx = 400;

        const Nx = this.Nx;
        const Ny = this.Ny;
        const beta = 1.8; // Stretching parameter (higher = more clustering)

        // ── Stretched coordinate generation (piecewise tanh) ──

        // X-direction: split at x=0, proportional allocation
        // Left [-10, 0] = 10 units, Right [0, 30] = 30 units, total = 40
        const NxLeft = Math.round(Nx * 10 / 40); // ~200 points for [-10, 0]
        // Right segment gets remaining points; x=0 is shared at index NxLeft
        const NxRight = Nx - NxLeft; // ~600 points for [0, 30]

        this.x = new Float32Array(Nx);
        // Left segment: cluster toward x=0 (right end)
        const leftCoords = this._tanhStretch(NxLeft + 1, this.xMin, 0.0, beta);
        for (let i = 0; i <= NxLeft; i++) this.x[i] = leftCoords[i];
        // Right segment: cluster toward x=0 (left end)
        const rightCoords = this._tanhStretch(NxRight, 0.0, this.xMax, beta);
        for (let i = 0; i < NxRight; i++) this.x[NxLeft + i] = rightCoords[i];

        // Y-direction: symmetric, split at y=0
        const NyBottom = Ny >> 1; // 200 points for [-10, 0]
        const NyTop = Ny - NyBottom; // 200 points for [0, 10]

        this.y = new Float32Array(Ny);
        const bottomCoords = this._tanhStretch(NyBottom + 1, this.yMin, 0.0, beta);
        for (let j = 0; j <= NyBottom; j++) this.y[j] = bottomCoords[j];
        const topCoords = this._tanhStretch(NyTop, 0.0, this.yMax, beta);
        for (let j = 0; j < NyTop; j++) this.y[NyBottom + j] = topCoords[j];

        // ── Cell spacings ──
        this.hx = new Float32Array(Nx - 1);
        this.hy = new Float32Array(Ny - 1);
        for (let i = 0; i < Nx - 1; i++) this.hx[i] = this.x[i + 1] - this.x[i];
        for (let j = 0; j < Ny - 1; j++) this.hy[j] = this.y[j + 1] - this.y[j];

        // Reciprocal spacings (for upwind convection)
        this.invHx = new Float32Array(Nx - 1);
        this.invHy = new Float32Array(Ny - 1);
        for (let i = 0; i < Nx - 1; i++) this.invHx[i] = 1.0 / this.hx[i];
        for (let j = 0; j < Ny - 1; j++) this.invHy[j] = 1.0 / this.hy[j];

        // Minimum spacings for CFL
        this.hxMin = this.hx[0];
        for (let i = 1; i < Nx - 1; i++) if (this.hx[i] < this.hxMin) this.hxMin = this.hx[i];
        this.hyMin = this.hy[0];
        for (let j = 1; j < Ny - 1; j++) if (this.hy[j] < this.hyMin) this.hyMin = this.hy[j];

        // Backward compat: dx/dy = minimum spacing
        this.dx = this.hxMin;
        this.dy = this.hyMin;

        // ── Precomputed Laplacian coefficients (for non-uniform 2nd derivative) ──
        // d²f/dx² at i ≈ aW[i]*f[i-1] + aE[i]*f[i+1] - (aW[i]+aE[i])*f[i]
        this.lapAW = new Float32Array(Nx);
        this.lapAE = new Float32Array(Nx);
        this.lapDenomX = new Float32Array(Nx);
        this.invHxC = new Float32Array(Nx); // 1/(hx[i-1]+hx[i]) for central diff
        for (let i = 1; i < Nx - 1; i++) {
            const hm = this.hx[i - 1];
            const hp = this.hx[i];
            const sum = hm + hp;
            this.lapAW[i] = 2.0 / (hm * sum);
            this.lapAE[i] = 2.0 / (hp * sum);
            this.lapDenomX[i] = this.lapAW[i] + this.lapAE[i];
            this.invHxC[i] = 1.0 / sum;
        }

        this.lapAS = new Float32Array(Ny);
        this.lapAN = new Float32Array(Ny);
        this.lapDenomY = new Float32Array(Ny);
        this.invHyC = new Float32Array(Ny);
        for (let j = 1; j < Ny - 1; j++) {
            const hm = this.hy[j - 1];
            const hp = this.hy[j];
            const sum = hm + hp;
            this.lapAS[j] = 2.0 / (hm * sum);
            this.lapAN[j] = 2.0 / (hp * sum);
            this.lapDenomY[j] = this.lapAS[j] + this.lapAN[j];
            this.invHyC[j] = 1.0 / sum;
        }

        // ── Inverse maps for renderer (world-norm → grid-UV) ──
        this.invMapX = this._buildInverseMap(this.x, Nx, this.xMin, this.xMax, 1024);
        this.invMapY = this._buildInverseMap(this.y, Ny, this.yMin, this.yMax, 1024);

        this._updateDt();
        this.stepsPerFrame = 3;
        this.maxHistoryLength = (this.flowThroughTime / this.dt) | 0;
        this.offCanvas = null;
    }

    // Generate N points from xStart to xEnd with tanh clustering toward xEnd
    _tanhStretch(N, xStart, xEnd, beta) {
        const coords = new Float32Array(N);
        if (N === 1) { coords[0] = xStart; return coords; }
        const tanhBeta = Math.tanh(beta);
        for (let k = 0; k < N; k++) {
            const xi = k / (N - 1); // 0..1
            const stretched = Math.tanh(beta * (2.0 * xi - 1.0)) / tanhBeta; // -1..1
            coords[k] = 0.5 * (xStart + xEnd) + 0.5 * (xEnd - xStart) * stretched;
        }
        // Enforce exact endpoints
        coords[0] = xStart;
        coords[N - 1] = xEnd;
        return coords;
    }

    // Build inverse map: given monotonic coords[0..N-1], create a lookup table
    // that maps normalized world [0,1] → grid UV [0,1]
    _buildInverseMap(coords, N, xMin, xMax, mapSize) {
        const map = new Float32Array(mapSize);
        const invMS = 1.0 / (mapSize - 1);
        const domainExtent = xMax - xMin;
        let si = 0; // search index, marches forward
        for (let k = 0; k < mapSize; k++) {
            const world = xMin + (k * invMS) * domainExtent;
            // Linear search from last position (O(N+mapSize) total)
            while (si < N - 2 && coords[si + 1] < world) si++;
            const frac = (coords[si + 1] - coords[si]) > 0
                ? (world - coords[si]) / (coords[si + 1] - coords[si])
                : 0.0;
            const gridUV = (si + frac) / (N - 1);
            map[k] = gridUV < 0.0 ? 0.0 : (gridUV > 1.0 ? 1.0 : gridUV);
        }
        return map;
    }

    _updateDt() {
        // Stable Fluids: unconditionally stable, use fixed dt
        this.dt = 0.05;
    }

    initializeFields() {
        const size = this.Nx * this.Ny;
        const Nx = this.Nx;
        const Ny = this.Ny;

        // Velocity components
        this.u = new Float32Array(size);
        this.v = new Float32Array(size);

        // Scratch buffers for semi-Lagrangian advection (previous velocity)
        this.u_prev = new Float32Array(size);
        this.v_prev = new Float32Array(size);

        // Vorticity buffer for confinement
        this._omega = new Float32Array(size);

        // Pressure (kept across steps as initial guess for Poisson solver)
        this.p = new Float32Array(size);

        // Solid mask (1 = fluid, 0 = solid)
        this.mask = new Float32Array(size);

        // Divergence of intermediate velocity
        this.div = new Float32Array(size);

        // WALE eddy viscosity per cell
        this.nu_t = new Float32Array(size);
        this.maxNuT = 0.0;

        // Initialize with free stream + symmetry-breaking perturbation
        const U = this.U;
        const L = this.L;
        const yArr = this.y;
        const TWO_PI = 2.0 * Math.PI;
        for (let j = 0; j < Ny; j++) {
            const jNx = j * Nx;
            const yj = yArr[j];
            for (let i = 0; i < Nx; i++) {
                const idx = jNx + i;
                this.u[idx] = U;
                this.v[idx] = 0.01 * U * Math.sin(TWO_PI * yj / L);
                this.p[idx] = 0.0;
                this.mask[idx] = 1.0;
            }
        }
    }

    generateBody() {
        const Nx = this.Nx;
        const Ny = this.Ny;
        const mask = this.mask;
        const xArr = this.x;
        const yArr = this.y;
        const L = this.L;
        const halfL = 0.5 * L;
        const bodyType = this.bodyType;
        const customPoly = this.customPolygon;

        // Rotate coordinates
        const angleRad = this.angle * Math.PI / 180.0;
        const cosA = Math.cos(-angleRad);
        const sinA = Math.sin(-angleRad);

        for (let j = 0; j < Ny; j++) {
            const jNx = j * Nx;
            const y = yArr[j];
            for (let i = 0; i < Nx; i++) {
                const idx = jNx + i;
                const x = xArr[i];

                const xRot = x * cosA - y * sinA;
                const yRot = x * sinA + y * cosA;

                let isInside = false;

                if (bodyType === 'circle') {
                    isInside = xRot * xRot + yRot * yRot < halfL * halfL;
                } else if (bodyType === 'square') {
                    const ax = xRot < 0 ? -xRot : xRot;
                    const ay = yRot < 0 ? -yRot : yRot;
                    isInside = ax < halfL && ay < halfL;
                } else if (bodyType === 'naca0012') {
                    const xc = (xRot + halfL) / L;
                    if (xc >= 0 && xc <= 1) {
                        const t = 0.12;
                        const sqrtXc = Math.sqrt(xc);
                        const xc2 = xc * xc;
                        const xc3 = xc2 * xc;
                        const xc4 = xc3 * xc;
                        const yt = 5 * t * (0.2969 * sqrtXc - 0.1260 * xc -
                                           0.3516 * xc2 + 0.2843 * xc3 -
                                           0.1015 * xc4);
                        const ayRot = yRot < 0 ? -yRot : yRot;
                        isInside = ayRot / L < yt;
                    }
                } else if (bodyType === 'custom' && customPoly) {
                    isInside = this._pointInPolygon(xRot, yRot, customPoly);
                }

                mask[idx] = isInside ? 0.0 : 1.0;
            }
        }

        // Add ground plane: solid from body bottom down to domain bottom
        if (this.addGround) {
            // Find lowest y coordinate of any solid cell
            let groundY = this.yMin; // default to domain bottom if no body
            for (let j = Ny - 1; j >= 0; j--) {
                const jNx = j * Nx;
                for (let i = 0; i < Nx; i++) {
                    if (mask[jNx + i] < 0.5) {
                        // Found a solid cell — track its y
                        if (yArr[j] < groundY || groundY === this.yMin) {
                            groundY = yArr[j];
                        }
                    }
                }
            }

            // Fill all cells below groundY as solid across full X range
            if (groundY > this.yMin) {
                for (let j = 0; j < Ny; j++) {
                    if (yArr[j] < groundY) {
                        const jNx = j * Nx;
                        for (let i = 0; i < Nx; i++) {
                            mask[jNx + i] = 0.0;
                        }
                    }
                }
            }
        }

        // Re-upload mask texture if WebGL is active
        if (this.gl && this.glTexMask) {
            this._uploadMaskTexture();
        }
    }

    // Apply boundary conditions — optimized with inlined idx
    applyBoundaryConditions() {
        const Nx = this.Nx;
        const Ny = this.Ny;
        const u = this.u;
        const v = this.v;
        const U = this.U;

        // Left boundary: inflow (u=U, v=0)
        for (let j = 0; j < Ny; j++) {
            const jNx = j * Nx;
            u[jNx] = U;
            v[jNx] = 0.0;
        }

        // Right boundary: outflow (zero gradient)
        const lastCol = Nx - 1;
        const prevCol = Nx - 2;
        for (let j = 0; j < Ny; j++) {
            const jNx = j * Nx;
            u[jNx + lastCol] = u[jNx + prevCol];
            v[jNx + lastCol] = v[jNx + prevCol];
        }

        // Top and bottom: slip walls (v=0, du/dy=0)
        for (let i = 0; i < Nx; i++) {
            // Bottom (j=0)
            v[i] = 0.0;
            u[i] = u[Nx + i];

            // Top (j=Ny-1)
            const topIdx = (Ny - 1) * Nx + i;
            const topIdx1 = (Ny - 2) * Nx + i;
            v[topIdx] = 0.0;
            u[topIdx] = u[topIdx1];
        }
    }

    // ── Explicit Projection Method (Chorin's method) ───────────────────────
    //
    // Algorithm per time step:
    //   1. Compute RHS: convection + diffusion (explicit finite differences)
    //   2. Intermediate velocity: u* = u + dt * RHS
    //   3. Immersed boundary: zero velocity inside solid
    //   4. Pressure Poisson: ∇²p = (1/dt) * ∇·u*
    //   5. Velocity correction: u = u* - dt * ∇p
    //   6. Boundary conditions

    // WALE (Wall-Adapting Local Eddy-viscosity) subgrid-scale model
    // Computes eddy viscosity nu_t per cell; vanishes in laminar/near-wall regions
    computeWALE() {
        const Nx = this.Nx;
        const Ny = this.Ny;
        const u = this.u;
        const v = this.v;
        const mask = this.mask;
        const nu_t = this.nu_t;
        const invHxC = this.invHxC;
        const invHyC = this.invHyC;
        const hx = this.hx;
        const hy = this.hy;
        const Cw = 0.325;
        const eps = 1.0e-12;

        let maxNuT = 0.0;

        for (let j = 1; j < Ny - 1; j++) {
            const jNx = j * Nx;
            const jm = jNx - Nx;
            const jp = jNx + Nx;
            const ihyc_j = invHyC[j];
            const hyLocal = 0.5 * (hy[j - 1] + hy[j]);

            for (let i = 1; i < Nx - 1; i++) {
                const c = jNx + i;

                if (mask[c] < 0.5) {
                    nu_t[c] = 0.0;
                    continue;
                }

                const ihxc_i = invHxC[i];
                const hxLocal = 0.5 * (hx[i - 1] + hx[i]);

                // Velocity gradient tensor (central differences)
                const g11 = (u[c + 1] - u[c - 1]) * ihxc_i;       // du/dx
                const g12 = (u[jp + i] - u[jm + i]) * ihyc_j;     // du/dy
                const g21 = (v[c + 1] - v[c - 1]) * ihxc_i;       // dv/dx
                const g22 = (v[jp + i] - v[jm + i]) * ihyc_j;     // dv/dy

                // Squared gradient tensor g² = g·g
                const g2_11 = g11 * g11 + g12 * g21;
                const g2_12 = g11 * g12 + g12 * g22;
                const g2_21 = g21 * g11 + g22 * g21;
                const g2_22 = g21 * g12 + g22 * g22;

                // Traceless symmetric part Sd (3D trace removal with g33=0)
                const trg2_3 = (g2_11 + g2_22) / 3.0;
                const Sd_11 = g2_11 - trg2_3;
                const Sd_22 = g2_22 - trg2_3;
                const Sd_33 = -trg2_3;
                const Sd_12 = 0.5 * (g2_12 + g2_21);

                // Inner products
                const SdSd = Sd_11 * Sd_11 + Sd_22 * Sd_22 + Sd_33 * Sd_33
                            + 2.0 * Sd_12 * Sd_12;
                const S_12 = 0.5 * (g12 + g21);
                const SS = g11 * g11 + g22 * g22 + 2.0 * S_12 * S_12;

                // Filter width and WALE constant
                const CwDelta2 = Cw * Cw * hxLocal * hyLocal;

                // WALE eddy viscosity (optimized: sqrt instead of pow)
                const sqrtSdSd = Math.sqrt(SdSd);
                const sqrtSS = Math.sqrt(SS);
                const SdSd_32 = SdSd * sqrtSdSd;                   // SdSd^(3/2)
                const SS_52 = SS * SS * sqrtSS;                     // SS^(5/2)
                const SdSd_54 = SdSd * Math.sqrt(sqrtSdSd);        // SdSd^(5/4)

                const nuT_val = CwDelta2 * SdSd_32 / (SS_52 + SdSd_54 + eps);
                nu_t[c] = nuT_val;

                if (nuT_val > maxNuT) maxNuT = nuT_val;
            }
        }

        // Boundary cells: nu_t = 0
        for (let j = 0; j < Ny; j++) {
            const jNx = j * Nx;
            nu_t[jNx] = 0.0;
            nu_t[jNx + Nx - 1] = 0.0;
        }
        for (let i = 0; i < Nx; i++) {
            nu_t[i] = 0.0;
            nu_t[(Ny - 1) * Nx + i] = 0.0;
        }

        this.maxNuT = maxNuT;
    }

    // Bilinear interpolation of a field at arbitrary world coordinates (non-uniform grid)
    // Uses precomputed inverse maps (invMapX/invMapY) for O(1) world→grid lookup
    _sampleField(field, wx, wy) {
        const xMin = this.xMin;
        const yMin = this.yMin;
        const invDomX = 1.0 / (this.xMax - xMin);
        const invDomY = 1.0 / (this.yMax - yMin);
        const Nx = this.Nx;
        const Ny = this.Ny;
        const invMapX = this.invMapX;
        const invMapY = this.invMapY;

        // Normalize world coords to [0,1], clamp
        let wxN = (wx - xMin) * invDomX;
        let wyN = (wy - yMin) * invDomY;
        if (wxN < 0.0) wxN = 0.0; else if (wxN > 1.0) wxN = 1.0;
        if (wyN < 0.0) wyN = 0.0; else if (wyN > 1.0) wyN = 1.0;

        // Inverse map lookup (linear interpolation in table) → grid UV [0,1]
        const mxF = wxN * 1023.0;
        const mxI = mxF | 0;
        const mxFrac = mxF - mxI;
        const gridU = mxI < 1023
            ? invMapX[mxI] * (1.0 - mxFrac) + invMapX[mxI + 1] * mxFrac
            : invMapX[1023];

        const myF = wyN * 1023.0;
        const myI = myF | 0;
        const myFrac = myF - myI;
        const gridV = myI < 1023
            ? invMapY[myI] * (1.0 - myFrac) + invMapY[myI + 1] * myFrac
            : invMapY[1023];

        // Convert grid UV to float index
        let gi = gridU * (Nx - 1);
        let gj = gridV * (Ny - 1);

        // Clamp to valid range
        if (gi < 0.0) gi = 0.0; else if (gi > Nx - 1.001) gi = Nx - 1.001;
        if (gj < 0.0) gj = 0.0; else if (gj > Ny - 1.001) gj = Ny - 1.001;

        const i0 = gi | 0;
        const j0 = gj | 0;
        const sx = gi - i0;
        const sy = gj - j0;

        // Bilinear interpolation
        const j0Nx = j0 * Nx;
        const j1Nx = j0Nx + Nx;
        return (1.0 - sx) * ((1.0 - sy) * field[j0Nx + i0] + sy * field[j1Nx + i0]) +
               sx * ((1.0 - sy) * field[j0Nx + i0 + 1] + sy * field[j1Nx + i0 + 1]);
    }

    // Semi-Lagrangian advection: backtrack along velocity, sample old field
    // Inlined sampling for performance (avoids per-cell function call + property lookups)
    advect() {
        const Nx = this.Nx;
        const Ny = this.Ny;
        const u = this.u;
        const v = this.v;
        const u_prev = this.u_prev;
        const v_prev = this.v_prev;
        const mask = this.mask;
        const xArr = this.x;
        const yArr = this.y;
        const dt = this.dt;
        const invMapX = this.invMapX;
        const invMapY = this.invMapY;
        const xMin = this.xMin;
        const yMin = this.yMin;
        const invDomX = 1.0 / (this.xMax - xMin);
        const invDomY = 1.0 / (this.yMax - yMin);
        const NxM1 = Nx - 1.001;
        const NyM1 = Ny - 1.001;

        for (let j = 1; j < Ny - 1; j++) {
            const jNx = j * Nx;
            const yj = yArr[j];
            for (let i = 1; i < Nx - 1; i++) {
                const c = jNx + i;
                if (mask[c] < 0.5) {
                    u[c] = 0.0;
                    v[c] = 0.0;
                    continue;
                }

                // Backtrack
                let wxN = (xArr[i] - dt * u_prev[c] - xMin) * invDomX;
                let wyN = (yj - dt * v_prev[c] - yMin) * invDomY;
                if (wxN < 0.0) wxN = 0.0; else if (wxN > 1.0) wxN = 1.0;
                if (wyN < 0.0) wyN = 0.0; else if (wyN > 1.0) wyN = 1.0;

                // Inverse map → grid indices
                const mxF = wxN * 1023.0;
                const mxI = mxF | 0;
                const mxFr = mxF - mxI;
                let gi = (mxI < 1023
                    ? invMapX[mxI] + (invMapX[mxI + 1] - invMapX[mxI]) * mxFr
                    : invMapX[1023]) * (Nx - 1);

                const myF = wyN * 1023.0;
                const myI = myF | 0;
                const myFr = myF - myI;
                let gj = (myI < 1023
                    ? invMapY[myI] + (invMapY[myI + 1] - invMapY[myI]) * myFr
                    : invMapY[1023]) * (Ny - 1);

                if (gi < 0.0) gi = 0.0; else if (gi > NxM1) gi = NxM1;
                if (gj < 0.0) gj = 0.0; else if (gj > NyM1) gj = NyM1;

                const i0 = gi | 0;
                const j0 = gj | 0;
                const sx = gi - i0;
                const sy = gj - j0;
                const osx = 1.0 - sx;
                const osy = 1.0 - sy;

                const j0Nx = j0 * Nx;
                const j1Nx = j0Nx + Nx;
                const idx00 = j0Nx + i0;
                const idx10 = idx00 + 1;
                const idx01 = j1Nx + i0;
                const idx11 = idx01 + 1;

                // Bilinear for both u and v (shared indices)
                u[c] = osx * (osy * u_prev[idx00] + sy * u_prev[idx01]) +
                        sx * (osy * u_prev[idx10] + sy * u_prev[idx11]);
                v[c] = osx * (osy * v_prev[idx00] + sy * v_prev[idx01]) +
                        sx * (osy * v_prev[idx10] + sy * v_prev[idx11]);
            }
        }
    }

    // Implicit diffusion: solve (I - nu*dt*∇²)u = rhs via Gauss-Seidel
    diffuse() {
        const Nx = this.Nx;
        const Ny = this.Ny;
        const u = this.u;
        const v = this.v;
        const u_prev = this.u_prev;
        const v_prev = this.v_prev;
        const mask = this.mask;
        const nu = this.nu;
        const dt = this.dt;
        const alpha = nu * dt;

        const lapAW = this.lapAW;
        const lapAE = this.lapAE;
        const lapAS = this.lapAS;
        const lapAN = this.lapAN;

        // Copy advected velocity as RHS
        u_prev.set(u);
        v_prev.set(v);

        const nIter = 2;
        for (let iter = 0; iter < nIter; iter++) {
            for (let j = 1; j < Ny - 1; j++) {
                const jNx = j * Nx;
                const jm = jNx - Nx;
                const jp = jNx + Nx;
                const aS_j = lapAS[j];
                const aN_j = lapAN[j];
                const denomY = aS_j + aN_j;

                for (let i = 1; i < Nx - 1; i++) {
                    const c = jNx + i;
                    if (mask[c] < 0.5) continue;

                    const aW_i = lapAW[i];
                    const aE_i = lapAE[i];
                    const denom = 1.0 + alpha * (aW_i + aE_i + denomY);

                    u[c] = (u_prev[c] + alpha * (aW_i * u[c - 1] + aE_i * u[c + 1] +
                            aS_j * u[jm + i] + aN_j * u[jp + i])) / denom;
                    v[c] = (v_prev[c] + alpha * (aW_i * v[c - 1] + aE_i * v[c + 1] +
                            aS_j * v[jm + i] + aN_j * v[jp + i])) / denom;
                }
            }
        }
    }

    // Vorticity confinement: re-inject vortical energy lost to numerical diffusion
    vorticityConfinement() {
        const Nx = this.Nx;
        const Ny = this.Ny;
        const u = this.u;
        const v = this.v;
        const mask = this.mask;
        const omega = this._omega;
        const invHxC = this.invHxC;
        const invHyC = this.invHyC;
        const hx = this.hx;
        const hy = this.hy;
        const dt = this.dt;
        const epsilon = 0.15;

        // Pass 1: compute vorticity ω = ∂v/∂x - ∂u/∂y
        for (let j = 1; j < Ny - 1; j++) {
            const jNx = j * Nx;
            const jm = jNx - Nx;
            const jp = jNx + Nx;
            const ihyc = invHyC[j];
            for (let i = 1; i < Nx - 1; i++) {
                const c = jNx + i;
                if (mask[c] < 0.5) {
                    omega[c] = 0.0;
                    continue;
                }
                omega[c] = (v[c + 1] - v[c - 1]) * invHxC[i] -
                           (u[jp + i] - u[jm + i]) * ihyc;
            }
        }

        // Pass 2: compute confinement force and apply
        // Need gradient of |ω|, so loop one cell inward (j=2..Ny-3, i=2..Nx-3)
        for (let j = 2; j < Ny - 2; j++) {
            const jNx = j * Nx;
            const jm = jNx - Nx;
            const jp = jNx + Nx;
            const ihyc = invHyC[j];
            const hyLocal = 0.5 * (hy[j - 1] + hy[j]);

            for (let i = 2; i < Nx - 2; i++) {
                const c = jNx + i;
                if (mask[c] < 0.5) continue;

                const ihxc = invHxC[i];
                const hxLocal = 0.5 * (hx[i - 1] + hx[i]);

                // Gradient of |ω|
                const absO_R = Math.abs(omega[c + 1]);
                const absO_L = Math.abs(omega[c - 1]);
                const absO_U = Math.abs(omega[jp + i]);
                const absO_D = Math.abs(omega[jm + i]);

                const etaX = (absO_R - absO_L) * ihxc;
                const etaY = (absO_U - absO_D) * ihyc;

                // Normalize
                const etaLen = Math.sqrt(etaX * etaX + etaY * etaY) + 1e-10;
                const nx = etaX / etaLen;
                const ny = etaY / etaLen;

                // Confinement force: f = ε * h * (N × ω)
                // In 2D: N × ω_z = (ny * ω, -nx * ω)
                const h = Math.sqrt(hxLocal * hyLocal);
                const force = epsilon * h * omega[c];
                u[c] += dt * ny * force;
                v[c] -= dt * nx * force;
            }
        }
    }

    // Pressure Poisson solve: ∇²p = (1/dt) * ∇·u*
    // Non-uniform grid, Red-Black SOR, warm-started from previous pressure
    solvePressure() {
        const Nx = this.Nx;
        const Ny = this.Ny;
        const mask = this.mask;
        const u = this.u;
        const v = this.v;
        const p = this.p;
        const div = this.div;
        const dt = this.dt;
        const invDt = 1.0 / dt;
        const omega = 1.5;
        const oneMinusOmega = 1.0 - omega;
        const lastCol = Nx - 1;
        const lastRow = Ny - 1;

        // Non-uniform metric arrays
        const invHxC = this.invHxC;
        const invHyC = this.invHyC;
        const lapAW = this.lapAW;
        const lapAE = this.lapAE;
        const lapAS = this.lapAS;
        const lapAN = this.lapAN;
        const lapDenomX = this.lapDenomX;
        const lapDenomY = this.lapDenomY;

        // Compute divergence of intermediate velocity: (1/dt) * ∇·u*
        // Central difference with non-uniform spacing (adjoint-consistent with gradient)
        for (let j = 1; j < Ny - 1; j++) {
            const jNx = j * Nx;
            const jm = jNx - Nx;
            const jp = jNx + Nx;
            const ihyc = invHyC[j];
            for (let i = 1; i < Nx - 1; i++) {
                const c = jNx + i;
                if (mask[c] > 0.5) {
                    div[c] = invDt * (
                        (u[c + 1] - u[c - 1]) * invHxC[i] +
                        (v[jp + i] - v[jm + i]) * ihyc
                    );
                } else {
                    div[c] = 0.0;
                }
            }
        }

        // Red-Black SOR (12 iterations, warm-started)
        for (let iter = 0; iter < 12; iter++) {
            // Red pass: (i+j) even
            for (let j = 1; j < Ny - 1; j++) {
                const jNx = j * Nx;
                const jm = jNx - Nx;
                const jp = jNx + Nx;
                const aS_j = lapAS[j];
                const aN_j = lapAN[j];
                const denY = lapDenomY[j];
                const iStart = 1 + (j & 1);
                for (let i = iStart; i < Nx - 1; i += 2) {
                    const c = jNx + i;
                    if (mask[c] > 0.5) {
                        const pGS = (lapAW[i] * p[c - 1] + lapAE[i] * p[c + 1] +
                                     aS_j * p[jm + i] + aN_j * p[jp + i] -
                                     div[c]) / (lapDenomX[i] + denY);
                        p[c] = oneMinusOmega * p[c] + omega * pGS;
                    }
                }
            }
            // Black pass: (i+j) odd
            for (let j = 1; j < Ny - 1; j++) {
                const jNx = j * Nx;
                const jm = jNx - Nx;
                const jp = jNx + Nx;
                const aS_j = lapAS[j];
                const aN_j = lapAN[j];
                const denY = lapDenomY[j];
                const iStart = 1 + ((j + 1) & 1);
                for (let i = iStart; i < Nx - 1; i += 2) {
                    const c = jNx + i;
                    if (mask[c] > 0.5) {
                        const pGS = (lapAW[i] * p[c - 1] + lapAE[i] * p[c + 1] +
                                     aS_j * p[jm + i] + aN_j * p[jp + i] -
                                     div[c]) / (lapDenomX[i] + denY);
                        p[c] = oneMinusOmega * p[c] + omega * pGS;
                    }
                }
            }
            // Pressure boundary conditions (Neumann: dp/dn = 0)
            for (let j = 0; j < Ny; j++) {
                const jNx = j * Nx;
                p[jNx] = p[jNx + 1];
                p[jNx + lastCol] = p[jNx + lastCol - 1];
            }
            for (let i = 0; i < Nx; i++) {
                p[i] = p[Nx + i];
                p[lastRow * Nx + i] = p[(lastRow - 1) * Nx + i];
            }
        }

        // Remove mean pressure to pin the constant (pure Neumann BCs)
        let pSum = 0.0;
        let nFluid = 0;
        const NxNy = Nx * Ny;
        for (let c = 0; c < NxNy; c++) {
            if (mask[c] > 0.5) {
                pSum += p[c];
                nFluid++;
            }
        }
        if (nFluid > 0) {
            const pMean = pSum / nFluid;
            for (let c = 0; c < NxNy; c++) {
                p[c] -= pMean;
            }
        }
    }

    // Correct velocity: u = u* - dt * ∇p (non-uniform grid)
    // Uses same central-difference operator as divergence (adjoint consistency)
    correctVelocity() {
        const Nx = this.Nx;
        const Ny = this.Ny;
        const u = this.u;
        const v = this.v;
        const p = this.p;
        const mask = this.mask;
        const dt = this.dt;
        const invHxC = this.invHxC;
        const invHyC = this.invHyC;

        for (let j = 1; j < Ny - 1; j++) {
            const jNx = j * Nx;
            const jm = jNx - Nx;
            const jp = jNx + Nx;
            const ihyc = invHyC[j];
            for (let i = 1; i < Nx - 1; i++) {
                const c = jNx + i;
                if (mask[c] > 0.5) {
                    u[c] -= dt * (p[c + 1] - p[c - 1]) * invHxC[i];
                    v[c] -= dt * (p[jp + i] - p[jm + i]) * ihyc;
                }
            }
        }
    }

    // Apply immersed boundary forcing: zero velocity inside solid
    applyImmersedBoundary() {
        const size = this.Nx * this.Ny;
        const mask = this.mask;
        const u = this.u;
        const v = this.v;
        for (let i = 0; i < size; i++) {
            if (mask[i] < 0.5) {
                u[i] = 0.0;
                v[i] = 0.0;
            }
        }
    }

    // Time step — Stable Fluids (Stam 1999)
    step() {
        // 1. Copy current velocity as source for semi-Lagrangian advection
        this.u_prev.set(this.u);
        this.v_prev.set(this.v);

        // 2. Semi-Lagrangian advection (already zeros solid cells)
        this.advect();
        this.applyBoundaryConditions();

        // 3. Implicit diffusion + vorticity confinement
        this.diffuse();
        this.vorticityConfinement();
        this.applyImmersedBoundary();
        this.applyBoundaryConditions();

        // 4. Pressure projection
        this.solvePressure();
        this.correctVelocity();
        this.applyImmersedBoundary();
        this.applyBoundaryConditions();

        // Update time + forces
        this.time += this.dt;
        this.computeForces();
    }

    // Force computation: pressure integration on body surface
    computeForces() {
        let fx = 0.0;
        let fy = 0.0;
        const Nx = this.Nx;
        const Ny = this.Ny;
        const mask = this.mask;
        const p = this.p;
        const hx = this.hx;
        const hy = this.hy;

        for (let j = 1; j < Ny - 1; j++) {
            const jNx = j * Nx;
            const jNx_prev = jNx - Nx;
            const jNx_next = jNx + Nx;
            const faceH = 0.5 * (hy[j - 1] + hy[j]);

            for (let i = 1; i < Nx - 1; i++) {
                const idx = jNx + i;

                if (mask[idx] < 0.5) {
                    const idx_r = idx + 1;
                    const idx_l = idx - 1;
                    const idx_u = jNx_next + i;
                    const idx_d = jNx_prev + i;
                    const faceW = 0.5 * (hx[i - 1] + hx[i]);

                    if (mask[idx_r] > 0.5) fx -= p[idx_r] * faceH;
                    if (mask[idx_l] > 0.5) fx += p[idx_l] * faceH;
                    if (mask[idx_u] > 0.5) fy -= p[idx_u] * faceW;
                    if (mask[idx_d] > 0.5) fy += p[idx_d] * faceW;
                }
            }
        }

        const qA = 0.5 * this.U * this.U * this.L;
        const invQA = 1.0 / qA;
        const drag = fx * invQA;
        const lift = fy * invQA;

        this.liftHistory.push(lift);
        this.dragHistory.push(drag);

        if (this.liftHistory.length > this.maxHistoryLength) {
            this.liftHistory.shift();
            this.dragHistory.shift();
        }

        document.getElementById('lift-inst').textContent = lift.toFixed(1);
        document.getElementById('drag-inst').textContent = drag.toFixed(1);

        if (this.liftHistory.length > 10) {
            const len = this.liftHistory.length;
            const invLen = 1.0 / len;
            let sumL = 0.0, sumD = 0.0;
            for (let i = 0; i < len; i++) {
                sumL += this.liftHistory[i];
                sumD += this.dragHistory[i];
            }
            document.getElementById('lift-avg').textContent = (sumL * invLen).toFixed(1);
            document.getElementById('drag-avg').textContent = (sumD * invLen).toFixed(1);
        }
    }

    // Compute 10th/90th percentile of field for dynamic colorbar
    _computeFieldRange(field) {
        const mask = this.mask;
        const size = field.length;
        // Collect fluid-cell values only (skip solid)
        if (!this._rangeBuf || this._rangeBuf.length < size) {
            this._rangeBuf = new Float32Array(size);
        }
        let n = 0;
        for (let i = 0; i < size; i++) {
            if (mask[i] > 0.5) this._rangeBuf[n++] = field[i];
        }
        if (n === 0) return { min: 0, max: 1 };
        // Partial sort: we only need 10th and 90th percentile
        // Use full sort on a subsample for speed (every 8th value)
        const step = n > 8000 ? 8 : 1;
        const sub = [];
        for (let i = 0; i < n; i += step) sub.push(this._rangeBuf[i]);
        sub.sort((a, b) => a - b);
        const lo = sub[Math.floor(sub.length * 0.01)];
        const hi = sub[Math.floor(sub.length * 0.99)];
        return { min: lo, max: hi > lo + 1e-8 ? hi : lo + 1e-8 };
    }

    // ── Canvas + WebGL setup ───────────────────────────────────────────────

    setupCanvas() {
        this.canvas = document.getElementById('renderCanvas');

        // Try WebGL2 first
        this.gl = null;
        this.useWebGL = false;

        try {
            const gl = this.canvas.getContext('webgl2', {
                antialias: false,
                depth: false,
                stencil: false,
                alpha: false,
                preserveDrawingBuffer: false,
                powerPreference: 'high-performance'
            });
            if (gl) {
                this.gl = gl;
                this.useWebGL = true;
                this._initWebGL(gl);
            }
        } catch (e) {
            // WebGL2 not available, fall back
        }

        if (!this.useWebGL) {
            this.ctx = this.canvas.getContext('2d');

            // Show fallback warning banner
            const warn = document.createElement('div');
            warn.style.cssText = 'position:absolute;top:0;left:0;right:0;padding:8px 36px 8px 12px;background:#b8860b;color:#fff;font-size:13px;z-index:10;text-align:center';
            warn.textContent = '\u26A0 WebGL2 not available \u2014 using software rendering. Performance may be reduced.';
            const dismiss = document.createElement('button');
            dismiss.textContent = '\u2715';
            dismiss.style.cssText = 'position:absolute;right:8px;top:50%;transform:translateY(-50%);background:none;border:none;color:#fff;font-size:16px;cursor:pointer;padding:4px';
            dismiss.addEventListener('click', () => warn.remove());
            warn.appendChild(dismiss);
            document.getElementById('canvas-container').appendChild(warn);
        }

        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // Setup colorbar
        this.colorbarCanvas = document.getElementById('colorbar');
        this.colorbarCtx = this.colorbarCanvas.getContext('2d');
        this.drawColorbar();
    }

    _initWebGL(gl) {
        // Check for float linear filtering extension
        this.hasFloatLinear = !!gl.getExtension('OES_texture_float_linear');

        // Compile shaders — inject #define for float linear (compile-time branch)
        const fragSrc = this.hasFloatLinear
            ? FRAGMENT_SHADER_TEMPLATE.replace('#version 300 es', '#version 300 es\n#define HAS_FLOAT_LINEAR')
            : FRAGMENT_SHADER_TEMPLATE;
        const vs = this._compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
        const fs = this._compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
        this.glProgram = this._linkProgram(gl, vs, fs);

        // Get uniform locations
        const prog = this.glProgram;
        this.glUniforms = {
            resolution: gl.getUniformLocation(prog, 'u_resolution'),
            domainMin: gl.getUniformLocation(prog, 'u_domainMin'),
            domainMax: gl.getUniformLocation(prog, 'u_domainMax'),
            viewMin: gl.getUniformLocation(prog, 'u_viewMin'),
            viewMax: gl.getUniformLocation(prog, 'u_viewMax'),
            texelSize: gl.getUniformLocation(prog, 'u_texelSize'),
            fieldMin: gl.getUniformLocation(prog, 'u_fieldMin'),
            fieldMax: gl.getUniformLocation(prog, 'u_fieldMax'),
            vizMode: gl.getUniformLocation(prog, 'u_vizMode'),
            texU: gl.getUniformLocation(prog, 'u_texU'),
            texV: gl.getUniformLocation(prog, 'u_texV'),
            texP: gl.getUniformLocation(prog, 'u_texP'),
            texMask: gl.getUniformLocation(prog, 'u_texMask'),
            texViridis: gl.getUniformLocation(prog, 'u_texViridis'),
            texInvMapX: gl.getUniformLocation(prog, 'u_texInvMapX'),
            texInvMapY: gl.getUniformLocation(prog, 'u_texInvMapY'),
        };

        // Create empty VAO (no vertex data needed)
        this.glVAO = gl.createVertexArray();

        // Create R32F textures for u, v, p, mask
        const filterMode = this.hasFloatLinear ? gl.LINEAR : gl.NEAREST;
        this.glTexU = this._createR32FTexture(gl, this.Nx, this.Ny, filterMode);
        this.glTexV = this._createR32FTexture(gl, this.Nx, this.Ny, filterMode);
        this.glTexP = this._createR32FTexture(gl, this.Nx, this.Ny, filterMode);
        this.glTexMask = this._createR32FTexture(gl, this.Nx, this.Ny, filterMode);

        // Create viridis 1D texture (RGBA8, 256×1)
        this.glTexViridis = this._createViridisTexture(gl);

        // Create inverse-map textures for stretched grid (1024×1, R32F, LINEAR)
        this.glTexInvMapX = this._createR32FTexture(gl, 1024, 1, gl.LINEAR);
        this.glTexInvMapY = this._createR32FTexture(gl, 1024, 1, gl.LINEAR);
        this._uploadInverseMapTextures();

        // Upload initial mask
        this._uploadMaskTexture();

        // Set static uniforms
        gl.useProgram(prog);
        gl.uniform2f(this.glUniforms.domainMin, this.xMin, this.yMin);
        gl.uniform2f(this.glUniforms.domainMax, this.xMax, this.yMax);
        gl.uniform2f(this.glUniforms.viewMin, this.viewXMin, this.viewYMin);
        gl.uniform2f(this.glUniforms.viewMax, this.viewXMax, this.viewYMax);
        gl.uniform2f(this.glUniforms.texelSize, 1.0 / this.Nx, 1.0 / this.Ny);

        // Bind texture units (sampler uniforms)
        gl.uniform1i(this.glUniforms.texU, 0);
        gl.uniform1i(this.glUniforms.texV, 1);
        gl.uniform1i(this.glUniforms.texP, 2);
        gl.uniform1i(this.glUniforms.texMask, 3);
        gl.uniform1i(this.glUniforms.texViridis, 4);
        gl.uniform1i(this.glUniforms.texInvMapX, 5);
        gl.uniform1i(this.glUniforms.texInvMapY, 6);

        // Bind all textures to their units permanently (single-program app)
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexU);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexV);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexP);
        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexMask);
        gl.activeTexture(gl.TEXTURE4);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexViridis);
        gl.activeTexture(gl.TEXTURE5);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexInvMapX);
        gl.activeTexture(gl.TEXTURE6);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexInvMapY);

        // Bind VAO permanently (no vertex data, just needs to be bound)
        gl.bindVertexArray(this.glVAO);

        // Log GPU info
        const dbg = gl.getExtension('WEBGL_debug_renderer_info');
        if (dbg) {
            this._gpuRenderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
        } else {
            this._gpuRenderer = gl.getParameter(gl.RENDERER);
        }
        console.log('WebGL2 GPU:', this._gpuRenderer);
    }

    _uploadInverseMapTextures() {
        const gl = this.gl;
        gl.activeTexture(gl.TEXTURE5);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexInvMapX);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 1024, 1, gl.RED, gl.FLOAT, this.invMapX);
        gl.activeTexture(gl.TEXTURE6);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexInvMapY);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 1024, 1, gl.RED, gl.FLOAT, this.invMapY);
    }

    _compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    _linkProgram(gl, vs, fs) {
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(prog));
            return null;
        }
        return prog;
    }

    _createR32FTexture(gl, width, height, filter) {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return tex;
    }

    _createViridisTexture(gl) {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        // Convert 3-channel LUT to RGBA
        const lut = this.viridisLUT;
        const N = lut.length / 3;
        const rgba = new Uint8Array(N * 4);
        for (let i = 0; i < N; i++) {
            rgba[i * 4] = lut[i * 3];
            rgba[i * 4 + 1] = lut[i * 3 + 1];
            rgba[i * 4 + 2] = lut[i * 3 + 2];
            rgba[i * 4 + 3] = 255;
        }
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, N, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return tex;
    }

    _uploadMaskTexture() {
        const gl = this.gl;
        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexMask);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.Nx, this.Ny, gl.RED, gl.FLOAT, this.mask);
    }

    resizeCanvas() {
        const container = document.getElementById('canvas-container');
        const w = container.clientWidth || window.innerWidth;
        const h = container.clientHeight || window.innerHeight;
        this.canvas.width = w;
        this.canvas.height = h;

        // Adjust view window to match canvas aspect ratio (keeps body centered)
        // Strategy: fix the shorter screen axis to baseHalf, expand the longer one
        const aspect = w / Math.max(h, 1);
        const baseHalf = this._viewHalfH;
        let halfW, halfH;
        if (aspect >= 1.0) {
            // Landscape: height is shorter axis
            halfH = baseHalf;
            halfW = baseHalf * aspect;
            // Clamp so we don't zoom out too far horizontally
            if (halfW > 8.0) halfW = 8.0;
        } else {
            // Portrait: width is shorter axis
            halfW = baseHalf;
            halfH = baseHalf / aspect;
            // Clamp so we don't zoom out too far vertically
            if (halfH > 8.0) halfH = 8.0;
        }
        this.viewXMin = this._viewCenterX - halfW;
        this.viewXMax = this._viewCenterX + halfW;
        this.viewYMin = this._viewCenterY - halfH;
        this.viewYMax = this._viewCenterY + halfH;

        if (this.useWebGL && this.gl) {
            this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        }

        // Invalidate offscreen canvas on resize (for Canvas2D fallback)
        this.offCanvas = null;

        // Resize mesh overlay and mark dirty
        if (this.meshCanvas) {
            this.meshCanvas.width = this.canvas.width;
            this.meshCanvas.height = this.canvas.height;
            this._meshDirty = true;
        }
    }

    // Draw grid mesh lines on overlay canvas (only in the view window)
    _drawMeshOverlay() {
        if (!this._meshDirty || !this.meshCanvas) return;
        this._meshDirty = false;

        const mc = this.meshCanvas;
        const ctx = mc.getContext('2d');
        const w = mc.width;
        const h = mc.height;
        ctx.clearRect(0, 0, w, h);

        const vxMin = this.viewXMin;
        const vxMax = this.viewXMax;
        const vyMin = this.viewYMin;
        const vyMax = this.viewYMax;
        const vxRange = vxMax - vxMin;
        const vyRange = vyMax - vyMin;

        ctx.strokeStyle = 'rgba(0,0,0,0.0)';
        ctx.lineWidth = 0.5;

        // Vertical lines (x = const)
        const x = this.x;
        const Nx = this.Nx;
        ctx.beginPath();
        for (let i = 0; i < Nx; i++) {
            if (x[i] < vxMin || x[i] > vxMax) continue;
            const px = ((x[i] - vxMin) / vxRange) * w;
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
        }
        ctx.stroke();

        // Horizontal lines (y = const)
        const y = this.y;
        const Ny = this.Ny;
        ctx.beginPath();
        for (let j = 0; j < Ny; j++) {
            if (y[j] < vyMin || y[j] > vyMax) continue;
            // Canvas y is flipped: top=0 corresponds to vyMax
            const py = ((vyMax - y[j]) / vyRange) * h;
            ctx.moveTo(0, py);
            ctx.lineTo(w, py);
        }
        ctx.stroke();
    }

    drawColorbar() {
        const width = 200;
        const height = 30;
        this.colorbarCanvas.width = width;
        this.colorbarCanvas.height = height;

        const lut = this.viridisLUT;
        const N = lut.length / 3;
        for (let i = 0; i < width; i++) {
            const lutIdx = ((i / width) * N) | 0;
            const li = (lutIdx < N ? lutIdx : N - 1) * 3;
            this.colorbarCtx.fillStyle = `rgb(${lut[li]},${lut[li + 1]},${lut[li + 2]})`;
            this.colorbarCtx.fillRect(i, 0, 1, height);
        }
    }

    // ── WebGL render ───────────────────────────────────────────────────────

    render() {
        if (this.useWebGL) {
            this._renderWebGL();
        } else {
            this._renderCanvas2D();
        }
        this._drawMeshOverlay();
    }

    _renderWebGL() {
        const gl = this.gl;
        const Nx = this.Nx;
        const Ny = this.Ny;

        // Colormap ranges: fixed for velocity, dynamic (10/90 percentile) for pressure
        let minVal, maxVal, vizModeInt;
        if (this.vizMode === 'velocity') {
            minVal = 0.;
            maxVal = 1.25;
            vizModeInt = 0;
        } else {
            const range = this._computeFieldRange(this.p);
            minVal = range.min;
            maxVal = range.max;
            vizModeInt = 1;
        }

        // Upload textures — explicit bind before each upload (required on iOS Safari)
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexU);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, Nx, Ny, gl.RED, gl.FLOAT, this.u);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexV);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, Nx, Ny, gl.RED, gl.FLOAT, this.v);

        if (vizModeInt === 1) {
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, this.glTexP);
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, Nx, Ny, gl.RED, gl.FLOAT, this.p);
        }

        // Re-bind mask texture (may have been re-uploaded by generateBody)
        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, this.glTexMask);

        // Static textures (3-6) are bound once in _initWebGL, no re-bind needed

        // Set per-frame uniforms (program is already bound from init)
        gl.uniform2f(this.glUniforms.resolution, this.canvas.width, this.canvas.height);
        gl.uniform2f(this.glUniforms.viewMin, this.viewXMin, this.viewYMin);
        gl.uniform2f(this.glUniforms.viewMax, this.viewXMax, this.viewYMax);
        gl.uniform1f(this.glUniforms.fieldMin, minVal);
        gl.uniform1f(this.glUniforms.fieldMax, maxVal);
        gl.uniform1i(this.glUniforms.vizMode, vizModeInt);

        // Draw full-screen triangle (VAO is already bound from init)
        gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    // ── Canvas 2D fallback ─────────────────────────────────────────────────

    _getViewGridDims() {
        // Count grid cells visible in view window
        let cellsX = 0;
        for (let i = 0; i < this.Nx - 1; i++) {
            if (this.x[i + 1] > this.viewXMin && this.x[i] < this.viewXMax) cellsX++;
        }
        let cellsY = 0;
        for (let j = 0; j < this.Ny - 1; j++) {
            if (this.y[j + 1] > this.viewYMin && this.y[j] < this.viewYMax) cellsY++;
        }
        return { cellsX: Math.max(cellsX, 1), cellsY: Math.max(cellsY, 1) };
    }

    _ensureOffCanvas() {
        if (this.offCanvas) return;
        const { cellsX, cellsY } = this._getViewGridDims();
        this.offCanvas = document.createElement('canvas');
        this.offCanvas.width = cellsX;
        this.offCanvas.height = cellsY;
        this.offCtx = this.offCanvas.getContext('2d');
        this.offImageData = this.offCtx.createImageData(cellsX, cellsY);
    }

    _renderCanvas2D() {
        this._ensureOffCanvas();

        const ow = this.offCanvas.width;
        const oh = this.offCanvas.height;
        const data = this.offImageData.data;
        const Nx = this.Nx;
        const Ny = this.Ny;
        const size = Nx * Ny;
        const mask = this.mask;

        // Compute field to visualize
        let field;
        if (this.vizMode === 'velocity') {
            if (!this._velMag || this._velMag.length !== size) {
                this._velMag = new Float32Array(size);
            }
            const mag = this._velMag;
            const u = this.u;
            const v = this.v;
            for (let i = 0; i < size; i++) {
                mag[i] = Math.sqrt(u[i] * u[i] + v[i] * v[i]);
            }
            field = mag;
        } else {
            field = this.p;
        }

        // Colormap ranges: fixed for velocity, dynamic (10/90 percentile) for pressure
        let minVal, maxVal;
        if (this.vizMode === 'velocity') {
            minVal = 0.;
            maxVal = 1.25;
        } else {
            const fieldRange = this._computeFieldRange(field);
            minVal = fieldRange.min;
            maxVal = fieldRange.max;
        }
        const span = maxVal - minVal;
        const invRange = 1.0 / span;

        const lut = this.viridisLUT;
        const lutN = lut.length / 3;
        const lutMax = lutN - 1;

        const viewXMin = this.viewXMin;
        const viewXMax = this.viewXMax;
        const viewYMin = this.viewYMin;
        const viewYMax = this.viewYMax;
        const xMin = this.xMin;
        const yMin = this.yMin;
        const domainW = this.xMax - this.xMin;
        const domainH = this.yMax - this.yMin;
        const invOw = 1.0 / ow;
        const invOh = 1.0 / oh;
        const NxM1 = Nx - 1;
        const NyM1 = Ny - 1;

        // Inverse map for world-to-grid on non-uniform grid
        const invMapX = this.invMapX;
        const invMapY = this.invMapY;
        const invMapMax = 1023;

        for (let py = 0; py < oh; py++) {
            const wy = viewYMax - ((py + 0.5) * invOh) * (viewYMax - viewYMin);
            // Inverse map lookup for y
            const wyNorm = (wy - yMin) / domainH;
            const myF = wyNorm * invMapMax;
            const myI = myF | 0;
            const myFrac = myF - myI;
            const gridVY = (myI < invMapMax)
                ? invMapY[myI] * (1.0 - myFrac) + invMapY[myI + 1] * myFrac
                : invMapY[invMapMax];
            const gj = gridVY * NyM1;

            for (let px = 0; px < ow; px++) {
                const wx = viewXMin + ((px + 0.5) * invOw) * (viewXMax - viewXMin);
                // Inverse map lookup for x
                const wxNorm = (wx - xMin) / domainW;
                const mxF = wxNorm * invMapMax;
                const mxI = mxF | 0;
                const mxFrac = mxF - mxI;
                const gridVX = (mxI < invMapMax)
                    ? invMapX[mxI] * (1.0 - mxFrac) + invMapX[mxI + 1] * mxFrac
                    : invMapX[invMapMax];
                const gi = gridVX * NxM1;
                const pixelIdx = (py * ow + px) * 4;

                if (gi >= 0 && gi < Nx - 1 && gj >= 0 && gj < Ny - 1) {
                    const i0 = gi | 0;
                    const j0 = gj | 0;

                    const sx = gi - i0;
                    const sy = gj - j0;

                    const j0Nx = j0 * Nx;
                    const j1Nx = j0Nx + Nx;
                    const idx00 = j0Nx + i0;
                    const idx10 = j0Nx + i0 + 1;
                    const idx01 = j1Nx + i0;
                    const idx11 = j1Nx + i0 + 1;

                    const maskVal = (1 - sx) * ((1 - sy) * mask[idx00] + sy * mask[idx01]) +
                                   sx * ((1 - sy) * mask[idx10] + sy * mask[idx11]);

                    if (maskVal < 0.5) {
                        data[pixelIdx] = 30;
                        data[pixelIdx + 1] = 30;
                        data[pixelIdx + 2] = 30;
                        data[pixelIdx + 3] = 255;
                    } else {
                        const val = (1 - sx) * ((1 - sy) * field[idx00] + sy * field[idx01]) +
                                   sx * ((1 - sy) * field[idx10] + sy * field[idx11]);

                        const t = (val - minVal) * invRange;
                        let lutIdx = (t * lutMax) | 0;
                        if (lutIdx < 0) lutIdx = 0;
                        else if (lutIdx > lutMax) lutIdx = lutMax;
                        const li = lutIdx * 3;

                        data[pixelIdx] = lut[li];
                        data[pixelIdx + 1] = lut[li + 1];
                        data[pixelIdx + 2] = lut[li + 2];
                        data[pixelIdx + 3] = 255;
                    }
                } else {
                    data[pixelIdx] = 0;
                    data[pixelIdx + 1] = 0;
                    data[pixelIdx + 2] = 0;
                    data[pixelIdx + 3] = 255;
                }
            }
        }

        this.offCtx.putImageData(this.offImageData, 0, 0);

        if (!this.ctx) {
            this.ctx = this.canvas.getContext('2d');
        }
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'low';
        this.ctx.drawImage(this.offCanvas, 0, 0, this.canvas.width, this.canvas.height);
    }

    // ── Custom Body Upload ────────────────────────────────────────────────

    setupUploadUI() {
        const uploadBtn = document.getElementById('uploadBodyBtn');
        const fileInput = document.getElementById('bodyFileInput');
        const overlay = document.getElementById('uploadOverlay');
        const uploadCanvas = document.getElementById('uploadCanvas');
        const thresholdSlider = document.getElementById('thresholdSlider');
        const thresholdValue = document.getElementById('thresholdValue');
        const acceptBtn = document.getElementById('uploadAccept');
        const cancelBtn = document.getElementById('uploadCancel');
        const instruction = document.getElementById('uploadInstruction');

        // Store references
        this._uploadOverlay = overlay;
        this._uploadCanvas = uploadCanvas;
        this._uploadInstruction = instruction;
        this._uploadImage = null;       // loaded Image object
        this._uploadRect = null;        // {x, y, w, h} in canvas pixels
        this._uploadLasso = null;       // Array of {x, y} for lasso path
        this._uploadRawPolygon = null;  // polygon in pixel coords before normalization
        this._uploadThreshold = 128;
        this._uploadDragging = false;
        this._uploadStart = null;
        this._uploadSelectMode = 'box'; // 'box' or 'lasso'

        // Mode toggle buttons
        const modeBoxBtn = document.getElementById('modeBox');
        const modeLassoBtn = document.getElementById('modeLasso');

        modeBoxBtn.addEventListener('click', () => {
            this._uploadSelectMode = 'box';
            modeBoxBtn.classList.add('active');
            modeLassoBtn.classList.remove('active');
            instruction.textContent = 'Draw a rectangle around the shape';
            if (this._uploadState === 'previewing' && this._uploadImage) {
                // Reset to selection mode, redraw clean image
                this._uploadState = 'selecting';
                const ctx = uploadCanvas.getContext('2d');
                ctx.drawImage(this._uploadImage, 0, 0, uploadCanvas.width, uploadCanvas.height);
            }
        });

        modeLassoBtn.addEventListener('click', () => {
            this._uploadSelectMode = 'lasso';
            modeLassoBtn.classList.add('active');
            modeBoxBtn.classList.remove('active');
            instruction.textContent = 'Draw a freehand outline around the shape';
            if (this._uploadState === 'previewing' && this._uploadImage) {
                // Reset to selection mode, redraw clean image
                this._uploadState = 'selecting';
                const ctx = uploadCanvas.getContext('2d');
                ctx.drawImage(this._uploadImage, 0, 0, uploadCanvas.width, uploadCanvas.height);
            }
        });

        // Upload button triggers file input
        uploadBtn.addEventListener('click', () => fileInput.click());

        // File selected
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                this._handleImageUpload(e.target.files[0]);
                e.target.value = ''; // reset so same file can be re-selected
            }
        });

        // Selection on upload canvas (handles both box and lasso)
        // Shared handlers for mouse + touch
        const onPointerDown = (clientX, clientY) => {
            if (this._uploadState !== 'selecting') return;
            const rect = uploadCanvas.getBoundingClientRect();
            const px = clientX - rect.left;
            const py = clientY - rect.top;
            this._uploadDragging = true;
            this._uploadStart = { x: px, y: py };

            if (this._uploadSelectMode === 'lasso') {
                this._uploadLasso = [{ x: px, y: py }];
            }
        };

        const onPointerMove = (clientX, clientY) => {
            if (!this._uploadDragging) return;
            const rect = uploadCanvas.getBoundingClientRect();
            const px = clientX - rect.left;
            const py = clientY - rect.top;

            if (this._uploadSelectMode === 'box') {
                this._drawSelectionRect(this._uploadStart.x, this._uploadStart.y, px, py);
            } else {
                this._uploadLasso.push({ x: px, y: py });
                this._drawLassoPath(this._uploadLasso);
            }
        };

        const onPointerUp = (clientX, clientY) => {
            if (!this._uploadDragging) return;
            this._uploadDragging = false;

            if (this._uploadSelectMode === 'box') {
                const rect = uploadCanvas.getBoundingClientRect();
                const cx = clientX - rect.left;
                const cy = clientY - rect.top;
                const sx = this._uploadStart.x;
                const sy = this._uploadStart.y;

                const x0 = Math.min(sx, cx);
                const y0 = Math.min(sy, cy);
                const bw = Math.abs(cx - sx);
                const bh = Math.abs(cy - sy);

                if (bw < 20 || bh < 20) {
                    instruction.textContent = 'Selection too small — draw a larger rectangle';
                    return;
                }

                this._uploadRect = { x: x0, y: y0, w: bw, h: bh };
                this._uploadLasso = null;
            } else {
                // Lasso mode — compute bounding rect from lasso path
                const lasso = this._uploadLasso;
                if (!lasso || lasso.length < 10) {
                    instruction.textContent = 'Draw a larger outline around the shape';
                    return;
                }

                let lxMin = Infinity, lxMax = -Infinity;
                let lyMin = Infinity, lyMax = -Infinity;
                for (let i = 0; i < lasso.length; i++) {
                    if (lasso[i].x < lxMin) lxMin = lasso[i].x;
                    if (lasso[i].x > lxMax) lxMax = lasso[i].x;
                    if (lasso[i].y < lyMin) lyMin = lasso[i].y;
                    if (lasso[i].y > lyMax) lyMax = lasso[i].y;
                }

                const bw = lxMax - lxMin;
                const bh = lyMax - lyMin;
                if (bw < 20 || bh < 20) {
                    instruction.textContent = 'Selection too small — draw a larger outline';
                    return;
                }

                this._uploadRect = { x: lxMin, y: lyMin, w: bw, h: bh };
            }

            // Auto-trace
            this._uploadThreshold = parseInt(thresholdSlider.value);
            this._runTraceAndPreview();
        };

        // Mouse events
        uploadCanvas.addEventListener('mousedown', (e) => onPointerDown(e.clientX, e.clientY));
        uploadCanvas.addEventListener('mousemove', (e) => onPointerMove(e.clientX, e.clientY));
        uploadCanvas.addEventListener('mouseup', (e) => onPointerUp(e.clientX, e.clientY));

        // Touch events (parallel to mouse, with preventDefault to suppress scroll/zoom)
        uploadCanvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const t = e.touches[0];
            onPointerDown(t.clientX, t.clientY);
        }, { passive: false });
        uploadCanvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const t = e.touches[0];
            onPointerMove(t.clientX, t.clientY);
        }, { passive: false });
        uploadCanvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const t = e.changedTouches[0];
            onPointerUp(t.clientX, t.clientY);
        }, { passive: false });

        // Threshold slider
        thresholdSlider.addEventListener('input', (e) => {
            this._uploadThreshold = parseInt(e.target.value);
            thresholdValue.textContent = e.target.value;
            if (this._uploadState === 'previewing' && this._uploadRect) {
                this._runTraceAndPreview();
            }
        });

        // Accept
        acceptBtn.addEventListener('click', () => this._acceptTrace());

        // Cancel
        cancelBtn.addEventListener('click', () => this._cancelUpload());

        // Flip buttons
        document.getElementById('flipHBtn').addEventListener('click', () => {
            if (!this.customPolygon) return;
            for (let i = 0; i < this.customPolygon.length; i++) {
                this.customPolygon[i].x = -this.customPolygon[i].x;
            }
            this.generateBody();
        });

        document.getElementById('flipVBtn').addEventListener('click', () => {
            if (!this.customPolygon) return;
            for (let i = 0; i < this.customPolygon.length; i++) {
                this.customPolygon[i].y = -this.customPolygon[i].y;
            }
            this.generateBody();
        });
    }

    _handleImageUpload(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this._uploadImage = img;
                this._showUploadOverlay(img);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    _showUploadOverlay(img) {
        // Pause simulation
        this.running = false;

        const overlay = this._uploadOverlay;
        const canvas = this._uploadCanvas;
        const instruction = this._uploadInstruction;

        // Size upload canvas to fit image while respecting overlay constraints
        const container = document.getElementById('canvas-container');
        const maxW = container.clientWidth * 0.9;
        const maxH = container.clientHeight * 0.7;
        const scale = Math.min(maxW / img.width, maxH / img.height, 1.0);
        canvas.width = Math.round(img.width * scale);
        canvas.height = Math.round(img.height * scale);

        // Draw image
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Store image scale for coordinate mapping
        this._uploadImgScale = scale;

        // Show overlay
        overlay.classList.add('visible');
        instruction.textContent = this._uploadSelectMode === 'lasso'
            ? 'Draw a freehand outline around the shape'
            : 'Draw a rectangle around the shape';
        this._uploadState = 'selecting';
        this._uploadRect = null;
        this._uploadLasso = null;
        this._uploadRawPolygon = null;
    }

    _drawSelectionRect(x0, y0, x1, y1) {
        const canvas = this._uploadCanvas;
        const ctx = canvas.getContext('2d');
        const img = this._uploadImage;
        const scale = this._uploadImgScale;

        // Redraw image
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Draw rectangle
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
    }

    _drawLassoPath(lasso) {
        const canvas = this._uploadCanvas;
        const ctx = canvas.getContext('2d');

        // Redraw image
        ctx.drawImage(this._uploadImage, 0, 0, canvas.width, canvas.height);

        if (lasso.length < 2) return;

        // Draw lasso path
        ctx.beginPath();
        ctx.moveTo(lasso[0].x, lasso[0].y);
        for (let i = 1; i < lasso.length; i++) {
            ctx.lineTo(lasso[i].x, lasso[i].y);
        }
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw closing line (dashed) from current point back to start
        ctx.beginPath();
        ctx.moveTo(lasso[lasso.length - 1].x, lasso[lasso.length - 1].y);
        ctx.lineTo(lasso[0].x, lasso[0].y);
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(76, 175, 80, 0.5)';
        ctx.stroke();
        ctx.setLineDash([]);
    }

    _runTraceAndPreview() {
        const canvas = this._uploadCanvas;
        const ctx = canvas.getContext('2d');
        const rect = this._uploadRect;

        // Get image data from selection rectangle
        // First redraw clean image to avoid overlays contaminating pixel data
        ctx.drawImage(this._uploadImage, 0, 0, canvas.width, canvas.height);

        const rx = Math.round(rect.x);
        const ry = Math.round(rect.y);
        const rw = Math.round(rect.w);
        const rh = Math.round(rect.h);
        const imageData = ctx.getImageData(rx, ry, rw, rh);

        // In lasso mode, mask out pixels outside the lasso polygon
        // Set them to white (background) so they don't get detected as solid
        if (this._uploadLasso && this._uploadLasso.length > 3) {
            const lasso = this._uploadLasso;
            const data = imageData.data;
            // Determine background color from corners of the sub-image
            const cornerSum = (
                (0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]) +
                (0.299 * data[(rw - 1) * 4] + 0.587 * data[(rw - 1) * 4 + 1] + 0.114 * data[(rw - 1) * 4 + 2]) +
                (0.299 * data[(rh - 1) * rw * 4] + 0.587 * data[(rh - 1) * rw * 4 + 1] + 0.114 * data[(rh - 1) * rw * 4 + 2]) +
                (0.299 * data[((rh - 1) * rw + rw - 1) * 4] + 0.587 * data[((rh - 1) * rw + rw - 1) * 4 + 1] + 0.114 * data[((rh - 1) * rw + rw - 1) * 4 + 2])
            ) / 4;
            // Use a bg color that will be classified as background
            const bgVal = cornerSum > 128 ? 255 : 0;

            for (let py = 0; py < rh; py++) {
                for (let px = 0; px < rw; px++) {
                    // Convert sub-image pixel to canvas coords and test against lasso
                    const canvasX = rx + px;
                    const canvasY = ry + py;
                    if (!this._pointInLasso(canvasX, canvasY, lasso)) {
                        const idx = (py * rw + px) * 4;
                        data[idx] = bgVal;
                        data[idx + 1] = bgVal;
                        data[idx + 2] = bgVal;
                    }
                }
            }
        }

        // Trace contour
        const polygon = this._traceContour(imageData, this._uploadThreshold);

        if (!polygon || polygon.length < 3) {
            this._uploadInstruction.textContent = 'No shape detected — adjust threshold and try again';
            // Still show the selection outline
            ctx.strokeStyle = '#ff4444';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            if (this._uploadLasso && this._uploadLasso.length > 3) {
                this._drawLassoOutline(ctx, this._uploadLasso, '#ff4444');
            } else {
                ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
            }
            ctx.setLineDash([]);
            this._uploadState = 'previewing';
            this._uploadRawPolygon = null;
            return;
        }

        this._uploadRawPolygon = polygon;
        this._showTracePreview(polygon, rect);
        this._uploadState = 'previewing';
        this._uploadInstruction.textContent = 'Accept this trace, adjust threshold, or cancel';
    }

    // Ray-casting point-in-polygon for lasso (canvas pixel coords)
    _pointInLasso(x, y, lasso) {
        let inside = false;
        const n = lasso.length;
        for (let i = 0, j = n - 1; i < n; j = i++) {
            const yi = lasso[i].y, yj = lasso[j].y;
            if ((yi > y) !== (yj > y) &&
                x < (lasso[j].x - lasso[i].x) * (y - yi) / (yj - yi) + lasso[i].x) {
                inside = !inside;
            }
        }
        return inside;
    }

    // Helper to draw a lasso outline with a given color
    _drawLassoOutline(ctx, lasso, color) {
        if (lasso.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(lasso[0].x, lasso[0].y);
        for (let i = 1; i < lasso.length; i++) {
            ctx.lineTo(lasso[i].x, lasso[i].y);
        }
        ctx.closePath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    _traceContour(imageData, threshold) {
        const w = imageData.width;
        const h = imageData.height;
        const data = imageData.data;
        const size = w * h;

        // Step 1: Grayscale
        const gray = new Uint8Array(size);
        for (let i = 0; i < size; i++) {
            const i4 = i * 4;
            gray[i] = Math.round(0.299 * data[i4] + 0.587 * data[i4 + 1] + 0.114 * data[i4 + 2]);
        }

        // Step 2: Auto-detect polarity from corner samples
        const corners = [
            gray[0], gray[w - 1],
            gray[(h - 1) * w], gray[(h - 1) * w + w - 1]
        ];
        const avgCorner = (corners[0] + corners[1] + corners[2] + corners[3]) / 4;
        const invert = avgCorner < 128; // dark background → solid = bright

        // Step 3: Binary threshold
        const binary = new Uint8Array(size);
        for (let i = 0; i < size; i++) {
            binary[i] = (invert ? gray[i] > threshold : gray[i] < threshold) ? 1 : 0;
        }

        // Step 4: Flood fill from border to identify background
        // Mark border-connected 0-pixels as background (-1), then fill enclosed 0-gaps
        const visited = new Uint8Array(size); // 0=unvisited, 1=visited
        const queue = [];

        // Seed from all border pixels that are 0
        for (let x = 0; x < w; x++) {
            if (binary[x] === 0 && !visited[x]) { queue.push(x); visited[x] = 1; }
            const bot = (h - 1) * w + x;
            if (binary[bot] === 0 && !visited[bot]) { queue.push(bot); visited[bot] = 1; }
        }
        for (let y = 1; y < h - 1; y++) {
            const left = y * w;
            if (binary[left] === 0 && !visited[left]) { queue.push(left); visited[left] = 1; }
            const right = y * w + w - 1;
            if (binary[right] === 0 && !visited[right]) { queue.push(right); visited[right] = 1; }
        }

        // BFS flood fill
        let head = 0;
        while (head < queue.length) {
            const idx = queue[head++];
            const ix = idx % w;
            const iy = (idx / w) | 0;
            const neighbors = [];
            if (ix > 0) neighbors.push(idx - 1);
            if (ix < w - 1) neighbors.push(idx + 1);
            if (iy > 0) neighbors.push(idx - w);
            if (iy < h - 1) neighbors.push(idx + w);
            for (let k = 0; k < neighbors.length; k++) {
                const n = neighbors[k];
                if (!visited[n] && binary[n] === 0) {
                    visited[n] = 1;
                    queue.push(n);
                }
            }
        }

        // Any unvisited 0-pixel is enclosed — fill it as solid
        for (let i = 0; i < size; i++) {
            if (binary[i] === 0 && !visited[i]) binary[i] = 1;
        }

        // Step 5: Find largest connected component of 1-pixels
        const labels = new Int32Array(size);
        labels.fill(-1);
        let nextLabel = 0;
        const compSizes = [];

        for (let i = 0; i < size; i++) {
            if (binary[i] === 1 && labels[i] === -1) {
                const label = nextLabel++;
                let count = 0;
                const bfsQ = [i];
                labels[i] = label;
                let bfsHead = 0;
                while (bfsHead < bfsQ.length) {
                    const idx = bfsQ[bfsHead++];
                    count++;
                    const ix = idx % w;
                    const iy = (idx / w) | 0;
                    if (ix > 0 && binary[idx - 1] === 1 && labels[idx - 1] === -1) {
                        labels[idx - 1] = label; bfsQ.push(idx - 1);
                    }
                    if (ix < w - 1 && binary[idx + 1] === 1 && labels[idx + 1] === -1) {
                        labels[idx + 1] = label; bfsQ.push(idx + 1);
                    }
                    if (iy > 0 && binary[idx - w] === 1 && labels[idx - w] === -1) {
                        labels[idx - w] = label; bfsQ.push(idx - w);
                    }
                    if (iy < h - 1 && binary[idx + w] === 1 && labels[idx + w] === -1) {
                        labels[idx + w] = label; bfsQ.push(idx + w);
                    }
                }
                compSizes.push(count);
            }
        }

        if (compSizes.length === 0) return null;

        // Find largest component
        let largestLabel = 0;
        let largestSize = compSizes[0];
        for (let i = 1; i < compSizes.length; i++) {
            if (compSizes[i] > largestSize) {
                largestSize = compSizes[i];
                largestLabel = i;
            }
        }

        // Zero out non-largest components
        for (let i = 0; i < size; i++) {
            if (binary[i] === 1 && labels[i] !== largestLabel) binary[i] = 0;
        }

        // Step 6: Moore neighborhood contour tracing
        // Find the topmost-leftmost solid pixel (guaranteed on boundary)
        let startPx = -1, startPy = -1;
        for (let sy = 0; sy < h; sy++) {
            for (let sx = 0; sx < w; sx++) {
                if (binary[sy * w + sx] === 1) {
                    startPx = sx;
                    startPy = sy;
                    break;
                }
            }
            if (startPx !== -1) break;
        }

        if (startPx === -1) return null;

        // Moore neighborhood: 8 dirs clockwise starting from right
        // 0=right, 1=down-right, 2=down, 3=down-left, 4=left, 5=up-left, 6=up, 7=up-right
        const mdx = [1, 1, 0, -1, -1, -1, 0, 1];
        const mdy = [0, 1, 1, 1, 0, -1, -1, -1];

        const contour = [];
        let cx = startPx, cy = startPy;
        // Start pixel is leftmost in its row, so pixel to the left is background
        // backDir points toward that background pixel: direction 4 (left)
        let backDir = 4;
        const maxIter = size * 4;

        for (let iter = 0; iter < maxIter; iter++) {
            contour.push({ x: cx, y: cy });

            // Search clockwise starting from (backDir + 1) % 8
            const searchStart = (backDir + 1) % 8;
            let found = false;

            for (let k = 0; k < 8; k++) {
                const d = (searchStart + k) % 8;
                const nx = cx + mdx[d];
                const ny = cy + mdy[d];

                if (nx >= 0 && nx < w && ny >= 0 && ny < h && binary[ny * w + nx] === 1) {
                    backDir = (d + 4) % 8; // opposite of movement direction
                    cx = nx;
                    cy = ny;
                    found = true;
                    break;
                }
            }

            if (!found) break; // isolated pixel

            // Stop when we return to start
            if (cx === startPx && cy === startPy && contour.length > 2) break;
        }

        if (contour.length < 3) return null;

        // Step 7: Simplify with Ramer-Douglas-Peucker
        const simplified = this._simplifyPolygon(contour, 1.5);
        return simplified.length >= 3 ? simplified : null;
    }

    _simplifyPolygon(points, epsilon) {
        if (points.length <= 2) return points;

        let maxDist = 0;
        let maxIdx = 0;
        const start = points[0];
        const end = points[points.length - 1];
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const lineLenSq = dx * dx + dy * dy;

        for (let i = 1; i < points.length - 1; i++) {
            let dist;
            if (lineLenSq === 0) {
                const ddx = points[i].x - start.x;
                const ddy = points[i].y - start.y;
                dist = Math.sqrt(ddx * ddx + ddy * ddy);
            } else {
                // Perpendicular distance
                const cross = Math.abs((points[i].x - start.x) * dy - (points[i].y - start.y) * dx);
                dist = cross / Math.sqrt(lineLenSq);
            }
            if (dist > maxDist) {
                maxDist = dist;
                maxIdx = i;
            }
        }

        if (maxDist > epsilon) {
            const left = this._simplifyPolygon(points.slice(0, maxIdx + 1), epsilon);
            const right = this._simplifyPolygon(points.slice(maxIdx), epsilon);
            return left.slice(0, -1).concat(right);
        }
        return [start, end];
    }

    _showTracePreview(polygon, rect) {
        const canvas = this._uploadCanvas;
        const ctx = canvas.getContext('2d');

        // Redraw image
        ctx.drawImage(this._uploadImage, 0, 0, canvas.width, canvas.height);

        // Dim area outside selection
        if (this._uploadLasso && this._uploadLasso.length > 3) {
            // Lasso mode: dim everything, then clear inside the lasso
            ctx.save();
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            // Cut out inside of lasso
            ctx.globalCompositeOperation = 'destination-out';
            ctx.beginPath();
            ctx.moveTo(this._uploadLasso[0].x, this._uploadLasso[0].y);
            for (let i = 1; i < this._uploadLasso.length; i++) {
                ctx.lineTo(this._uploadLasso[i].x, this._uploadLasso[i].y);
            }
            ctx.closePath();
            ctx.fill();
            ctx.restore();
            // Redraw image inside lasso (composite)
            ctx.save();
            ctx.globalCompositeOperation = 'destination-over';
            ctx.drawImage(this._uploadImage, 0, 0, canvas.width, canvas.height);
            ctx.restore();
        } else {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.fillRect(0, 0, canvas.width, rect.y);
            ctx.fillRect(0, rect.y + rect.h, canvas.width, canvas.height - rect.y - rect.h);
            ctx.fillRect(0, rect.y, rect.x, rect.h);
            ctx.fillRect(rect.x + rect.w, rect.y, canvas.width - rect.x - rect.w, rect.h);
        }

        // Draw traced polygon (filled semi-transparent green + bright outline)
        ctx.beginPath();
        ctx.moveTo(rect.x + polygon[0].x, rect.y + polygon[0].y);
        for (let i = 1; i < polygon.length; i++) {
            ctx.lineTo(rect.x + polygon[i].x, rect.y + polygon[i].y);
        }
        ctx.closePath();

        ctx.fillStyle = 'rgba(76, 175, 80, 0.3)';
        ctx.fill();
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Selection outline
        if (this._uploadLasso && this._uploadLasso.length > 3) {
            this._drawLassoOutline(ctx, this._uploadLasso, 'rgba(255,255,255,0.5)');
        } else {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
            ctx.setLineDash([]);
        }
    }

    _normalizePolygon(polygon, rect) {
        // Find bounding box of polygon (in pixel coords within selection rect)
        let pxMin = Infinity, pxMax = -Infinity;
        let pyMin = Infinity, pyMax = -Infinity;
        for (let i = 0; i < polygon.length; i++) {
            if (polygon[i].x < pxMin) pxMin = polygon[i].x;
            if (polygon[i].x > pxMax) pxMax = polygon[i].x;
            if (polygon[i].y < pyMin) pyMin = polygon[i].y;
            if (polygon[i].y > pyMax) pyMax = polygon[i].y;
        }

        const cx = (pxMin + pxMax) * 0.5;
        const cy = (pyMin + pyMax) * 0.5;
        const spanX = pxMax - pxMin;
        const spanY = pyMax - pyMin;
        const maxSpan = Math.max(spanX, spanY);

        if (maxSpan < 1) return null;

        const scale = this.L / maxSpan;

        // Transform to body frame: center on centroid, scale to L, flip Y
        const result = [];
        for (let i = 0; i < polygon.length; i++) {
            result.push({
                x: (polygon[i].x - cx) * scale,
                y: -(polygon[i].y - cy) * scale  // flip Y (image Y-down → sim Y-up)
            });
        }
        return result;
    }

    _pointInPolygon(x, y, polygon) {
        let inside = false;
        const n = polygon.length;
        for (let i = 0, j = n - 1; i < n; j = i++) {
            const xi = polygon[i].x, yi = polygon[i].y;
            const xj = polygon[j].x, yj = polygon[j].y;
            if ((yi > y) !== (yj > y) &&
                x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
        }
        return inside;
    }

    _acceptTrace() {
        if (!this._uploadRawPolygon || !this._uploadRect) return;

        // Normalize polygon to body frame
        const normalized = this._normalizePolygon(this._uploadRawPolygon, this._uploadRect);
        if (!normalized || normalized.length < 3) {
            this._uploadInstruction.textContent = 'Could not normalize shape — try again';
            return;
        }

        this.customPolygon = normalized;

        // Enable dropdown option and select it
        const option = document.getElementById('customBodyOption');
        option.disabled = false;
        document.getElementById('bodyShape').value = 'custom';
        this.bodyType = 'custom';

        // Generate body mask
        this.generateBody();

        // Hide overlay and resume
        this._hideUploadOverlay();
    }

    _cancelUpload() {
        this._hideUploadOverlay();
    }

    _hideUploadOverlay() {
        this._uploadOverlay.classList.remove('visible');
        this._uploadState = 'idle';
        this._uploadImage = null;
        this._uploadDragging = false;

        // Resume simulation
        if (!this.running) {
            this.running = true;
            this.lastTime = performance.now();
            this.animate();
        }
    }

    // ── Controls ───────────────────────────────────────────────────────────

    setupControls() {
        // Angle
        const angleSlider = document.getElementById('angle');
        const angleValue = document.getElementById('angle-value');

        angleSlider.addEventListener('input', (e) => {
            this.angle = -parseFloat(e.target.value);
            angleValue.textContent = parseFloat(e.target.value) + '\u00B0';
            this.generateBody();
        });

        // Body shape
        document.getElementById('bodyShape').addEventListener('change', (e) => {
            if (e.target.value === 'custom' && !this.customPolygon) {
                // No polygon yet — trigger upload
                document.getElementById('bodyFileInput').click();
                e.target.value = this.bodyType; // revert dropdown
                return;
            }
            this.bodyType = e.target.value;
            this.generateBody();
        });

        // Add Ground checkbox
        const groundCheckbox = document.getElementById('addGround');
        groundCheckbox.addEventListener('change', (e) => {
            this.addGround = e.target.checked;
            // Lock/unlock angle slider
            angleSlider.disabled = this.addGround;
            this.generateBody();
            // Re-initialize flow field to avoid unphysical transients
            this.initializeFields();
            this.generateBody();
            this.time = 0.0;
            this.liftHistory.length = 0;
            this.dragHistory.length = 0;
        });

        // Visualization mode
        document.querySelectorAll('[data-viz]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-viz]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.vizMode = e.target.dataset.viz;
            });
        });
    }

    animate() {
        if (!this.running) return;

        // Run physics steps
        for (let i = 0; i < this.stepsPerFrame; i++) {
            this.step();
        }

        // Render (GPU does the heavy lifting via WebGL)
        this.render();
        this.frameCount++;

        // Temporary debug overlay
        if (!this._dbgDiv) {
            this._dbgDiv = document.createElement('div');
            this._dbgDiv.style.cssText = 'position:fixed;top:10px;left:10px;color:#0f0;font:bold 12px monospace;z-index:999;background:rgba(0,0,0,0.8);padding:8px;border-radius:4px;pointer-events:none;white-space:pre';
            document.body.appendChild(this._dbgDiv);
        }
        if (this.frameCount % 30 === 0) {
            // Count solid cells in mask
            let solidCount = 0;
            const m = this.mask;
            for (let k = 0; k < m.length; k++) {
                if (m[k] < 0.5) solidCount++;
            }
            this._dbgDiv.textContent =
                'canvas: ' + this.canvas.width + 'x' + this.canvas.height +
                '\nview X: [' + this.viewXMin.toFixed(2) + ', ' + this.viewXMax.toFixed(2) + ']' +
                '\nview Y: [' + this.viewYMin.toFixed(2) + ', ' + this.viewYMax.toFixed(2) + ']' +
                '\nWebGL: ' + this.useWebGL +
                '\nfloatLinear: ' + this.hasFloatLinear +
                '\ngrid: ' + this.Nx + 'x' + this.Ny +
                '\nsolid cells: ' + solidCount + '/' + m.length +
                '\nground: ' + this.addGround +
                '\nbody: ' + this.bodyType;
        }

        requestAnimationFrame(this._boundAnimate);
    }
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', () => {
    window.sim = new AeroSlicer();
});
