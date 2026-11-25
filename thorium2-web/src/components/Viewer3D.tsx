"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface MeshElement {
  id: number;
  type: string;
  pid: number;
  faces: number[][];
  medium: number;
  mass: number;
  volume: number;
  area?: number;
  thickness?: number;
  optics: number[];
  t0: number;
  heat_source: number;
  fixed: boolean;
}

export interface Mesh {
  nodes: [number, number, number][];
  elements: MeshElement[];
  time: { start: number; end: number; step: number };
  photons: number;
  warnings: string[];
}

export type DrawMode = "contour" | "plain" | "wireframe";

/** Camera direction (from the model towards the camera); `n` changes to re-apply the same view. */
export interface View {
  dir: [number, number, number];
  n: number;
}

interface Props {
  mesh: Mesh;
  view: View;
  /** rgb per element, 0..1 */
  colors: Float32Array;
  mode: DrawMode;
  selected: number | null;
  onPick: (element: number | null, x: number, y: number) => void;
}

function setCamera(s: { camera: THREE.PerspectiveCamera; controls: OrbitControls; sphere: THREE.Sphere }, dir: [number, number, number]) {
  const d = new THREE.Vector3(...dir).normalize();
  // Up: +Z, unless we look along Z.
  const up = Math.abs(d.z) > 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
  s.camera.up.copy(up);
  s.camera.position.copy(s.sphere.center).add(d.multiplyScalar(s.sphere.radius * 3.8));
  s.controls.target.copy(s.sphere.center);
  s.controls.update();
}

/** Model window: elements coloured by value, white contours, picking by click (as in version 1). */
export default function Viewer3D({ mesh, view, colors, mode, selected, onPick }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    surface: THREE.Mesh;
    edges: THREE.LineSegments;
    triElement: Int32Array;
    elementTris: number[][];
    highlight: THREE.LineSegments;
    sphere: THREE.Sphere;
  } | null>(null);
  const pickRef = useRef(onPick);
  useEffect(() => {
    pickRef.current = onPick;
  }, [onPick]);

  // Build the scene once per model.
  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x000000);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();

    // Triangles (fan around the polygon centre for quads, like the solver does).
    const pos: number[] = [];
    const triElement: number[] = [];
    const elementTris: number[][] = mesh.elements.map(() => []);
    const edgePos: number[] = [];
    const P = (i: number) => mesh.nodes[i];
    mesh.elements.forEach((e, ei) => {
      for (const f of e.faces) {
        const pts = f.map(P);
        const tris: [number, number, number][][] = [];
        if (pts.length === 3) tris.push([pts[0], pts[1], pts[2]]);
        else {
          const c: [number, number, number] = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p[k], 0) / pts.length) as [number, number, number];
          for (let k = 0; k < pts.length; k++) tris.push([pts[k], pts[(k + 1) % pts.length], c]);
        }
        for (const t of tris) {
          elementTris[ei].push(triElement.length);
          triElement.push(ei);
          for (const p of t) pos.push(...p);
        }
        for (let k = 0; k < pts.length; k++) edgePos.push(...pts[k], ...pts[(k + 1) % pts.length]);
      }
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(pos.length), 3));
    const surface = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }),
    );
    scene.add(surface);
    const edgeGeom = new THREE.BufferGeometry();
    edgeGeom.setAttribute("position", new THREE.Float32BufferAttribute(edgePos, 3));
    const edges = new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({ color: 0xffffff }));
    scene.add(edges);
    const highlight = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff00ff }));
    scene.add(highlight);

    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere ?? new THREE.Sphere(new THREE.Vector3(), 1);
    const camera = new THREE.PerspectiveCamera(35, 1, sphere.radius / 100, sphere.radius * 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(sphere.center);
    controls.update();

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    // Pick on a click that did not rotate the view.
    let down: { x: number; y: number } | null = null;
    const onDown = (ev: PointerEvent) => (down = { x: ev.clientX, y: ev.clientY });
    const onUp = (ev: PointerEvent) => {
      if (!down || Math.hypot(ev.clientX - down.x, ev.clientY - down.y) > 4) return;
      const r = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      const ray = new THREE.Raycaster();
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObject(surface)[0];
      pickRef.current(hit && hit.faceIndex != null ? triElement[hit.faceIndex] : null, ev.clientX - r.left, ev.clientY - r.top);
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onUp);
    renderer.domElement.addEventListener("contextmenu", (ev) => ev.preventDefault());

    let frame = 0;
    const loop = () => {
      frame = requestAnimationFrame(loop);
      controls.update();
      renderer.render(scene, camera);
    };
    loop();
    state.current = { renderer, scene, camera, controls, surface, edges, triElement: Int32Array.from(triElement), elementTris, highlight, sphere };
    setCamera(state.current, [1, 1, 1]);
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      geometry.dispose();
      edgeGeom.dispose();
      el.removeChild(renderer.domElement);
      state.current = null;
    };
  }, [mesh]);

  // Standard views.
  useEffect(() => {
    if (state.current && view.n > 0) setCamera(state.current, view.dir);
  }, [view]);

  // Colours.
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    const attr = s.surface.geometry.getAttribute("color") as THREE.BufferAttribute;
    const a = attr.array as Float32Array;
    for (let t = 0; t < s.triElement.length; t++) {
      const e = s.triElement[t];
      for (let v = 0; v < 3; v++) a.set([colors[3 * e], colors[3 * e + 1], colors[3 * e + 2]], 9 * t + 3 * v);
    }
    attr.needsUpdate = true;
  }, [colors, mesh]);

  // Drawing mode and the selected element.
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    s.surface.visible = mode !== "wireframe";
    s.edges.visible = mode !== "plain";
    (s.edges.material as THREE.LineBasicMaterial).color.set(mode === "wireframe" ? 0x9fd0ff : 0xffffff);
    const pts: number[] = [];
    if (selected != null)
      for (const f of mesh.elements[selected].faces)
        for (let k = 0; k < f.length; k++) pts.push(...mesh.nodes[f[k]], ...mesh.nodes[f[(k + 1) % f.length]]);
    s.highlight.geometry.dispose();
    s.highlight.geometry = new THREE.BufferGeometry();
    s.highlight.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  }, [mode, selected, mesh]);

  return <div ref={host} className="h-full w-full" data-testid="viewer" />;
}
