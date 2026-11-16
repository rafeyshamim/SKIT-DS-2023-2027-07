import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// 3D stage (dark scene + grid). Accepts either
//   { vertices, faces }                         -> one model
//   { parts: [{ name, color, solid, vertices, faces }] } -> several layers

const LEGACY_PART = { name: "Reconstructed model", color: "#d98a86", solid: false };

export function getParts(mesh) {
  if (!mesh) return [];
  if (Array.isArray(mesh.parts)) return mesh.parts;
  if (mesh.vertices && mesh.faces) {
    return [{ ...LEGACY_PART, vertices: mesh.vertices, faces: mesh.faces }];
  }
  return [];
}

function getBounds(parts) {
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  parts.forEach((p) => p.vertices.forEach((q) => box.expandByPoint(v.set(q[0], q[1], q[2]))));
  return box;
}

function buildGeometry(part, center, k) {
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(part.vertices.length * 3);
  part.vertices.forEach((q, i) => {
    pos[i * 3] = (q[0] - center.x) * k;
    pos[i * 3 + 1] = (q[1] - center.y) * k;
    pos[i * 3 + 2] = (q[2] - center.z) * k;
  });
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(part.faces.flat());
  g.computeVertexNormals();
  return g;
}

function buildMaterial(part) {
  if (part.solid) {
    return new THREE.MeshStandardMaterial({
      color: part.color,
      roughness: 0.5,
      emissive: "#6b4a3a",
      emissiveIntensity: 0.35,
      side: THREE.DoubleSide,
    });
  }
  return new THREE.MeshPhysicalMaterial({
    color: part.color,
    roughness: 0.55,
    clearcoat: 0.35,
    clearcoatRoughness: 0.4,
    transparent: true,
    opacity: 0.85,
  });
}

const ModelViewer = forwardRef(function ModelViewer(
  { mesh, opacity, autoRotate, wireframe, layers },
  ref
) {
  const mountRef = useRef(null);
  const st = useRef({});

  useImperativeHandle(ref, () => ({
    zoomIn: () => zoom(0.85),
    zoomOut: () => zoom(1.18),
    reset: () => setView("front"),
    setView: (v) => setView(v),
    screenshot: () => {
      const { renderer, scene, camera } = st.current;
      renderer.render(scene, camera);
      return renderer.domElement.toDataURL("image/png");
    },
  }));

  function zoom(f) {
    const { camera, controls } = st.current;
    const dir = camera.position.clone().sub(controls.target).multiplyScalar(f);
    const len = THREE.MathUtils.clamp(dir.length(), 3, 16);
    camera.position.copy(controls.target).add(dir.setLength(len));
  }

  function setView(v) {
    const { camera, controls } = st.current;
    const d = 9;
    const spots = { front: [0, 1.2, d], side: [d, 1.2, 0], top: [0, d, 0.01] };
    camera.position.set(...(spots[v] || spots.front));
    controls.target.set(0, 0, 0);
    controls.update();
  }

  useEffect(() => {
    const mount = mountRef.current;
    const w = mount.clientWidth;
    const h = mount.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0d1519");
    const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight("#ffffff", "#22343b", 0.95));
    const key = new THREE.DirectionalLight("#ffffff", 1.1);
    key.position.set(4, 6, 7);
    scene.add(key);
    const rim = new THREE.DirectionalLight("#6fd0c4", 0.7);
    rim.position.set(-6, 3, -6);
    scene.add(rim);

    const grid = new THREE.GridHelper(24, 48, "#1f5a55", "#16313a");
    grid.position.y = -2.6;
    scene.add(grid);

    const parts = getParts(mesh);
    const built = [];
    if (parts.length) {
      const box = getBounds(parts);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const k = 4.2 / Math.max(size.x, size.y, size.z);
      parts.forEach((p) => {
        const m = new THREE.Mesh(buildGeometry(p, center, k), buildMaterial(p));
        scene.add(m);
        built.push({ name: p.name, mesh: m, solid: !!p.solid });
      });
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 3;
    controls.maxDistance = 16;
    controls.autoRotateSpeed = 1.6;

    st.current = { scene, camera, renderer, controls, parts: built };
    setView("front");

    let raf;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    const onResize = () => {
      const nw = mount.clientWidth;
      const nh = mount.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        o.geometry?.dispose();
        o.material?.dispose?.();
      });
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mesh]);

  useEffect(() => {
    const s = st.current;
    if (!s.scene) return;
    const list = s.parts || [];
    list.forEach(({ name, mesh: m, solid }) => {
      if (!solid) {
        m.material.opacity = opacity;
        m.material.wireframe = wireframe;
      }
      m.visible = layers?.[name] !== false;
    });
    s.controls.autoRotate = autoRotate && list.length > 0;
  }, [mesh, opacity, autoRotate, wireframe, layers]);

  return <div className="viewport" ref={mountRef} />;
});

export default ModelViewer;