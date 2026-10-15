import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Empty 3D stage (dark scene + grid). When the backend returns a mesh
// ({ vertices: number[][], faces: number[][] }) it is rendered here.

function buildFromMesh(m) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(m.vertices.flat(), 3));
  g.setIndex(m.faces.flat());
  g.computeBoundingBox();
  const c = g.boundingBox.getCenter(new THREE.Vector3());
  const sz = g.boundingBox.getSize(new THREE.Vector3());
  g.translate(-c.x, -c.y, -c.z);
  const k = 4.2 / Math.max(sz.x, sz.y, sz.z);
  g.scale(k, k, k);
  g.computeVertexNormals();
  return g;
}

const ModelViewer = forwardRef(function ModelViewer(
  { mesh, opacity, autoRotate, wireframe, showModel },
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

    scene.add(new THREE.HemisphereLight("#ffffff", "#22343b", 1.1));
    const key = new THREE.DirectionalLight("#ffffff", 1.2);
    key.position.set(4, 6, 6);
    scene.add(key);

    const grid = new THREE.GridHelper(24, 48, "#1f5a55", "#16313a");
    grid.position.y = -2.6;
    scene.add(grid);

    let model = null;
    if (mesh?.vertices && mesh?.faces) {
      model = new THREE.Mesh(
        buildFromMesh(mesh),
        new THREE.MeshPhysicalMaterial({
          color: "#d98a86",
          roughness: 0.5,
          clearcoat: 0.3,
          transparent: true,
          opacity: 0.85,
          side: THREE.DoubleSide,
        })
      );
      scene.add(model);
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 3;
    controls.maxDistance = 16;
    controls.autoRotateSpeed = 1.6;

    st.current = { scene, camera, renderer, controls, model };
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
    if (s.model) {
      s.model.material.opacity = opacity;
      s.model.material.wireframe = wireframe;
      s.model.visible = showModel;
    }
    s.controls.autoRotate = autoRotate && !!s.model;
  }, [mesh, opacity, autoRotate, wireframe, showModel]);

  return <div className="viewport" ref={mountRef} />;
});

export default ModelViewer;
