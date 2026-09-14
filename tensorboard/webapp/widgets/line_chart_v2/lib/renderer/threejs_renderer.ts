/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as THREE from 'three';
import {hsl, interpolateHsl} from '../../../../third_party/d3';
import {Point, Polyline, Rect} from '../internal_types';
import {ThreeCoordinator} from '../threejs_coordinator';
import {ChartUtils} from '../utils';
import {
  CirclePaintOption,
  LinePaintOption,
  ObjectRenderer,
  TrapezoidPaintOption,
  TrianglePaintOption,
} from './renderer_types';

function createOpacityAdjustedColor(
  baseColorHex: string,
  hex: string,
  opacity: number
): THREE.Color {
  if (opacity === 1) return new THREE.Color(hex);
  const newD3Color = hsl(hex);
  if (!newD3Color) {
    throw new Error(`d3 failed to recognize the color: ${hex}`);
  }
  return new THREE.Color(interpolateHsl(newD3Color, baseColorHex)(1 - opacity));
}

enum CacheType {
  CIRCLE,
  LINE,
  TRIANGLE,
  TRAPEZOID,
}

interface LineCacheValue {
  type: CacheType.LINE;
  obj3d: THREE.Mesh;
  data: Polyline;
  width: number;
}

interface TriangleCacheValue {
  type: CacheType.TRIANGLE;
  obj3d: THREE.Mesh;
  data: Point;
}

interface CircleCacheValue {
  type: CacheType.CIRCLE;
  obj3d: THREE.Mesh;
  data: {loc: Point; radius: number};
}

interface TrapezoidCacheValue {
  type: CacheType.TRAPEZOID;
  obj3d: THREE.Mesh;
  data: [Point, Point];
}

type CacheValue =
  | LineCacheValue
  | TriangleCacheValue
  | CircleCacheValue
  | TrapezoidCacheValue;

/**
 * Updates BufferGeometry with Float32Array that denotes flattened array of Vec2
 * (<x, y>) representing points that form a connected set of line segments.
 */
function updatePolylineGeometry(
  geometry: THREE.BufferGeometry,
  flatVec2: Float32Array
) {
  const numVertices = flatVec2.length / 2;
  let positionAttributes = geometry.attributes[
    'position'
  ] as THREE.BufferAttribute;
  if (!positionAttributes || positionAttributes.count !== numVertices * 3) {
    positionAttributes = new THREE.BufferAttribute(
      new Float32Array(numVertices * 3),
      3
    );
    geometry.setAttribute('position', positionAttributes);
  }
  const values = positionAttributes.array as number[];
  for (let index = 0; index < numVertices; index++) {
    values[index * 3] = flatVec2[index * 2];
    values[index * 3 + 1] = flatVec2[index * 2 + 1];
    // z-value (index * 3 + 2) is implicitly 0 (they are set when initializing
    // Float32Array).
  }
  positionAttributes.needsUpdate = true;
  geometry.setDrawRange(0, numVertices * 3);
  // Need to update the bounding sphere so renderer does not skip rendering
  // this object because it is outside of the camera viewpoint (frustum).
  geometry.computeBoundingSphere();
}

/**
 * Updates BufferGeometry with Float32Array that denotes flattened array of Vec2
 * (<x, y>) representing points that form a connected set of line segments.
 * Unlike the simpler `updatePolylineGeometry`, this variant constructs more
 * vertices to support line thickness.
 *
 * This custom logic handles line thickness, since the 'linewidth' property of
 * THREE.LineBasicMaterial is ignored [1]. Each line segment is a rectangle
 * that is split into 2 triangles [A->B->C] and [C->B->D]. Each triangle has
 * 3 coordinates (x, y, z).
 *
 * Assuming a line segment is as follows (thickness = distance from A to B):
 *              A             C
 *              |             |
 * (startPoint) |-------------| (endPoint)
 *              |             |
 *              B             D
 *
 * The renderer will draw 2 triangles:
 *
 *    ^   A----C
 *    |   |   /|
 * dy |   | /  |
 *    |   |/   |
 *    v   B----D
 *
 *        <---->
 *          dx
 *
 * [1] https://github.com/mrdoob/three.js/issues/14627
 */
function updateThickPolylineGeometry(
  geometry: THREE.BufferGeometry,
  flatVec2: Float32Array,
  thickness: number
) {
  const numSegments = Math.max(flatVec2.length / 2 - 1, 0);
  const numVertices = numSegments * 2 * 3;
  const numCoordinates = numVertices * 3;
  let positionAttributes = geometry.attributes[
    'position'
  ] as THREE.BufferAttribute;
  if (!positionAttributes || positionAttributes.count !== numVertices) {
    positionAttributes = new THREE.BufferAttribute(
      new Float32Array(numCoordinates),
      3
    );
    geometry.setAttribute('position', positionAttributes);
  }

  const values = positionAttributes.array as Float32Array;
  const halfThickness = thickness / 2;
  // Extent of the vertices written below, tracked as they are written.
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < numSegments; i++) {
    const x1 = flatVec2[2 * i];
    const y1 = flatVec2[2 * i + 1];
    const x2 = flatVec2[2 * i + 2];
    const y2 = flatVec2[2 * i + 3];
    // The normal 90 degrees counterclockwise of the segment, with half the
    // thickness as its length. Written out rather than built from vectors:
    // every visible series rebuilds its geometry on each frame, and vector
    // objects per segment dominated that work.
    const segmentX = x2 - x1;
    const segmentY = y2 - y1;
    const segmentLength =
      Math.sqrt(segmentX * segmentX + segmentY * segmentY) || 1;
    const normalX = (-segmentY / segmentLength) * halfThickness;
    const normalY = (segmentX / segmentLength) * halfThickness;

    const ax = x1 + normalX;
    const ay = y1 + normalY;
    const bx = x1 - normalX;
    const by = y1 - normalY;
    const cx = x2 + normalX;
    const cy = y2 + normalY;
    const dx = x2 - normalX;
    const dy = y2 - normalY;

    // Keep each face's vertices in counterclockwise order, to ensure normals
    // point outwards from the screen. Coordinate z is left at zero.
    const offset = i * 18;
    // A->B->C triangle.
    values[offset] = ax;
    values[offset + 1] = ay;
    values[offset + 3] = bx;
    values[offset + 4] = by;
    values[offset + 6] = cx;
    values[offset + 7] = cy;
    // C->B->D triangle.
    values[offset + 9] = cx;
    values[offset + 10] = cy;
    values[offset + 12] = bx;
    values[offset + 13] = by;
    values[offset + 15] = dx;
    values[offset + 16] = dy;

    // A segment with a non-finite endpoint always degenerates its normal to
    // NaN, so all six of its vertices carry a NaN component and rasterize to
    // nothing. Leaving such a segment out of the extent keeps the sphere
    // finite while still bounding every vertex that can draw.
    if (
      Number.isFinite(x1) &&
      Number.isFinite(y1) &&
      Number.isFinite(x2) &&
      Number.isFinite(y2)
    ) {
      if (ax < minX) minX = ax;
      if (bx < minX) minX = bx;
      if (cx < minX) minX = cx;
      if (dx < minX) minX = dx;
      if (ax > maxX) maxX = ax;
      if (bx > maxX) maxX = bx;
      if (cx > maxX) maxX = cx;
      if (dx > maxX) maxX = dx;
      if (ay < minY) minY = ay;
      if (by < minY) minY = by;
      if (cy < minY) minY = cy;
      if (dy < minY) minY = dy;
      if (ay > maxY) maxY = ay;
      if (by > maxY) maxY = by;
      if (cy > maxY) maxY = cy;
      if (dy > maxY) maxY = dy;
    }
  }

  positionAttributes.needsUpdate = true;
  geometry.setDrawRange(0, numCoordinates);
  // Need to update the bounding sphere so renderer does not skip rendering
  // this object because it is outside of the camera viewpoint (frustum). The
  // extent is accumulated in the loop above instead of by
  // `computeBoundingSphere`, which walks every expanded vertex twice more; at
  // hundreds of series redrawn per frame those passes are the dominant cost.
  // The extent covers exactly the vertices written above, which is exactly the
  // draw range set here, and is rebuilt from scratch on every update, so an
  // over-allocated buffer contributes no stale vertices.
  // The sentinels survive only when no segment contributed a finite extent:
  // an empty polyline, a single point, or nothing but non-finite coordinates.
  // Those collapse to a zero-radius sphere at the origin, as
  // `computeBoundingSphere` does for the first two; for the third it reports a
  // NaN radius to the console, and such a line draws nothing either way.
  const isBounded = minX <= maxX;
  const centerX = isBounded ? (minX + maxX) / 2 : 0;
  const centerY = isBounded ? (minY + maxY) / 2 : 0;
  const spanX = isBounded ? maxX - minX : 0;
  const spanY = isBounded ? maxY - minY : 0;
  // Half the box diagonal: the smallest radius about the box center that is
  // guaranteed to contain every corner.
  const radius = Math.sqrt(spanX * spanX + spanY * spanY) / 2;
  // Coordinate z is always zero, so the sphere is centered on the xy plane.
  if (geometry.boundingSphere) {
    geometry.boundingSphere.center.set(centerX, centerY, 0);
    geometry.boundingSphere.radius = radius;
  } else {
    geometry.boundingSphere = new THREE.Sphere(
      new THREE.Vector3(centerX, centerY, 0),
      radius
    );
  }
}

/**
 * Updates an THREE.Object3D like object with geometry and material. Returns true if
 * geometry is updated (i.e., updateGeometry callback is invoked) and returns false
 * otherwise. It is possible that we minimally update the material without updating the
 * geometry.
 */
function updateObject(
  baseColorHex: string,
  object: THREE.Mesh,
  updateGeometry: (geometry: THREE.BufferGeometry) => THREE.BufferGeometry,
  materialOption: {visible: boolean; color: string; opacity?: number}
): boolean {
  const {visible, color, opacity} = materialOption;

  if (Array.isArray(object.material)) {
    throw new Error('Invariant error: only expect one material on an object');
  }

  const material = object.material as THREE.MeshBasicMaterial;
  if (material.visible !== visible) {
    material.visible = visible;
    material.needsUpdate = true;
  }

  if (!visible) return false;

  const newColor = createOpacityAdjustedColor(
    baseColorHex,
    color,
    opacity ?? 1
  );

  const newGeom = updateGeometry(object.geometry as THREE.BufferGeometry);
  if (object.geometry !== newGeom) {
    object.geometry = newGeom;
  }

  const currentColor = material.color;
  if (!currentColor.equals(newColor)) {
    material.color.set(newColor);
    material.needsUpdate = true;
  }

  return true;
}

const ThreeWrapper = {
  createScene: () => {
    return new THREE.Scene();
  },
};

export class ThreeRenderer implements ObjectRenderer<CacheValue> {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = ThreeWrapper.createScene();
  private backgroundColor: string = '#fff';

  constructor(
    private readonly canvas: HTMLCanvasElement | OffscreenCanvas,
    private readonly coordinator: ThreeCoordinator,
    devicePixelRatio: number,
    private readonly onContextLost?: EventListener
  ) {
    if (
      typeof OffscreenCanvas !== 'undefined' &&
      canvas instanceof OffscreenCanvas
    ) {
      // THREE.js requires a style object which OffscreenCanvas lacks.
      // Only the canvas type matters here. The host checks WebGL2 support
      // before choosing WorkerChart; probing again would create a throwaway
      // WebGL context in every pooled worker before its first real renderer.
      const styleless = canvas as unknown as {style?: object};
      styleless.style = styleless.style ?? {};
    }
    // WebGL contexts may be abandoned by the browser if too many contexts are
    // created on the same page.
    if (onContextLost) {
      canvas.addEventListener('webglcontextlost', onContextLost);
    }

    this.renderer = new THREE.WebGLRenderer({
      canvas: canvas as HTMLCanvasElement,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(devicePixelRatio);
  }

  onResize(rect: Rect) {
    this.renderer.setSize(rect.width, rect.height);
  }

  destroyObject(cacheValue: CacheValue): void {
    this.releaseObject(cacheValue.obj3d);
  }

  private releaseObject(obj3d: THREE.Object3D): void {
    this.scene.remove(obj3d);

    if (obj3d instanceof THREE.Mesh) {
      obj3d.geometry.dispose();
      const materials = Array.isArray(obj3d.material)
        ? obj3d.material
        : [obj3d.material];
      for (const material of materials) {
        material.dispose();
      }
    }
  }

  setUseDarkMode(useDarkMode: boolean): void {
    this.backgroundColor = useDarkMode ? '#303030' : '#fff';
    // Normally, we should invoke `setClearColor` but we are using
    // `alpha: false` mode in threejs (transparent) so it does not matter.
  }

  createOrUpdateLineObject(
    cachedLine: LineCacheValue | null,
    polyline: Polyline,
    paintOpt: LinePaintOption
  ): LineCacheValue | null {
    // If a line is not cached and is not even visible, skip drawing line.
    if (!cachedLine && !paintOpt.visible) return null;

    const {visible, width} = paintOpt;

    if (!cachedLine) {
      const newColor = createOpacityAdjustedColor(
        this.backgroundColor,
        paintOpt.color,
        paintOpt.opacity ?? 1
      );
      const geometry = new THREE.BufferGeometry();
      const material = new THREE.LineBasicMaterial({color: newColor});
      const line = new THREE.Mesh(geometry, material);
      material.visible = visible;
      updateThickPolylineGeometry(geometry, polyline, width);
      this.scene.add(line);
      return {type: CacheType.LINE, data: polyline, obj3d: line, width};
    }

    const {data: prevPolyline, obj3d: line, width: prevWidth} = cachedLine;
    const geomUpdated = updateObject(
      this.backgroundColor,
      line,
      (geometry) => {
        if (
          width !== prevWidth ||
          !prevPolyline ||
          !ChartUtils.arePolylinesEqual(prevPolyline, polyline)
        ) {
          updateThickPolylineGeometry(geometry, polyline, width);
        }
        return geometry;
      },
      paintOpt
    );
    if (!geomUpdated) return cachedLine;

    return {
      type: CacheType.LINE,
      data: polyline,
      obj3d: line,
      width,
    };
  }

  private createMesh(
    geometry: THREE.BufferGeometry,
    materialOption: {visible: boolean; color: string; opacity?: number}
  ): THREE.Mesh | null {
    if (!materialOption.visible) return null;

    const {visible, color, opacity} = materialOption;
    const newColor = createOpacityAdjustedColor(
      this.backgroundColor,
      color,
      opacity ?? 1
    );
    const material = new THREE.MeshBasicMaterial({color: newColor, visible});
    return new THREE.Mesh(geometry, material);
  }

  createOrUpdateTriangleObject(
    cached: TriangleCacheValue | null,
    loc: Point,
    paintOpt: TrianglePaintOption
  ): TriangleCacheValue | null {
    const {size} = paintOpt;
    const altitude = (size * Math.sqrt(3)) / 2;
    const vertices = new Float32Array([
      loc.x - size / 2,
      loc.y - altitude / 3,
      loc.x + size / 2,
      loc.y - altitude / 3,
      loc.x,
      loc.y + (altitude * 2) / 3,
    ]);

    if (!cached) {
      const geom = new THREE.BufferGeometry();
      updatePolylineGeometry(geom, vertices);
      const mesh = this.createMesh(geom, paintOpt);
      if (mesh === null) return null;
      this.scene.add(mesh);
      return {type: CacheType.TRIANGLE, data: loc, obj3d: mesh};
    }

    const geomUpdated = updateObject(
      this.backgroundColor,
      cached.obj3d,
      (geom) => {
        // Updating a geometry with three vertices is cheap enough. Update always.
        updatePolylineGeometry(geom, vertices);
        return geom;
      },
      paintOpt
    );
    return geomUpdated
      ? {type: CacheType.TRIANGLE, data: loc, obj3d: cached.obj3d}
      : cached;
  }

  createOrUpdateCircleObject(
    cached: CircleCacheValue | null,
    loc: Point,
    paintOpt: CirclePaintOption
  ): CircleCacheValue | null {
    const {radius} = paintOpt;
    const geom = new THREE.CircleBufferGeometry(paintOpt.radius);

    if (!cached) {
      const mesh = this.createMesh(geom, paintOpt);
      if (mesh === null) return null;
      mesh.position.set(loc.x, loc.y, 0);
      this.scene.add(mesh);
      return {type: CacheType.CIRCLE, data: {loc, radius}, obj3d: mesh};
    }

    // geometry/vertices are created by CircleBufferGeometry and it is quite complex.
    // Since it has N vertices (N < 20), update always.
    const geomUpdated = updateObject(
      this.backgroundColor,
      cached.obj3d,
      () => geom,
      paintOpt
    );
    if (!geomUpdated) return cached;
    cached.obj3d.position.set(loc.x, loc.y, 0);
    return {type: CacheType.CIRCLE, data: {loc, radius}, obj3d: cached.obj3d};
  }

  createOrUpdateTrapezoidObject(
    cached: TrapezoidCacheValue | null,
    start: Point,
    end: Point,
    paintOpt: TrapezoidPaintOption
  ): TrapezoidCacheValue | null {
    if (start.y !== end.y) {
      throw new RangeError('Input error: start.y != end.y.');
    }

    const {altitude} = paintOpt;
    const width = (2 / Math.sqrt(3)) * altitude;
    const shape = new THREE.Shape([
      new THREE.Vector2(start.x - width / 2, start.y - altitude / 2),
      new THREE.Vector2(start.x, start.y + altitude / 2),
      new THREE.Vector2(end.x, end.y + altitude / 2),
      new THREE.Vector2(end.x + width / 2, end.y - altitude / 2),
    ]);
    shape.autoClose = true;
    const geom = new THREE.ShapeBufferGeometry(shape);

    if (!cached) {
      const mesh = this.createMesh(geom, paintOpt);
      if (mesh === null) return null;
      this.scene.add(mesh);
      return {type: CacheType.TRAPEZOID, data: [start, end], obj3d: mesh};
    }

    const geomUpdated = updateObject(
      this.backgroundColor,
      cached.obj3d,
      () => geom,
      paintOpt
    );
    return geomUpdated
      ? {type: CacheType.TRAPEZOID, data: [start, end], obj3d: cached.obj3d}
      : cached;
  }

  flush() {
    this.renderer.render(this.scene, this.coordinator.getCamera());
  }

  dispose() {
    // Objects normally leave the scene one frame at a time via
    // `destroyObject`. Anything still in it holds GPU buffers, so release
    // those before the renderer itself.
    for (const obj3d of [...this.scene.children]) {
      this.releaseObject(obj3d);
    }
    this.renderer.dispose();
    // `dispose` frees the GL objects but keeps the context, and a page may
    // only hold a small number of them (16 in Chrome) before the browser
    // starts abandoning the oldest. Charts are created and destroyed as
    // cards mount, and this canvas is never reused, so drop the context too.
    // The loss is deliberate, so stop reporting it as a renderer failure
    // first; otherwise the owner would try to recover a discarded chart.
    if (this.onContextLost) {
      this.canvas.removeEventListener('webglcontextlost', this.onContextLost);
    }
    this.renderer.forceContextLoss();
  }
}

export const TEST_ONLY = {
  ThreeWrapper,
};
