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

import {DataDrawable} from './drawable';
import {Polyline} from './internal_types';

enum PartitionType {
  NUMBER,
  NAN,
}

export class SeriesLineView extends DataDrawable {
  private recordPartition(
    isNumberPartition: boolean,
    slice: Float32Array,
    nanSubstitute: {x: number; y: number}
  ) {
    return isNumberPartition
      ? {type: PartitionType.NUMBER, polyline: slice}
      : {
          type: PartitionType.NAN,
          polyline: slice.map((x, ind) => {
            if (!isNaN(x)) return x;
            return ind % 2 === 0 ? nanSubstitute.x : nanSubstitute.y;
          }),
        };
  }

  /**
   * @param polyline Expects the polyline to have even length and encode two-dimensional
   *   coordinates.
   * @param hasNaN Whether `polyline` holds a NaN, as recorded by the
   *   coordinate transform.
   */
  private partitionPolyline(
    polyline: Polyline,
    hasNaN: boolean
  ): Array<{polyline: Float32Array; type: PartitionType}> {
    // A series without NaNs is a single number partition covering the whole
    // polyline, which needs neither a copy of it nor the zero coordinate.
    if (!hasNaN) {
      return [{type: PartitionType.NUMBER, polyline}];
    }

    const partition = [];
    let partitionStartInd: number = 0;
    let isPrevValueNaN = false;
    const zeroCoord = this.coordinator.transformDataToUiCoord(
      this.getLayoutRect(),
      [0, 0]
    );
    const zeroPoint = {x: zeroCoord[0], y: zeroCoord[1]};

    let lastLegalNumber: {
      x: number;
      y: number;
    } | null = null;

    for (let index = 0; index < polyline.length; index += 2) {
      const x = polyline[index];
      const y = polyline[index + 1];
      const hasNaN = isNaN(x) || isNaN(y);
      if (hasNaN !== isPrevValueNaN && partitionStartInd !== index) {
        partition.push(
          this.recordPartition(
            !isPrevValueNaN,
            polyline.slice(partitionStartInd, index),
            lastLegalNumber === null ? {x, y} : lastLegalNumber
          )
        );
        partitionStartInd = index;
      }

      if (!hasNaN) {
        lastLegalNumber = {x, y};
      }

      isPrevValueNaN = hasNaN;
    }

    if (partitionStartInd !== polyline.length - 1) {
      partition.push(
        this.recordPartition(
          !isPrevValueNaN,
          polyline.slice(partitionStartInd, polyline.length),
          lastLegalNumber ?? zeroPoint
        )
      );
    }

    return partition;
  }

  redraw() {
    const map = this.getMetadataMap();
    for (const series of this.series) {
      const metadata = map[series.id];
      if (!metadata) continue;
      if (series.polyline.length % 2 !== 0) {
        throw new Error(
          `Cannot have odd length-ed polyline: ${series.polyline.length}`
        );
      }

      const partitionedPolyline = this.partitionPolyline(
        series.polyline,
        series.hasNaN
      );

      for (const [
        partitionInd,
        {type, polyline},
      ] of partitionedPolyline.entries()) {
        if (type === PartitionType.NUMBER) {
          if (polyline.length === 2) {
            this.paintBrush.setCircle(
              // The leading number keeps ids with delimiters unambiguous;
              // serializing them on every frame is measurable.
              'circle:' + partitionInd + ':' + series.id,
              {x: polyline[0], y: polyline[1]},
              {
                color: metadata.color,
                visible: metadata.visible,
                opacity: metadata.opacity ?? 1,
                radius: 4,
              }
            );
          } else {
            this.paintBrush.setLine(
              'line:' + partitionInd + ':' + series.id,
              polyline,
              {
                color: metadata.color,
                visible: metadata.visible,
                opacity: metadata.opacity ?? 1,
                width: 2,
              }
            );
          }
          // Should not render triangles to mark NaNs for auxiliary lines.
        } else if (!metadata.aux) {
          for (let index = 0; index < polyline.length; index += 2) {
            this.paintBrush.setTriangle(
              // Keyed by position in the series, not by coordinate: two NaNs
              // can share a coordinate (identical wall times), and a
              // coordinate key would both collide (leaking the overwritten
              // renderer object) and miss the cache on every pan and zoom.
              'NaN:' + partitionInd + ':' + index + ':' + series.id,
              {x: polyline[index], y: polyline[index + 1]},
              {
                color: metadata.color,
                visible: metadata.visible,
                opacity: metadata.opacity ?? 1,
                size: 12,
              }
            );
          }
        }
      }
    }
  }
}
