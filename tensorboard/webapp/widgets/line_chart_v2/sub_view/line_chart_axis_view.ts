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
import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  Output,
} from '@angular/core';
import {Dimension, Formatter, Scale} from '../lib/public_types';
import {LinearScale, TemporalScale} from '../lib/scale';
import {
  getDomSizeInformedTickCount,
  getScaleRangeFromDomDim,
} from './chart_view_utils';
import {AxisUtils, MajorTick, MinorTick} from './line_chart_axis_utils';

const AXIS_FONT = '11px Roboto, sans-serif';

/**
 * Space right of a y label: the 1px axis line, its 5px margin, and 2px so
 * subpixel rounding cannot clip the widest label.
 */
const Y_AXIS_LABEL_GUTTER = 8;

/**
 * Narrowest y axis. The extent edit button is 24px wide and sits 5px from the
 * right edge, so anything narrower would let it cover the labels entirely.
 */
const MIN_Y_AXIS_WIDTH = 30;

/** Widest y axis, which is also the width when major labels need room. */
const MAX_Y_AXIS_WIDTH = 50;

@Component({
  standalone: false,
  selector: 'line-chart-axis',
  templateUrl: 'line_chart_axis_view.ng.html',
  styleUrls: ['line_chart_axis_view.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LineChartAxisComponent {
  @Input()
  axisExtent!: [number, number];

  @Input()
  axis!: 'x' | 'y';

  @Input()
  scale!: Scale;

  @Input()
  gridCount!: number;

  @Input()
  domDim!: Dimension;

  @Input()
  customFormatter?: Formatter;

  @Output()
  onViewExtentChange = new EventEmitter<[number, number]>();

  /**
   * Width the y axis needs for its widest label. The chart reserves this much
   * instead of a gutter wide enough for labels the chart does not have. The x
   * axis spans its grid column and never emits.
   */
  @Output()
  yAxisWidthChanged = new EventEmitter<number>();

  editMenuOpened = false;

  majorTicks: MajorTick[] = [];
  minorTicks: MinorTick[] = [];

  private yAxisWidth?: number;

  ngOnChanges() {
    let ticks: {minor: MinorTick[]; major: MajorTick[]} | null = null;
    const domSize = this.axis === 'x' ? this.domDim.width : this.domDim.height;
    const maxTickSize = getDomSizeInformedTickCount(domSize, this.gridCount);

    if (this.scale instanceof LinearScale) {
      ticks = AxisUtils.getTicksForLinearScale(
        this.scale,
        this.getFormatter(),
        maxTickSize,
        this.axisExtent
      );
    } else if (this.scale instanceof TemporalScale) {
      ticks = AxisUtils.getTicksForTemporalScale(
        this.scale,
        this.getFormatter(),
        maxTickSize,
        this.axisExtent
      );
    } else {
      ticks = AxisUtils.getStandardTicks(
        this.scale,
        this.getFormatter(),
        maxTickSize,
        this.axisExtent
      );
    }

    this.majorTicks = ticks.major;
    this.minorTicks = AxisUtils.filterTicksByVisibility(
      ticks.minor,
      (tick) => this.getDomPos(tick.value),
      this.axis,
      AXIS_FONT
    );

    if (this.axis === 'y') {
      // Major labels share the column with the minor ones, so keep the full
      // width whenever they are present.
      const width = this.majorTicks.length
        ? MAX_Y_AXIS_WIDTH
        : Math.min(
            MAX_Y_AXIS_WIDTH,
            Math.max(
              MIN_Y_AXIS_WIDTH,
              Math.ceil(
                AxisUtils.getMaxTickLabelWidth(this.minorTicks, AXIS_FONT)
              ) + Y_AXIS_LABEL_GUTTER
            )
          );
      if (width !== this.yAxisWidth) {
        this.yAxisWidth = width;
        this.yAxisWidthChanged.emit(width);
      }
    }
  }

  getFormatter(): Formatter {
    return this.customFormatter ?? this.scale.defaultFormatter;
  }

  trackByMinorTick(index: number, tick: MinorTick): number {
    return tick.value;
  }

  trackByMajorTick(index: number, tick: MajorTick): number {
    return tick.start;
  }

  private getDomPos(data: number): number {
    return this.scale.forward(
      this.axisExtent,
      getScaleRangeFromDomDim(this.domDim, this.axis),
      data
    );
  }

  textXPosition(tick: number): string {
    return this.axis === 'x' ? String(this.getDomPos(tick)) : '100%';
  }

  textYPosition(tick: number): string {
    return this.axis === 'x' ? '' : String(this.getDomPos(tick));
  }

  getMajorXPosition(tick: MajorTick): number {
    if (this.axis === 'y') return 0;

    return Math.min(this.domDim.width, Math.max(0, this.getDomPos(tick.start)));
  }

  getMajorWidthString(
    tick: MajorTick,
    isLast: boolean,
    nextTick?: MajorTick
  ): string {
    if (this.axis === 'y') return '';

    return (
      (isLast || !nextTick
        ? this.domDim.width
        : this.getMajorXPosition(nextTick)) -
      this.getMajorXPosition(tick) +
      'px'
    );
  }

  getMajorYPosition(tick: MajorTick): number {
    if (this.axis === 'x') return 0;

    return (
      this.domDim.height -
      Math.min(this.domDim.height, Math.max(0, this.getDomPos(tick.start)))
    );
  }

  getMajorHeightString(
    tick: MajorTick,
    isLast: boolean,
    nextTick?: MajorTick
  ): string {
    if (this.axis === 'x') return '';

    return (
      (isLast || !nextTick
        ? this.domDim.height
        : this.getMajorYPosition(nextTick)) -
      this.getMajorYPosition(tick) +
      'px'
    );
  }

  keydownPreventClose(event: KeyboardEvent) {
    // Any keydown or interaction inside mat-menu automatically closes the menu
    // which is not what we want. Stop the propoagation and do not let mat-menu
    // know about any keydowns except for `Escape`.
    if (event.key !== 'Escape') {
      event.stopPropagation();
    }
  }

  extentChanged(minValue: string, maxValue: string) {
    let min = Number(minValue);
    let max = Number(maxValue);

    if (max < min) {
      const temp = min;
      min = max;
      max = temp;
    }

    if (!Number.isFinite(min) || !Number.isFinite(max)) return;
    this.onViewExtentChange.emit([min, max]);
  }

  onAxisUpdateMenuOpen(
    minInput: HTMLInputElement,
    maxInput: HTMLInputElement,
    axisExtent: [number, number]
  ): void {
    minInput.value = String(axisExtent[0]);
    maxInput.value = String(axisExtent[1]);
    minInput.focus();
  }

  setEditMenuOpened(opened: boolean): void {
    this.editMenuOpened = opened;
  }
}
