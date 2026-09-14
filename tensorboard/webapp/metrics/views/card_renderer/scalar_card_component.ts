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
import {ComponentType} from '@angular/cdk/overlay';
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  Output,
  ViewChild,
} from '@angular/core';
import {MatDialog} from '@angular/material/dialog';
import {DataLoadState} from '../../../types/data';
import {
  TimeSelection,
  TimeSelectionAffordance,
  TimeSelectionToggleAffordance,
} from '../../../widgets/card_fob/card_fob_types';
import {
  Formatter,
  intlNumberFormatter,
  numberFormatter,
  relativeTimeFormatter,
  siNumberFormatter,
} from '../../../widgets/line_chart_v2/lib/formatter';
import {Extent} from '../../../widgets/line_chart_v2/lib/public_types';
import {LineChartComponent} from '../../../widgets/line_chart_v2/line_chart_component';
import {
  RendererType,
  ScaleType,
  TooltipDatum,
} from '../../../widgets/line_chart_v2/types';
import {CardState} from '../../store';
import {
  HeaderEditInfo,
  HeaderToggleInfo,
  TooltipSort,
  XAxisType,
} from '../../types';
import {
  MinMaxStep,
  ScalarCardDataSeries,
  ScalarCardPoint,
  ScalarCardSeriesMetadata,
  ScalarCardSeriesMetadataMap,
} from './scalar_card_types';
import {
  ColumnHeader,
  DataTableMode,
  SortingInfo,
  SortingOrder,
  DiscreteFilter,
  IntervalFilter,
  FilterAddedEvent,
  AddColumnEvent,
} from '../../../widgets/data_table/types';
import {isDatumVisible, TimeSelectionView} from './utils';
import {RunToHparamMap} from '../../../runs/types';

type ScalarTooltipDatum = TooltipDatum<
  ScalarCardSeriesMetadata & {
    closest: boolean;
  },
  ScalarCardPoint
>;

interface TooltipDataCache {
  tooltipData: TooltipDatum<ScalarCardSeriesMetadata, ScalarCardPoint>[];
  cursorLocationInDataCoord: {x: number; y: number};
  cursorLocation: {x: number; y: number};
  tooltipSort: TooltipSort;
  isTooltipRowsLimitEnabled: boolean;
  tooltipRowsLimit: number;
  additionalItemsCount: number;
  rows: ScalarTooltipDatum[];
}

@Component({
  standalone: false,
  selector: 'scalar-card-component',
  templateUrl: 'scalar_card_component.ng.html',
  styleUrls: ['scalar_card_component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ScalarCardComponent<Downloader> {
  readonly DataLoadState = DataLoadState;
  readonly RendererType = RendererType;
  readonly ScaleType = ScaleType;

  @Input() cardId!: string;
  @Input() chartMetadataMap!: ScalarCardSeriesMetadataMap;
  @Input() cardState?: Partial<CardState>;
  @Input() DataDownloadComponent!: ComponentType<Downloader>;
  @Input() dataSeries!: ScalarCardDataSeries[];
  @Input() ignoreOutliers!: boolean;
  @Input() isTooltipRowsLimitEnabled!: boolean;
  @Input() tooltipRowsLimit!: number;
  @Input() isPinned!: boolean;
  @Input() loadState!: DataLoadState;
  @Input() showFullWidth!: boolean;
  @Input() smoothingEnabled!: boolean;
  @Input() tag!: string;
  @Input() title!: string;
  @Input() tooltipSort!: TooltipSort;
  @Input() xAxisType!: XAxisType;
  @Input() xScaleType!: ScaleType;
  @Input() useDarkMode!: boolean;
  @Input() forceSvg!: boolean;
  @Input() columnCustomizationEnabled!: boolean;
  @Input() columnContextMenusEnabled!: boolean;
  @Input() linkedTimeSelection: TimeSelectionView | undefined;
  @Input() stepOrLinkedTimeSelection: TimeSelection | undefined;
  @Input() minMaxStep!: MinMaxStep;
  @Input() userViewBox!: Extent | null;
  @Input() columnHeaders!: ColumnHeader[];
  @Input() rangeEnabled!: boolean;
  @Input() columnFilters!: Map<string, DiscreteFilter | IntervalFilter>;
  @Input() selectableColumns!: ColumnHeader[];
  @Input() numColumnsLoaded!: number;
  @Input() numColumnsToLoad!: number;
  @Input() runToHparamMap!: RunToHparamMap;

  @Output() onFullSizeToggle = new EventEmitter<void>();
  @Output() onPinClicked = new EventEmitter<boolean>();
  @Output() onTimeSelectionChanged = new EventEmitter<{
    timeSelection: TimeSelection;
    affordance?: TimeSelectionAffordance;
  }>();
  @Output() onStepSelectorToggled =
    new EventEmitter<TimeSelectionToggleAffordance>();
  @Output() onDataTableSorting = new EventEmitter<SortingInfo>();
  @Output() editColumnHeaders = new EventEmitter<HeaderEditInfo>();
  @Output() openTableEditMenuToMode = new EventEmitter<DataTableMode>();
  @Output() addColumn = new EventEmitter<AddColumnEvent>();
  @Output() removeColumn = new EventEmitter<HeaderToggleInfo>();
  @Output() addFilter = new EventEmitter<FilterAddedEvent>();
  @Output() loadAllColumns = new EventEmitter<null>();

  @Output() onLineChartZoom = new EventEmitter<Extent | null>();
  @Output() onCardStateChanged = new EventEmitter<Partial<CardState>>();

  // Line chart may not exist when was never visible (*ngIf).
  @ViewChild(LineChartComponent)
  lineChart?: LineChartComponent;
  sortingInfo: SortingInfo = {
    name: 'run',
    order: SortingOrder.ASCENDING,
  };

  @ViewChild('chartContainer')
  chartContainer?: ElementRef;

  @ViewChild('dataTableContainer')
  dataTableContainer?: ElementRef;

  constructor(private readonly ref: ElementRef, private dialog: MatDialog) {}

  ngOnChanges() {
    if (this.cardState?.tableSorting) {
      this.sortingInfo = this.cardState.tableSorting;
    }
    if (this.cardState?.logScale !== undefined) {
      this.yScaleType = this.cardState.logScale
        ? ScaleType.LOG10
        : ScaleType.LINEAR;
    }
  }

  yScaleType = ScaleType.LINEAR;
  additionalItemsCount = 0;

  toggleYScaleType() {
    this.yScaleType =
      this.yScaleType === ScaleType.LINEAR ? ScaleType.LOG10 : ScaleType.LINEAR;
    this.onCardStateChanged.emit({
      logScale: this.yScaleType === ScaleType.LOG10,
    });
  }

  sortDataBy(sortingInfo: SortingInfo) {
    this.sortingInfo = sortingInfo;
    this.onDataTableSorting.emit(sortingInfo);
    this.onCardStateChanged.emit({tableSorting: sortingInfo});
  }

  resetDomain() {
    if (this.lineChart) {
      this.lineChart.viewBoxReset();
    }
  }

  trackByTooltipDatum(index: number, datum: ScalarTooltipDatum) {
    return datum.id;
  }

  readonly relativeXFormatter = relativeTimeFormatter;
  readonly valueFormatter = numberFormatter;
  readonly stepFormatter = intlNumberFormatter;

  getCustomXFormatter(): Formatter | undefined {
    switch (this.xAxisType) {
      case XAxisType.RELATIVE:
        return relativeTimeFormatter;
      case XAxisType.STEP:
        return siNumberFormatter;
      case XAxisType.WALL_TIME:
      default:
        return undefined;
    }
  }

  // `getCursorAwareTooltipData` is invoked from a template expression, so it
  // runs on every change detection pass while its inputs only change when the
  // cursor moves or the data reloads.
  private tooltipDataCache: TooltipDataCache | null = null;

  getCursorAwareTooltipData(
    tooltipData: TooltipDatum<ScalarCardSeriesMetadata, ScalarCardPoint>[],
    cursorLocationInDataCoord: {x: number; y: number},
    cursorLocation: {x: number; y: number}
  ): ScalarTooltipDatum[] {
    const cache = this.tooltipDataCache;
    if (
      cache !== null &&
      cache.tooltipData === tooltipData &&
      cache.cursorLocationInDataCoord === cursorLocationInDataCoord &&
      cache.cursorLocation === cursorLocation &&
      cache.tooltipSort === this.tooltipSort &&
      cache.isTooltipRowsLimitEnabled === this.isTooltipRowsLimitEnabled &&
      cache.tooltipRowsLimit === this.tooltipRowsLimit
    ) {
      this.additionalItemsCount = cache.additionalItemsCount;
      return cache.rows;
    }

    const seriesCount = tooltipData.length;
    // Every series needs its distance to the cursor: it marks the closest row
    // and, for `TooltipSort.NEAREST`, orders the rows.
    const distToCursorPixels = new Float64Array(seriesCount);
    const order: number[] = [];
    let minDist = Infinity;
    let closestIndex = 0;
    for (let index = 0; index < seriesCount; index++) {
      const domPoint = tooltipData[index].domPoint;
      const dist = Math.hypot(
        domPoint.x - cursorLocation.x,
        domPoint.y - cursorLocation.y
      );
      distToCursorPixels[index] = dist;
      if (minDist > dist) {
        minDist = dist;
        closestIndex = index;
      }
      order.push(index);
    }

    // Sorting indices rather than rows keeps the sort allocation free and lets
    // the row limit apply before any row is built.
    const cursorY = cursorLocationInDataCoord.y;
    switch (this.tooltipSort) {
      case TooltipSort.ASCENDING:
        order.sort(
          (a, b) => tooltipData[a].dataPoint.y - tooltipData[b].dataPoint.y
        );
        break;
      case TooltipSort.DESCENDING:
        order.sort(
          (a, b) => tooltipData[b].dataPoint.y - tooltipData[a].dataPoint.y
        );
        break;
      case TooltipSort.NEAREST:
        order.sort((a, b) => distToCursorPixels[a] - distToCursorPixels[b]);
        break;
      case TooltipSort.NEAREST_Y:
        order.sort((a, b) => {
          const distToCursorYA = tooltipData[a].dataPoint.y - cursorY;
          const distToCursorYB = tooltipData[b].dataPoint.y - cursorY;
          return distToCursorYA - distToCursorYB;
        });
        break;
      case TooltipSort.DEFAULT:
      case TooltipSort.ALPHABETICAL:
        order.sort((a, b) => {
          const nameA = tooltipData[a].metadata.displayName;
          const nameB = tooltipData[b].metadata.displayName;
          if (nameA < nameB) {
            return -1;
          }
          if (nameA > nameB) {
            return 1;
          }
          return 0;
        });
        break;
    }

    const rowCount = this.isTooltipRowsLimitEnabled
      ? Math.max(0, Math.min(this.tooltipRowsLimit, seriesCount))
      : seriesCount;
    // The closest series is not marked when the limit hides it.
    const rows = order.slice(0, rowCount).map((index) => {
      const datum = tooltipData[index];
      return {
        ...datum,
        metadata: {...datum.metadata, closest: index === closestIndex},
      };
    });

    this.additionalItemsCount = seriesCount - rowCount;
    this.tooltipDataCache = {
      tooltipData,
      cursorLocationInDataCoord,
      cursorLocation,
      tooltipSort: this.tooltipSort,
      isTooltipRowsLimitEnabled: this.isTooltipRowsLimitEnabled,
      tooltipRowsLimit: this.tooltipRowsLimit,
      additionalItemsCount: this.additionalItemsCount,
      rows,
    };
    return rows;
  }

  openDataDownloadDialog(): void {
    this.dialog.open(this.DataDownloadComponent, {
      data: {cardId: this.cardId},
    });
  }

  onFobRemoved() {
    this.onStepSelectorToggled.emit(TimeSelectionToggleAffordance.FOB_DESELECT);
  }

  showDataTable() {
    return this.xAxisType === XAxisType.STEP && this.stepOrLinkedTimeSelection;
  }

  showFobController() {
    return this.xAxisType === XAxisType.STEP && this.minMaxStep;
  }

  canExpandTable() {
    const visbleRuns = this.dataSeries.filter((datum) => {
      return isDatumVisible(datum, this.chartMetadataMap);
    });

    // 3 is the maximum number of runs that can be shown before
    // the height of the table exceeds $_data_table_initial_height.
    return visbleRuns.length > 3;
  }

  shouldExpandTable() {
    // If the user has resized the data table a height style will be set.
    // If the data table has been resized we always want to expand the table.
    // Otherwise the table should be toggled.
    return Boolean(
      this.dataTableContainer?.nativeElement.style.height ||
        !this.cardState?.tableExpanded
    );
  }

  toggleTableExpanded() {
    this.onCardStateChanged.emit({
      ...this.cardState,
      tableExpanded: this.shouldExpandTable(),
    });
    // Manually resizing an element sets a style value on the element which takes
    // precedence over any classes the element may have. This value must be removed
    // for the table to expand or collapse correctly.
    if (this.dataTableContainer) {
      this.dataTableContainer.nativeElement.style.height = '';
    }
  }

  /**
   * Restores the persisted heights. `resize: vertical` makes the browser write
   * an inline height when the user drags a container, so the stored heights go
   * in the same place rather than through a binding, and only while that place
   * is empty: overwriting it would revert a drag that change detection has not
   * yet reported.
   */
  ngAfterViewChecked() {
    this.restoreHeight(this.chartContainer, this.cardState?.chartHeight);
    // An expanded table is sized by its content; `.expanded` sets
    // `height: auto` and an inline height would override it.
    if (!this.cardState?.tableExpanded) {
      this.restoreHeight(this.dataTableContainer, this.cardState?.tableHeight);
    }
  }

  private restoreHeight(
    container: ElementRef<HTMLElement> | undefined,
    height: number | undefined
  ) {
    const element = container?.nativeElement;
    if (!element || !height || element.style.height) {
      return;
    }
    element.style.height = `${height}px`;
  }

  /**
   * Height of a container the user drag-resized, or null when this resize was
   * not a drag. Only a drag leaves an inline height, so relayouts from window
   * resizes, expanding the card, or expanding the table are not persisted, and
   * re-observing a height this card just applied does not report it again.
   */
  private draggedHeight(
    container: ElementRef<HTMLElement> | undefined,
    appliedHeight: number | undefined
  ): number | null {
    const element = container?.nativeElement;
    if (!element?.style.height) {
      return null;
    }
    // `offsetHeight` is 0 while the container is detached or hidden; that is
    // not a resize the user performed.
    const height = Math.round(element.offsetHeight);
    if (!height || height === appliedHeight) {
      return null;
    }
    return height;
  }

  onChartResized() {
    const height = this.draggedHeight(
      this.chartContainer,
      this.cardState?.chartHeight
    );
    if (height !== null) {
      this.onCardStateChanged.emit({chartHeight: height});
    }
  }

  onDataTableResized() {
    if (this.cardState?.tableExpanded) {
      return;
    }
    const height = this.draggedHeight(
      this.dataTableContainer,
      this.cardState?.tableHeight
    );
    if (height !== null) {
      this.onCardStateChanged.emit({tableHeight: height});
    }
  }

  openTableEditMenu() {
    const currentTableMode = this.rangeEnabled
      ? DataTableMode.RANGE
      : DataTableMode.SINGLE;
    this.openTableEditMenuToMode.emit(currentTableMode);
  }
}
