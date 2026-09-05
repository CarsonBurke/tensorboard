/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
  OnChanges,
  Output,
  SimpleChanges,
} from '@angular/core';
import {
  ColumnHeader,
  TableData,
  SortingInfo,
  SortingOrder,
  ColumnHeaderType,
  FilterAddedEvent,
  DiscreteFilter,
  IntervalFilter,
  ReorderColumnEvent,
  AddColumnEvent,
} from '../../../widgets/data_table/types';
import {memoize} from '../../../util/memoize';
import {RUN_START_TIME_SORT_KEY} from './sorting_utils';

const ROW_HEIGHT_IN_PX = 48;
const OVERSCAN_ROWS = 10;
const INITIAL_RENDERED_ROWS = 50;

@Component({
  standalone: false,
  selector: 'runs-data-table',
  templateUrl: 'runs_data_table.ng.html',
  styleUrls: ['runs_data_table.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RunsDataTable implements OnChanges {
  @Input() headers!: ColumnHeader[];
  @Input() data!: TableData[];
  @Input() sortingInfo!: SortingInfo;
  @Input() experimentIds!: string[];
  @Input() regexFilter!: string;
  @Input() selectableColumns!: ColumnHeader[];
  @Input() numColumnsLoaded!: number;
  @Input() numColumnsToLoad!: number;
  @Input() loading!: boolean;
  @Input() columnFilters!: Map<string, DiscreteFilter | IntervalFilter>;
  @Input() scrollTop = 0;
  @Input() viewportHeight = 0;

  ColumnHeaderType = ColumnHeaderType;
  SortingOrder = SortingOrder;
  runStartTimeSortKey = RUN_START_TIME_SORT_KEY;

  visibleData: TableData[] = [];
  topSpacerHeightInPx = 0;
  bottomSpacerHeightInPx = 0;

  private renderedData: TableData[] | null = null;
  private renderedStart = -1;
  private renderedEnd = -1;

  @Output() sortDataBy = new EventEmitter<SortingInfo>();
  @Output() orderColumns = new EventEmitter<ReorderColumnEvent>();
  @Output() onSelectionToggle = new EventEmitter<string>();
  @Output() onAllSelectionToggle = new EventEmitter<string[]>();
  @Output() onRegexFilterChange = new EventEmitter<string>();
  @Output() onRunColorChange = new EventEmitter<{
    runId: string;
    newColor: string;
  }>();
  @Output() addColumn = new EventEmitter<AddColumnEvent>();
  @Output() removeColumn = new EventEmitter<ColumnHeader>();
  @Output() onSelectionDblClick = new EventEmitter<string>();
  @Output() addFilter = new EventEmitter<FilterAddedEvent>();
  @Output() loadAllColumns = new EventEmitter<null>();

  // Columns must be memoized to stop needless re-rendering of the content and headers in these
  // columns. This has been known to cause problems with the controls in these columns,
  // specifically the add button.
  extendHeaders = memoize(this.internalExtendHeaders);

  private internalExtendHeaders(headers: ColumnHeader[]) {
    return ([] as Array<ColumnHeader>).concat(
      [
        {
          name: 'selected',
          displayName: '',
          type: ColumnHeaderType.CUSTOM,
          enabled: true,
        },
      ],
      headers,
      [
        {
          name: 'color',
          displayName: '',
          type: ColumnHeaderType.COLOR,
          enabled: true,
        },
      ]
    );
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['data'] || changes['scrollTop'] || changes['viewportHeight']) {
      this.updateVisibleData();
    }
  }

  private updateVisibleData() {
    if (!this.data?.length) {
      this.visibleData = [];
      this.topSpacerHeightInPx = 0;
      this.bottomSpacerHeightInPx = 0;
      return;
    }

    const viewportRows = this.viewportHeight
      ? Math.ceil(this.viewportHeight / ROW_HEIGHT_IN_PX)
      : INITIAL_RENDERED_ROWS;
    const renderedRows = viewportRows + 2 * OVERSCAN_ROWS;
    const firstVisibleRow = Math.floor(this.scrollTop / ROW_HEIGHT_IN_PX);
    const start = Math.min(
      Math.max(0, this.data.length - renderedRows),
      Math.max(0, firstVisibleRow - OVERSCAN_ROWS)
    );
    const end = Math.min(this.data.length, start + renderedRows);

    if (
      this.renderedData === this.data &&
      this.renderedStart === start &&
      this.renderedEnd === end
    ) {
      return;
    }
    this.renderedData = this.data;
    this.renderedStart = start;
    this.renderedEnd = end;

    this.visibleData = this.data.slice(start, end);
    this.topSpacerHeightInPx = start * ROW_HEIGHT_IN_PX;
    this.bottomSpacerHeightInPx = (this.data.length - end) * ROW_HEIGHT_IN_PX;
  }

  selectionClick(event: MouseEvent, runId: string) {
    // Keyboard activation has detail 0; the first mouse click has detail 1.
    if (event.detail <= 1) {
      this.onSelectionToggle.emit(runId);
      return;
    }

    // Keep later clicks from toggling the controlled checkbox away from the
    // single-run selection produced by a double click.
    event.preventDefault();
    if (event.detail === 2) {
      this.onSelectionDblClick.emit(runId);
    }
  }

  allRowsSelected() {
    return this.data?.every((row) => row['selected']);
  }

  someRowsSelected() {
    return this.data?.some((row) => row['selected']);
  }

  handleSelectAll() {
    this.onAllSelectionToggle.emit(this.data?.map((row) => row.id));
  }

  onFilterKeyUp(event: KeyboardEvent) {
    const input = event.target! as HTMLInputElement;
    this.onRegexFilterChange.emit(input.value);
  }

  sortRunsByDefault() {
    this.sortDataBy.emit({name: 'run', order: SortingOrder.ASCENDING});
  }

  sortRunsByNewest() {
    this.sortDataBy.emit({
      name: RUN_START_TIME_SORT_KEY,
      order: SortingOrder.DESCENDING,
    });
  }

  sortRunsByOldest() {
    this.sortDataBy.emit({
      name: RUN_START_TIME_SORT_KEY,
      order: SortingOrder.ASCENDING,
    });
  }

  isSortingBy(name: string, order: SortingOrder) {
    return this.sortingInfo?.name === name && this.sortingInfo.order === order;
  }

  /**
   * Keep row components stable while their displayed values update. Reusing by
   * run ID avoids serializing every row during each change detection pass and
   * prevents large run tables from rebuilding unchanged controls.
   */
  trackByRuns(index: number, data: TableData) {
    return data.id;
  }
}
