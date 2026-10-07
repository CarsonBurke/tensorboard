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
import {CdkScrollable} from '@angular/cdk/scrolling';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  NgZone,
  Optional,
  Output,
  SimpleChanges,
} from '@angular/core';
import {MetricsCatalogViewport} from '../../data_source';
import {CardObserver} from '../card_renderer/card_lazy_loader';
import {CardIdWithMetadata} from '../metrics_view_types';
import {
  CatalogScrollWindow,
  CatalogScrollGeometry,
  CATALOG_ROW_HEIGHT,
  catalogGridColumns,
} from './card_groups_component';

// The "Tags matching filter" toolbar above the first row of cards.
const TOOLBAR_HEIGHT = 42;

interface FilteredCatalogView {
  totalCards: number;
  filteredOffset: number;
  viewport: MetricsCatalogViewport;
  cardMinWidth: number | null;
  scope: string;
}

@Component({
  standalone: false,
  selector: 'metrics-filtered-view-component',
  template: `
    <div class="group-toolbar">
      <span class="group-text">
        <span class="group-title" aria-role="heading" aria-level="3"
          >Tags matching filter</span
        >
        <span
          *ngIf="
            (catalog ? catalog.totalCards : cardIdsWithMetadata.length) > 1
          "
          class="group-card-count"
          >{{
            (catalog ? catalog.totalCards : cardIdsWithMetadata.length) | number
          }}
          cards</span
        >
      </span>
    </div>
    <metrics-empty-tag-match
      *ngIf="catalog ? catalog.totalCards === 0 : isEmptyMatch"
      class="warn"
    ></metrics-empty-tag-match>
    <div
      class="catalog-spacer"
      aria-hidden="true"
      [style.height.px]="beforeHeight"
    ></div>
    <div class="card-window">
      <metrics-card-grid
        [cardIdsWithMetadata]="cardIdsWithMetadata"
        [virtualWindow]="catalog !== null"
        [cardObserver]="cardObserver"
      ></metrics-card-grid>
    </div>
    <div
      class="catalog-spacer"
      aria-hidden="true"
      [style.height.px]="afterHeight"
    ></div>
  `,
  styleUrls: ['filtered_view_component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class FilteredViewComponent {
  @Input() isEmptyMatch!: boolean;
  @Input() cardObserver!: CardObserver;
  @Input() cardIdsWithMetadata: CardIdWithMetadata[] = [];
  @Input() catalog: FilteredCatalogView | null = null;
  @Output() viewportChanged = new EventEmitter<MetricsCatalogViewport>();

  beforeHeight = 0;
  afterHeight = 0;
  private windowHeight: number | null = null;
  private columns = 1;
  private lastRequest = '';
  private scrollWindow?: CatalogScrollWindow;
  private readonly geometry = new CatalogScrollGeometry();
  private cardBounds: Array<{top: number; bottom: number}> = [];
  private renderedWindowHeight = 0;

  constructor(
    private readonly element: ElementRef<HTMLElement>,
    private readonly changeDetector: ChangeDetectorRef,
    private readonly zone: NgZone,
    @Optional() private readonly scrollable: CdkScrollable | null
  ) {}

  ngAfterViewInit() {
    if (!this.scrollable) return;
    this.scrollWindow = new CatalogScrollWindow(
      this.scrollable.getElementRef().nativeElement,
      this.element.nativeElement,
      this.zone,
      (measure) => this.updateViewport(measure),
      () => this.catalog !== null
    );
  }

  ngOnChanges(changes: SimpleChanges) {
    const previous = changes['catalog']
      ?.previousValue as FilteredCatalogView | null;
    const scopeChanged =
      changes['catalog'] && previous?.scope !== this.catalog?.scope;
    if (scopeChanged) {
      this.lastRequest = '';
      if (this.scrollable)
        this.scrollable.getElementRef().nativeElement.scrollTop = 0;
    }
    this.updateGeometry();
    this.scrollWindow?.invalidate();
  }

  private updateGeometry() {
    if (!this.catalog) {
      this.beforeHeight = this.afterHeight = 0;
      this.windowHeight = null;
      return;
    }
    const {filteredOffset, totalCards} = this.catalog;
    const count = this.cardIdsWithMetadata.length;
    const start =
      Math.floor(filteredOffset / this.columns) * CATALOG_ROW_HEIGHT;
    this.geometry.update(
      Math.ceil(totalCards / this.columns) * CATALOG_ROW_HEIGHT,
      start,
      start + Math.ceil(count / this.columns) * CATALOG_ROW_HEIGHT
    );
    this.beforeHeight = this.geometry.before;
    this.afterHeight = this.geometry.after;
    this.windowHeight =
      Math.ceil(count / this.columns) * CATALOG_ROW_HEIGHT + 16;
  }

  private updateViewport(measure: boolean) {
    if (!this.catalog || !this.scrollWindow) return;
    const catalog = this.catalog;
    const bounds = this.scrollWindow.bounds();
    const columns = catalogGridColumns(bounds.width, catalog.cardMinWidth);
    const geometryChanged = columns !== this.columns;
    if (geometryChanged) {
      this.columns = columns;
      this.updateGeometry();
      this.scrollWindow.invalidate();
    }
    if (measure || geometryChanged) {
      const cards =
        this.element.nativeElement.querySelectorAll<HTMLElement>('.card-space');
      this.scrollWindow.observeContent(cards);
      this.cardBounds = Array.from(cards, (card) => {
        const rect = card.getBoundingClientRect();
        return {
          top: rect.top - bounds.hostTop,
          bottom: rect.bottom - bounds.hostTop,
        };
      });
      this.renderedWindowHeight =
        this.element.nativeElement
          .querySelector<HTMLElement>('.card-window')
          ?.getBoundingClientRect().height ?? 0;
    }
    const bufferTop = Math.max(0, bounds.top - bounds.height);
    const top = Math.max(0, bounds.top - TOOLBAR_HEIGHT);
    const firstVisible = this.cardBounds.findIndex(
      (card) => card.bottom > bufferTop
    );
    const block = this.columns * 4;
    const filteredLimit = Math.max(
      40,
      (Math.ceil((bounds.height * 3) / CATALOG_ROW_HEIGHT) + 8) * this.columns
    );
    let first: number;
    if (this.scrollWindow.isAtBottom()) {
      // Map End to a complete final window, rather than the first logical row
      // covered by a compressed spacer's last physical viewport.
      first =
        Math.ceil(Math.max(0, catalog.totalCards - filteredLimit) / block) *
        block;
    } else if (
      bufferTop >= this.beforeHeight + TOOLBAR_HEIGHT &&
      firstVisible >= 0
    ) {
      // Use actual card geometry for full-width cards and expanded run tables.
      first = catalog.filteredOffset + firstVisible;
    } else {
      const extraHeight =
        top >= this.beforeHeight
          ? Math.max(
              0,
              this.renderedWindowHeight - (this.windowHeight ?? 16) + 16
            )
          : 0;
      first = Math.max(
        0,
        Math.floor(
          this.geometry.toLogical(Math.max(0, top - extraHeight)) /
            CATALOG_ROW_HEIGHT
        ) *
          this.columns -
          Math.ceil(bounds.height / CATALOG_ROW_HEIGHT) * this.columns
      );
    }
    const filteredOffset =
      Math.floor(Math.min(first, Math.max(0, catalog.totalCards - 1)) / block) *
      block;
    const viewport = {
      ...catalog.viewport,
      visibleGroups: [],
      filteredOffset,
      filteredLimit,
    };
    const request = JSON.stringify(viewport);
    if (request !== this.lastRequest || geometryChanged) {
      this.zone.run(() => {
        if (request !== this.lastRequest) {
          this.lastRequest = request;
          this.viewportChanged.emit(viewport);
        }
        if (geometryChanged) this.changeDetector.markForCheck();
      });
    }
  }

  ngOnDestroy() {
    this.scrollWindow?.destroy();
  }
}
