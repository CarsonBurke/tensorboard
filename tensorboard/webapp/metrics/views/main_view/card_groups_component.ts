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
import {
  CardObserver,
  CARD_RETENTION_VIEWPORTS,
} from '../card_renderer/card_lazy_loader';
import {CardGroup} from '../metrics_view_types';

export interface CatalogGroupView {
  groupOffset: number;
  totalGroups: number;
  viewport: MetricsCatalogViewport;
  expanded: Map<string, boolean>;
  pages: Map<string, number>;
  pageSize: number;
  cardMinWidth: number | null;
  scope: string;
}

export interface CatalogCardGroup extends CardGroup {
  totalCards?: number;
}

// Keep in sync with the grid's 16px gap, 335px minimum width and 320px height.
export const CATALOG_ROW_HEIGHT = 336;
export const CATALOG_GROUP_HEIGHT = 42;
export function catalogGridColumns(width: number, cardMinWidth: number | null) {
  const minimum =
    cardMinWidth && cardMinWidth >= 335 && cardMinWidth <= 735
      ? cardMinWidth
      : 335;
  return Math.max(1, Math.floor((width - 16) / (minimum + 16)));
}

/** Compress only unloaded spacers to stay below browser scroll-height limits. */
export class CatalogScrollGeometry {
  private start = 0;
  private end = 0;
  private scale = 1;
  before = 0;
  after = 0;

  update(total: number, start: number, end: number) {
    this.start = start;
    this.end = end;
    this.scale = Math.min(1, 8_000_000 / Math.max(1, total));
    this.before = start * this.scale;
    this.after = Math.max(0, total - end) * this.scale;
  }

  toLogical(position: number) {
    if (position < this.before) return position / this.scale;
    if (position <= this.before + this.end - this.start) {
      return this.start + position - this.before;
    }
    return (
      this.end + (position - this.before - this.end + this.start) / this.scale
    );
  }

  toPhysical(position: number) {
    if (position < this.start) return position * this.scale;
    if (position <= this.end) return this.before + position - this.start;
    return (
      this.before + this.end - this.start + (position - this.end) * this.scale
    );
  }
}

/** One frame-coalesced listener per active virtual view, not per category. */
export class CatalogScrollWindow {
  private frame: number | null = null;
  private readonly resizeObserver: ResizeObserver;
  private measure = true;
  private readonly contentElements = new Set<HTMLElement>();
  readonly schedule = () => {
    if (this.frame !== null) return;
    this.zone.runOutsideAngular(() => {
      this.frame = requestAnimationFrame(() => {
        this.frame = null;
        const measure = this.measure;
        this.measure = false;
        this.update(measure);
      });
    });
  };

  invalidate() {
    this.measure = true;
    this.schedule();
  }

  observeContent(elements: ArrayLike<HTMLElement>) {
    const current = new Set<HTMLElement>();
    for (let index = 0; index < elements.length; index++) {
      current.add(elements[index]);
    }
    for (const element of this.contentElements) {
      if (!current.has(element)) {
        this.resizeObserver.unobserve(element);
        this.contentElements.delete(element);
      }
    }
    for (const element of current) {
      if (!this.contentElements.has(element)) {
        this.contentElements.add(element);
        this.resizeObserver.observe(element);
      }
    }
  }

  constructor(
    readonly root: HTMLElement,
    private readonly host: HTMLElement,
    private readonly zone: NgZone,
    private readonly update: (measure: boolean) => void
  ) {
    this.resizeObserver = this.zone.runOutsideAngular(
      () => new ResizeObserver(() => this.invalidate())
    );
    this.zone.runOutsideAngular(() => {
      root.addEventListener('scroll', this.schedule, {passive: true});
      this.resizeObserver.observe(root);
      this.resizeObserver.observe(host);
      // Pinned cards can resize above this view without resizing the scroll root.
      Array.from(root.children).forEach((child) =>
        this.resizeObserver.observe(child)
      );
    });
    this.schedule();
  }

  bounds() {
    const root = this.root.getBoundingClientRect();
    const host = this.host.getBoundingClientRect();
    return {
      top: Math.max(0, root.top - host.top),
      bottom: root.top + this.root.clientHeight - host.top,
      width: this.host.clientWidth,
      height: this.root.clientHeight,
      hostTop: host.top,
      rootTop: root.top,
    };
  }

  destroy() {
    this.root.removeEventListener('scroll', this.schedule);
    this.resizeObserver.disconnect();
    if (this.frame !== null) cancelAnimationFrame(this.frame);
  }
}

@Component({
  standalone: false,
  selector: 'metrics-card-groups-component',
  template: `
    <div aria-hidden="true" [style.height.px]="beforeHeight"></div>
    <div
      *ngFor="let group of cardGroups; trackBy: trackByGroup"
      class="card-group"
      [style.min-height.px]="reservedHeight(group)"
    >
      <metrics-card-group-toolbar
        [numberOfCards]="group.totalCards ?? group.items.length"
        [groupName]="group.groupName"
      ></metrics-card-group-toolbar>
      <metrics-card-grid
        [cardIdsWithMetadata]="group.items"
        [serverTotalCards]="catalog ? group.totalCards ?? null : null"
        [cardObserver]="cardObserver"
        [groupName]="group.groupName"
      ></metrics-card-grid>
    </div>
    <div aria-hidden="true" [style.height.px]="afterHeight"></div>
  `,
  styleUrls: ['card_groups_component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CardGroupsComponent {
  @Input() cardGroups: CatalogCardGroup[] = [];
  @Input() cardObserver!: CardObserver;
  @Input() catalog: CatalogGroupView | null = null;
  @Output() viewportChanged = new EventEmitter<MetricsCatalogViewport>();

  beforeHeight = 0;
  afterHeight = 0;
  private columns = 1;
  private scrollWindow?: CatalogScrollWindow;
  // Only expanded categories need geometry beyond the fixed summary height.
  private readonly expandedGroups = new Map<number, CatalogCardGroup>();
  private readonly measuredHeights = new Map<string, number>();
  private groupBounds: Array<{top: number; bottom: number}> = [];
  private lastRequest = '';
  private readonly geometry = new CatalogScrollGeometry();
  private logicalAnchor: number | null = null;
  private bottomAnchor = false;

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
      (measure) => this.updateViewport(measure)
    );
  }

  ngOnChanges(changes: SimpleChanges) {
    const previous = changes['catalog']
      ?.previousValue as CatalogGroupView | null;
    const scopeChanged =
      changes['catalog'] && previous?.scope !== this.catalog?.scope;
    if (scopeChanged) {
      this.logicalAnchor = null;
      this.bottomAnchor = false;
      this.expandedGroups.clear();
      this.measuredHeights.clear();
      this.lastRequest = '';
      if (this.scrollable)
        this.scrollable.getElementRef().nativeElement.scrollTop = 0;
    } else if (
      previous &&
      this.catalog &&
      this.scrollWindow &&
      previous.groupOffset !== this.catalog.groupOffset
    ) {
      this.logicalAnchor = this.geometry.toLogical(
        this.scrollWindow.bounds().top
      );
    }
    if (
      previous &&
      this.catalog &&
      (previous.pages !== this.catalog.pages ||
        previous.pageSize !== this.catalog.pageSize ||
        previous.expanded !== this.catalog.expanded)
    ) {
      for (const name of this.measuredHeights.keys()) {
        if (
          previous.pageSize !== this.catalog.pageSize ||
          previous.pages.get(name) !== this.catalog.pages.get(name) ||
          previous.expanded.get(name) !== this.catalog.expanded.get(name)
        ) {
          this.measuredHeights.delete(name);
        }
      }
    }
    if (!scopeChanged && this.isAtBottom()) {
      this.bottomAnchor = true;
      this.logicalAnchor = null;
    }
    this.updateGeometry();
    this.scrollWindow?.invalidate();
  }

  private updateGeometry() {
    const catalog = this.catalog;
    if (!catalog) {
      this.beforeHeight = this.afterHeight = 0;
      return;
    }
    for (const [index, group] of this.expandedGroups) {
      if (!catalog.expanded.get(group.groupName))
        this.expandedGroups.delete(index);
    }
    this.cardGroups.forEach((group, index) => {
      if (catalog.expanded.get(group.groupName)) {
        this.expandedGroups.set(catalog.groupOffset + index, {
          groupName: group.groupName,
          totalCards: group.totalCards ?? group.items.length,
          items: [],
        });
      }
    });
    this.geometry.update(
      this.offsetHeight(catalog.totalGroups),
      this.offsetHeight(catalog.groupOffset),
      this.offsetHeight(catalog.groupOffset + this.cardGroups.length)
    );
    this.beforeHeight = this.geometry.before;
    this.afterHeight = this.geometry.after;
  }

  reservedHeight(group: CatalogCardGroup) {
    if (!this.catalog) return null;
    if (!this.catalog.expanded.get(group.groupName))
      return CATALOG_GROUP_HEIGHT;
    const {pageSize, pages} = this.catalog;
    const count = Math.min(
      pageSize,
      Math.max(
        0,
        (group.totalCards ?? 0) - (pages.get(group.groupName) ?? 0) * pageSize
      )
    );
    const controls = (group.totalCards ?? 0) > pageSize ? 104 : 0;
    return Math.max(
      CATALOG_GROUP_HEIGHT +
        Math.ceil(count / this.columns) * CATALOG_ROW_HEIGHT +
        16 +
        controls,
      this.measuredHeights.get(group.groupName) ?? 0
    );
  }

  private offsetHeight(index: number) {
    let height = index * CATALOG_GROUP_HEIGHT;
    for (const [groupIndex, group] of this.expandedGroups) {
      if (groupIndex < index)
        height += this.reservedHeight(group)! - CATALOG_GROUP_HEIGHT;
    }
    return height;
  }

  private indexAt(position: number) {
    let low = 0;
    let high = this.catalog!.totalGroups;
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (this.offsetHeight(mid + 1) <= position) low = mid + 1;
      else high = mid;
    }
    return Math.min(low, Math.max(0, this.catalog!.totalGroups - 1));
  }

  private isAtBottom() {
    const root = this.scrollWindow?.root;
    return (
      !!root &&
      root.scrollTop > 0 &&
      root.scrollHeight - root.scrollTop - root.clientHeight <= 1
    );
  }

  private updateViewport(measure: boolean) {
    const catalog = this.catalog;
    if (!catalog || !this.scrollWindow) return;
    if (this.bottomAnchor) {
      this.scrollWindow.root.scrollTop =
        this.scrollWindow.root.scrollHeight -
        this.scrollWindow.root.clientHeight;
      this.bottomAnchor = false;
    }
    if (this.logicalAnchor !== null) {
      this.scrollWindow.root.scrollTop +=
        this.geometry.toPhysical(this.logicalAnchor) -
        this.scrollWindow.bounds().top;
      this.logicalAnchor = null;
    }
    const bounds = this.scrollWindow.bounds();
    const columns = catalogGridColumns(bounds.width, catalog.cardMinWidth);
    const columnsChanged = columns !== this.columns;
    let geometryChanged = columnsChanged;
    if (columnsChanged) {
      this.columns = columns;
      this.measuredHeights.clear();
      this.scrollWindow.invalidate();
    }
    if (measure || columnsChanged) {
      const elements =
        this.element.nativeElement.querySelectorAll<HTMLElement>('.card-group');
      const observed: HTMLElement[] = Array.from(elements);
      this.groupBounds = Array.from(elements, (element, index) => {
        const rect = element.getBoundingClientRect();
        const group = this.cardGroups[index];
        const content = element.querySelectorAll<HTMLElement>(
          '.card-grid, .group-controls'
        );
        content.forEach((part) => observed.push(part));
        if (
          !columnsChanged &&
          group?.items.length &&
          catalog.expanded.get(group.groupName) &&
          content.length
        ) {
          // Measure intrinsic loaded content, not the min-height-constrained
          // group box: explicit table/full-width changes must be able to shrink.
          // With no metadata, retain the last loaded measurement instead.
          let height = CATALOG_GROUP_HEIGHT;
          content.forEach(
            (part) => (height += part.getBoundingClientRect().height)
          );
          if (height !== this.measuredHeights.get(group.groupName)) {
            this.measuredHeights.set(group.groupName, height);
            geometryChanged = true;
          }
        }
        return {
          top: rect.top - bounds.hostTop,
          bottom: rect.bottom - bounds.hostTop,
        };
      });
      this.scrollWindow.observeContent(observed);
    }
    if (geometryChanged) {
      this.updateGeometry();
      if (this.isAtBottom()) {
        this.bottomAnchor = true;
        this.scrollWindow.invalidate();
      }
    }
    const bufferGroups = Math.ceil(bounds.height / CATALOG_GROUP_HEIGHT);
    const groupLimit = Math.max(60, bufferGroups * 3 + 40);
    const atBottom = this.isAtBottom();
    const renderedFirst = this.groupBounds.findIndex(
      (rect) => rect.top <= bounds.top && rect.bottom > bounds.top
    );
    const first =
      renderedFirst >= 0
        ? catalog.groupOffset + renderedFirst
        : this.indexAt(this.geometry.toLogical(bounds.top));
    const groupOffset =
      bounds.bottom + bounds.height > 0
        ? atBottom
          ? Math.ceil(Math.max(0, catalog.totalGroups - groupLimit) / 20) * 20
          : Math.floor(Math.max(0, first - bufferGroups) / 20) * 20
        : catalog.viewport.groupOffset;
    // Use actual rendered bounds, not accumulated unloaded estimates. Cache
    // their host-relative positions until layout changes, so scroll-only frames
    // do not query every category or read every category's geometry.
    const viewportTop = bounds.rootTop - bounds.hostTop;
    const retainedGroups = new Set(catalog.viewport.visibleGroups);
    const visibleGroups = this.cardGroups
      .filter((group, index) => {
        const rect = this.groupBounds[index];
        const margin =
          bounds.height *
          (retainedGroups.has(group.groupName) ? CARD_RETENTION_VIEWPORTS : 1);
        return (
          rect &&
          catalog.expanded.get(group.groupName) &&
          rect.top < bounds.bottom + margin &&
          rect.bottom > viewportTop - margin
        );
      })
      .map((group) => group.groupName);
    const viewport = {
      ...catalog.viewport,
      groupOffset,
      groupLimit,
      visibleGroups,
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

  trackByGroup(index: number, group: CatalogCardGroup) {
    return group.groupName;
  }
}
