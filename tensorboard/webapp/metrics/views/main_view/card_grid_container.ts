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
  Input,
  OnChanges,
  OnDestroy,
  SimpleChanges,
} from '@angular/core';
import {Store} from '@ngrx/store';
import {BehaviorSubject, combineLatest, Observable, of, Subject} from 'rxjs';
import {
  distinctUntilChanged,
  map,
  shareReplay,
  switchMap,
  takeUntil,
  tap,
} from 'rxjs/operators';
import {State} from '../../../app_state';
import * as selectors from '../../../selectors';
import {
  getMetricsCardMinWidth,
  getMetricsTagGroupPageIndex,
  getMetricsTagGroupExpansionState,
} from '../../../selectors';
import {selectors as settingsSelectors} from '../../../settings';
import {CardObserver} from '../card_renderer/card_lazy_loader';
import {CardIdWithMetadata} from '../metrics_view_types';
import {metricsTagGroupPageIndexChanged} from '../../actions';
import {CardId} from '../../types';
import {CardGridSizing} from './card_grid_component';

function areSetsEqual(a: ReadonlySet<CardId>, b: ReadonlySet<CardId>): boolean {
  if (a.size !== b.size) {
    return false;
  }
  for (const cardId of a) {
    if (!b.has(cardId)) {
      return false;
    }
  }
  return true;
}

@Component({
  standalone: false,
  selector: 'metrics-card-grid',
  template: `
    <metrics-card-grid-component
      [isGroupExpanded]="isGroupExpanded$ | async"
      [pageIndex]="normalizedPageIndex$ | async"
      [numPages]="numPages$ | async"
      [showPaginationControls]="showPaginationControls$ | async"
      [cardIdsWithMetadata]="pagedItems$ | async"
      [cardMinWidth]="cardMinWidth$ | async"
      [cardObserver]="cardObserver"
      [cardSizing]="cardSizing$ | async"
      [groupName]="groupName"
      (pageIndexChanged)="onPageIndexChanged($event)"
    >
    </metrics-card-grid-component>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CardGridContainer implements OnChanges, OnDestroy {
  // groupName must be non-null if the group should be collapse/expand-able.
  @Input() groupName: string | null = null;
  @Input() cardIdsWithMetadata!: CardIdWithMetadata[];
  @Input() cardObserver!: CardObserver;
  /** Exact server page: never slice this list a second time. */
  @Input() serverTotalCards: number | null = null;
  @Input() virtualWindow = false;

  private readonly groupName$ = new BehaviorSubject<string | null>(null);
  private readonly localPageIndex$ = new BehaviorSubject<number>(0);
  private readonly items$ = new BehaviorSubject<CardIdWithMetadata[]>([]);
  private readonly serverTotalCards$ = new BehaviorSubject<number | null>(null);
  private readonly virtualWindow$ = new BehaviorSubject(false);
  private readonly ngUnsubscribe = new Subject<void>();
  readonly cardSizing$: Observable<CardGridSizing>;

  readonly numPages$;

  readonly isGroupExpanded$: Observable<boolean>;

  readonly showPaginationControls$: Observable<boolean>;

  readonly pageIndex$;

  readonly normalizedPageIndex$;

  readonly pagedItems$;

  readonly cardMinWidth$;

  constructor(private readonly store: Store<State>) {
    this.numPages$ = combineLatest([
      this.items$,
      this.store.select(settingsSelectors.getPageSize),
      this.serverTotalCards$,
    ]).pipe(
      map(([items, pageSize, total]) => {
        return Math.ceil((total ?? items.length) / pageSize);
      })
    );
    this.isGroupExpanded$ = this.groupName$.pipe(
      switchMap((groupName) => {
        return groupName !== null
          ? this.store.select(getMetricsTagGroupExpansionState, groupName)
          : of(true);
      })
    );
    this.showPaginationControls$ = combineLatest([
      this.numPages$,
      this.virtualWindow$,
    ]).pipe(map(([numPages, virtualWindow]) => !virtualWindow && numPages > 1));
    this.pageIndex$ = this.groupName$.pipe(
      switchMap((groupName) => {
        return groupName !== null
          ? this.store.select(getMetricsTagGroupPageIndex, groupName)
          : this.localPageIndex$;
      })
    );
    this.normalizedPageIndex$ = combineLatest([
      this.pageIndex$,
      this.numPages$,
      this.groupName$,
    ]).pipe(
      takeUntil(this.ngUnsubscribe),
      tap(([pageIndex, numPages, groupName]) => {
        // Cycle in the Observable but only loops when pageIndex is not
        // valid and does not repeat more than once.
        if (numPages === 0) {
          return;
        }
        if (pageIndex >= numPages) {
          this.setPageIndex(groupName, numPages - 1);
        } else if (pageIndex < 0) {
          this.setPageIndex(groupName, 0);
        }
      }),
      map(([pageIndex, numPages]) => {
        return Math.min(Math.max(pageIndex, 0), numPages - 1);
      }),
      shareReplay(1)
    );
    this.pagedItems$ = combineLatest([
      this.items$,
      this.store.select(settingsSelectors.getPageSize),
      this.normalizedPageIndex$,
      this.isGroupExpanded$,
      this.serverTotalCards$,
      this.virtualWindow$,
    ]).pipe(
      map(([items, pageSize, pageIndex, expanded, total, virtualWindow]) => {
        if (!expanded) return [];
        if (total !== null || virtualWindow) return items;
        const startIndex = pageSize * pageIndex;
        const endIndex = pageSize * pageIndex + (expanded ? pageSize : 0);
        return items.slice(startIndex, endIndex);
      })
    );
    // Scoped to the cards on this page: the whole card state map changes
    // identity on every card state change, e.g. once per mousemove while any
    // chart is panned, and would mark every group's grid dirty.
    this.cardSizing$ = combineLatest([
      this.pagedItems$,
      this.store.select(selectors.getCardStateMap),
    ]).pipe(
      map(([items, cardStateMap]) => {
        const fullWidth = new Set<CardId>();
        const tableExpanded = new Set<CardId>();
        for (const {cardId} of items) {
          const cardState = cardStateMap[cardId];
          if (!cardState) {
            continue;
          }
          if (cardState.fullWidth) {
            fullWidth.add(cardId);
          }
          if (cardState.tableExpanded) {
            tableExpanded.add(cardId);
          }
        }
        return {fullWidth, tableExpanded};
      }),
      distinctUntilChanged(
        (before, after) =>
          areSetsEqual(before.fullWidth, after.fullWidth) &&
          areSetsEqual(before.tableExpanded, after.tableExpanded)
      )
    );
    this.cardMinWidth$ = this.store.select(getMetricsCardMinWidth);
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['serverTotalCards']) {
      this.serverTotalCards$.next(this.serverTotalCards);
    }
    if (changes['virtualWindow']) {
      this.virtualWindow$.next(this.virtualWindow);
    }
    if (changes['cardIdsWithMetadata']) {
      this.items$.next(this.cardIdsWithMetadata);
    }

    if (changes['groupName']) {
      this.groupName$.next(this.groupName);
    }
  }

  ngOnDestroy() {
    this.ngUnsubscribe.next();
    this.ngUnsubscribe.complete();
  }

  onPageIndexChanged(newIndex: number) {
    this.setPageIndex(this.groupName$.value, newIndex);
  }

  private setPageIndex(groupName: string | null, pageIndex: number) {
    if (groupName !== null) {
      this.store.dispatch(
        metricsTagGroupPageIndexChanged({tagGroup: groupName, pageIndex})
      );
      return;
    }
    this.localPageIndex$.next(pageIndex);
  }
}
