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
import {ChangeDetectionStrategy, Component, Input} from '@angular/core';
import {Store} from '@ngrx/store';
import {combineLatest} from 'rxjs';
import {map} from 'rxjs/operators';
import {State} from '../../../app_state';
import {selectors as settingsSelectors} from '../../../settings';
import {metricsCatalogViewportChanged} from '../../actions';
import {
  getMetricsFilteredPluginTypes,
  getMetricsCatalogEnabled,
  getMetricsCatalogGroups,
  getMetricsCatalogTotalGroups,
  getMetricsCatalogGroupOffset,
  getMetricsCatalogViewport,
  getMetricsTagGroupExpandedMap,
  getMetricsTagGroupPageIndexMap,
  getMetricsCardMinWidth,
} from '../../store';
import {groupCardIdWithMetdata} from '../../utils';
import {CardObserver} from '../card_renderer/card_lazy_loader';
import {MetricsCatalogViewport} from '../../data_source';
import {
  getSortedRenderableCardIdsWithMetadata,
  getCatalogCardIdsWithMetadata,
  getCatalogViewScope,
} from './common_selectors';

@Component({
  standalone: false,
  selector: 'metrics-card-groups',
  template: `
    <metrics-card-groups-component
      *ngIf="view$ | async as view"
      [cardGroups]="view.groups"
      [catalog]="view.catalog"
      [cardObserver]="cardObserver"
      (viewportChanged)="onViewportChanged($event)"
    ></metrics-card-groups-component>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CardGroupsContainer {
  @Input() cardObserver!: CardObserver;

  readonly view$;

  constructor(private readonly store: Store<State>) {
    this.view$ = combineLatest({
      cards: this.store.select(getSortedRenderableCardIdsWithMetadata),
      catalogCards: this.store.select(getCatalogCardIdsWithMetadata),
      plugins: this.store.select(getMetricsFilteredPluginTypes),
      enabled: this.store.select(getMetricsCatalogEnabled),
      summaries: this.store.select(getMetricsCatalogGroups),
      groupOffset: this.store.select(getMetricsCatalogGroupOffset),
      totalGroups: this.store.select(getMetricsCatalogTotalGroups),
      viewport: this.store.select(getMetricsCatalogViewport),
      expanded: this.store.select(getMetricsTagGroupExpandedMap),
      pages: this.store.select(getMetricsTagGroupPageIndexMap),
      pageSize: this.store.select(settingsSelectors.getPageSize),
      cardMinWidth: this.store.select(getMetricsCardMinWidth),
      scope: this.store.select(getCatalogViewScope),
    }).pipe(
      map((view) => {
        if (!view.enabled) {
          const cards = view.plugins.size
            ? view.cards.filter((card) => view.plugins.has(card.plugin))
            : view.cards;
          return {groups: groupCardIdWithMetdata(cards), catalog: null};
        }
        const byGroup = new Map(
          groupCardIdWithMetdata(view.catalogCards).map((group) => [
            group.groupName,
            group.items,
          ])
        );
        return {
          groups: view.summaries.map((group) => ({
            groupName: group.name,
            totalCards: group.totalCards,
            items: byGroup.get(group.name) ?? [],
          })),
          catalog: {
            groupOffset: view.groupOffset,
            totalGroups: view.totalGroups,
            viewport: view.viewport,
            expanded: view.expanded,
            pages: view.pages,
            pageSize: view.pageSize,
            cardMinWidth: view.cardMinWidth,
            scope: view.scope,
          },
        };
      })
    );
  }

  onViewportChanged(viewport: MetricsCatalogViewport) {
    this.store.dispatch(metricsCatalogViewportChanged(viewport));
  }
}
