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
  Component,
  EventEmitter,
  Input,
  OnChanges,
  OnDestroy,
  OnInit,
  Optional,
  Output,
  SimpleChanges,
} from '@angular/core';
import {PluginType} from '../../data_source';
import {CardId} from '../../types';
import {holdInViewport} from '../../../util/dom';
import {CardObserver} from '../card_renderer/card_lazy_loader';
import {CardIdWithMetadata} from '../metrics_view_types';

const MIN_CARD_MIN_WIDTH_IN_PX = 335;
const MAX_CARD_MIN_WIDTH_IN_PX = 735;

/**
 * Card state the grid renders, scoped to the cards it is currently showing.
 *
 * The grid deliberately does not take the whole card state map: that map gets
 * a new identity on every card state change, including one per mousemove
 * while any chart is panned, which would mark every tag group's grid dirty.
 */
export interface CardGridSizing {
  fullWidth: ReadonlySet<CardId>;
  tableExpanded: ReadonlySet<CardId>;
}

@Component({
  standalone: false,
  selector: 'metrics-card-grid-component',
  templateUrl: './card_grid_component.ng.html',
  styleUrls: ['./card_grid_component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CardGridComponent implements OnInit, OnChanges, OnDestroy {
  readonly PluginType = PluginType;
  gridTemplateColumn = '';

  cardsAtFullWidth = new Set<CardId>();
  cardsAtFullHeight = new Set<CardId>();

  @Input() isGroupExpanded!: boolean;
  @Input() pageIndex!: number;
  @Input() numPages!: number;
  @Input() cardIdsWithMetadata!: CardIdWithMetadata[];
  @Input() pendingCards: number | null = 0;
  @Input() cardMinWidth!: number | null;
  @Input() cardObserver!: CardObserver;
  @Input() showPaginationControls!: boolean;
  @Input() cardSizing!: CardGridSizing;
  @Input() groupName: string | null = null;

  @Output() pageIndexChanged = new EventEmitter<number>();

  placeholders: readonly null[] = [];
  private releasePagingControl: (() => void) | null = null;

  constructor(
    @Optional() private readonly cdkScrollable: CdkScrollable | null
  ) {}

  ngOnInit() {
    if (this.isCardWidthValid(this.cardMinWidth)) {
      this.gridTemplateColumn = `repeat(auto-fill, minmax(${this.cardMinWidth}px, 1fr))`;
    }
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['pendingCards']) {
      this.placeholders = Array<null>(this.pendingCards ?? 0).fill(null);
    }
    if (changes['cardMinWidth']) {
      const newCardWidth = changes['cardMinWidth'].currentValue;
      if (this.isCardWidthValid(newCardWidth)) {
        this.cardMinWidth = newCardWidth;
        this.gridTemplateColumn = `repeat(auto-fill, minmax(${this.cardMinWidth}px, 1fr))`;
      } else {
        this.gridTemplateColumn = '';
      }
    }
  }

  isCardWidthValid(cardMinWidth: number | null) {
    return (
      cardMinWidth &&
      cardMinWidth >= MIN_CARD_MIN_WIDTH_IN_PX &&
      cardMinWidth <= MAX_CARD_MIN_WIDTH_IN_PX
    );
  }

  showPaginationInput(isBottomControl: boolean) {
    return isBottomControl;
  }

  handlePageChange(pageIndex: number, target: HTMLElement) {
    // The page being left and the one arriving differ in height, and the new
    // cards keep resizing as they load. Hold the control that was used so it
    // can be used again without chasing it. The page input also reports the
    // page it is already on, when it loses focus to a click elsewhere.
    if (pageIndex !== this.pageIndex) {
      this.releasePagingControl?.();
      const scroller = this.cdkScrollable?.getElementRef().nativeElement;
      this.releasePagingControl = scroller
        ? holdInViewport(scroller, target)
        : null;
    }

    this.pageIndexChanged.emit(pageIndex);
  }

  ngOnDestroy() {
    this.releasePagingControl?.();
  }

  trackByCards(index: number, cardIdWithMetadata: CardIdWithMetadata) {
    return cardIdWithMetadata.cardId;
  }

  onPaginationInputChange(event: Event) {
    const input = event.target as HTMLInputElement;

    if (event.type === 'input' && input.value === '') {
      return;
    }

    const currentValue = Number(input.value) - 1;
    const nextValue = Math.min(Math.max(0, currentValue), this.numPages - 1);

    // Rectifying logic is at the container but the rectified value does not
    // appropriately make changes to the `<input>`.
    // Speculation: Angular seems to have some check on the template level that
    // does shallow equals and prevent changing on the same value.
    // Evidence: pageIndex change does fire, but `ngOnChanges` does not get
    // triggered when the value is the same (rectified value).
    if (input.value !== String(nextValue + 1)) {
      input.value = String(nextValue + 1);
    }

    this.handlePageChange(nextValue, input);
  }

  onFullWidthChanged(cardId: CardId, showFullWidth: boolean) {
    if (showFullWidth) {
      this.cardsAtFullWidth.add(cardId);
    } else {
      this.cardsAtFullWidth.delete(cardId);
    }
  }

  onFullHeightChanged(cardId: CardId, showFullHeight: boolean) {
    if (showFullHeight) {
      this.cardsAtFullHeight.add(cardId);
    } else {
      this.cardsAtFullHeight.delete(cardId);
    }
  }
}
