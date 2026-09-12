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
  Directive,
  ElementRef,
  Input,
  NgZone,
  OnDestroy,
  OnInit,
} from '@angular/core';
import {Store} from '@ngrx/store';
import {State} from '../../../app_state';
import {ElementId, nextElementId} from '../../../util/dom';
import * as actions from '../../actions';
import {CardId} from '../../types';

const elementToIds = new WeakMap<
  Element,
  {elementId: ElementId; cardId: CardId}
>();

type CardObserverCallback = (
  enteredCards: Set<Element>,
  exitedCards: Set<Element>
) => void;

export const CARD_RETENTION_VIEWPORTS = 4;

export class CardObserver {
  private intersectionObserver?: IntersectionObserver;
  private retentionObserver?: IntersectionObserver;
  private intersectionCallback?: CardObserverCallback;
  private readonly removedTargets = new WeakSet<Element>();
  private readonly targets = new Set<Element>();
  private readonly enteredTargets = new Set<Element>();
  private readonly transitionTimes = new WeakMap<Element, number>();
  private resizeObserver?: ResizeObserver;
  private rootMargin = '';
  private destroyed = false;

  /**
   * Buffer determines how far a card can be, beyond the root's bounding rect,
   * and still be loaded. It corresponds to an IntersectionObserver's
   * 'rootMargin'. For example, "50px 0 100px 0"' will treat observed elements
   * as 'intersecting' when they come within 50px of the root top or within
   * 100px of the root's bottom. Adding buffer allows nearby, offscreen cards
   * to load, preventing blank cards from being seen too often.
   * A numeric buffer is measured in root viewport heights and follows resizes.
   *
   * If positive 'rootMargin' is provided, a scrollable 'root' is required.
   *
   * https://w3c.github.io/IntersectionObserver/#dom-intersectionobserverinit-rootmargin
   * https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API
   */
  constructor(
    private readonly root?: Element,
    private readonly buffer?: string | number,
    private readonly zone?: NgZone,
    private readonly retentionBuffer?: number
  ) {}

  initialize(intersectionCallback: CardObserverCallback) {
    if (this.intersectionObserver) {
      return;
    }
    this.intersectionCallback = intersectionCallback;

    const initialize = () => {
      this.updateObserver();
      if (typeof this.buffer === 'number' && this.root) {
        this.resizeObserver = new ResizeObserver(() => this.updateObserver());
        this.resizeObserver.observe(this.root);
      }
    };
    if (this.zone) this.zone.runOutsideAngular(initialize);
    else initialize();
  }

  private updateObserver() {
    const rootMargin =
      typeof this.buffer === 'number'
        ? `${(this.root?.clientHeight ?? 0) * this.buffer}px 0px`
        : this.buffer ?? '0px';
    if (this.intersectionObserver && rootMargin === this.rootMargin) return;
    this.rootMargin = rootMargin;
    this.intersectionObserver?.disconnect();
    this.intersectionObserver = new IntersectionObserver(
      (entries) =>
        this.onCardIntersection(
          entries,
          this.retentionBuffer === undefined ? 'both' : 'enter'
        ),
      {
        // Report edge-touch -> positive-area transitions.
        threshold: Number.EPSILON,
        root: this.root ?? null,
        rootMargin,
      }
    );
    this.retentionObserver?.disconnect();
    if (this.retentionBuffer !== undefined) {
      // Enter near the viewport, leave farther away. Reversing scroll direction
      // should reuse mounted charts rather than rebuild their tables.
      this.retentionObserver = new IntersectionObserver(
        (entries) => this.onCardIntersection(entries, 'exit'),
        {
          root: this.root ?? null,
          threshold: Number.EPSILON,
          rootMargin: `${
            (this.root?.clientHeight ?? 0) * this.retentionBuffer
          }px 0px`,
        }
      );
    }
    this.targets.forEach((target) => {
      this.intersectionObserver!.observe(target);
      this.retentionObserver?.observe(target);
    });
  }

  destroy() {
    this.destroyed = true;
    this.resizeObserver?.disconnect();
    this.intersectionObserver?.disconnect();
    this.retentionObserver?.disconnect();
    this.targets.clear();
    this.enteredTargets.clear();
  }

  add(target: Element) {
    if (this.ensureInitialized()) {
      this.removedTargets.delete(target);
      this.targets.add(target);
      this.intersectionObserver!.observe(target);
      this.retentionObserver?.observe(target);
    }
  }

  remove(target: Element) {
    if (this.ensureInitialized()) {
      this.removedTargets.add(target);
      this.targets.delete(target);
      this.enteredTargets.delete(target);
      this.intersectionObserver!.unobserve(target);
      this.retentionObserver?.unobserve(target);
    }
  }

  private ensureInitialized() {
    if (!this.intersectionObserver) {
      throw new Error('CardObserver must be initialized before use');
    }
    return true;
  }

  private onCardIntersection(
    entries: IntersectionObserverEntry[],
    mode: 'both' | 'enter' | 'exit' = 'both'
  ) {
    if (this.destroyed) return;
    // Collapse queued transitions before filtering either hysteresis boundary.
    const latest = new Map<Element, IntersectionObserverEntry>();
    for (const entry of entries) {
      const previous = latest.get(entry.target);
      if (!previous || entry.time >= previous.time)
        latest.set(entry.target, entry);
    }

    const enteredElements = new Set<Element>();
    const exitedElements = new Set<Element>();
    const exitHeights = new Map<Element, number>();
    for (const {
      isIntersecting,
      intersectionRect,
      boundingClientRect,
      target,
      time,
    } of latest.values()) {
      if (this.removedTargets.has(target)) {
        continue;
      }

      const entered =
        isIntersecting &&
        intersectionRect.width > 0 &&
        intersectionRect.height > 0;
      if ((mode === 'enter' && !entered) || (mode === 'exit' && entered))
        continue;
      if (time < (this.transitionTimes.get(target) ?? -Infinity)) continue;
      this.transitionTimes.set(target, time);
      if (entered) {
        enteredElements.add(target);
        exitedElements.delete(target);
      } else {
        enteredElements.delete(target);
        exitedElements.add(target);
        exitHeights.set(target, boundingClientRect.height);
      }
    }
    for (const target of enteredElements) {
      if (this.enteredTargets.has(target)) enteredElements.delete(target);
      else {
        this.enteredTargets.add(target);
        if (target instanceof HTMLElement)
          target.style.removeProperty('min-height');
      }
    }
    for (const target of exitedElements) {
      if (!this.enteredTargets.delete(target)) exitedElements.delete(target);
      else if (target instanceof HTMLElement) {
        // Unmounting chart contents must not collapse a row during scrolling.
        target.style.minHeight = `${exitHeights.get(target) ?? 0}px`;
      }
    }
    if (!enteredElements.size && !exitedElements.size) return;
    const notify = () =>
      this.intersectionCallback!(enteredElements, exitedElements);
    if (this.zone) this.zone.run(notify);
    else notify();
  }

  onCardIntersectionForTest(
    entries: IntersectionObserverEntry[],
    mode: 'both' | 'enter' | 'exit' = 'both'
  ) {
    this.onCardIntersection(entries, mode);
  }
}

/**
 * A directive applied to elements that represent a card container. When the
 * element is ready to be loaded, this is responsible for marking cardId as
 * visible.
 *
 * Card container:
 *
 * <div [cardLazyLoader]="card1"></div>
 *
 * Card container that can load within 100px of a scrollable element's bounding
 * box:
 *
 * <div
 *   [cardLazyLoader]="card1"
 *   [cardObserver]="new CardObserver(scrollableElement, '100px')"
 * ></div>
 */
@Directive({
  standalone: false,
  selector: '[cardLazyLoader]',
})
export class CardLazyLoader implements OnInit, OnDestroy {
  @Input('cardLazyLoader') cardId!: CardId;
  @Input() cardObserver?: CardObserver;

  constructor(
    private readonly host: ElementRef,
    private readonly store: Store<State>
  ) {}

  onCardIntersection(
    enteredElements: Set<Element>,
    exitedElements: Set<Element>
  ) {
    const enteredCards = [...enteredElements].map((element) => {
      const ids = elementToIds.get(element);
      if (!ids) {
        throw new Error(
          'A CardObserver element must have an associated element id and card id.'
        );
      }
      return {elementId: ids.elementId, cardId: ids.cardId};
    });
    const exitedCards = [...exitedElements].map((element) => {
      const ids = elementToIds.get(element);
      if (!ids) {
        throw new Error(
          'A CardObserver element must have an associated element id and card id.'
        );
      }
      return {elementId: ids.elementId, cardId: ids.cardId};
    });
    this.store.dispatch(
      actions.cardVisibilityChanged({enteredCards, exitedCards})
    );
  }

  ngOnInit() {
    const element = this.host.nativeElement;
    elementToIds.set(element, {
      elementId: nextElementId(),
      cardId: this.cardId,
    });

    if (!this.cardObserver) {
      this.cardObserver = new CardObserver();
    }
    this.cardObserver.initialize(this.onCardIntersection.bind(this));
    this.cardObserver.add(element);
  }

  ngOnDestroy() {
    if (this.cardObserver) {
      const element = this.host.nativeElement;
      const ids = elementToIds.get(element);
      if (ids) {
        this.store.dispatch(
          actions.cardVisibilityChanged({
            enteredCards: [],
            exitedCards: [{elementId: ids.elementId, cardId: ids.cardId}],
          })
        );
      }
      this.cardObserver.remove(element);
    }
  }

  hostForTest() {
    return this.host;
  }
}
